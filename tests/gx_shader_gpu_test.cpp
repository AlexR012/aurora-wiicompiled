#include <gtest/gtest.h>
#include "gx/ubershader.hpp"
#include "gx/shader_info.hpp"
#include "gx/frame_interpolation.hpp"
#include "gx/fifo.hpp"
#include "gfx/frame.hpp"
#include "gfx/recording.hpp"
#include "gfx/resources.hpp"
#include "gfx/render_worker.hpp"
#include "gfx/resource_cache.hpp"
#include "gfx/texture.hpp"
#include "webgpu/gpu.hpp"
#include "dolphin/vi/vi_internal.hpp"
#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <condition_variable>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <future>
#include <mutex>
#include <span>
#include <iostream>
#include <png.h>
#include <thread>

namespace {
using namespace aurora;
using namespace std::chrono_literals;
constexpr u64 TimeoutNs = 10'000'000'000;
constexpr auto ColorFormat = wgpu::TextureFormat::RGBA8Unorm;
constexpr auto DepthFormat = wgpu::TextureFormat::Depth32Float;
constexpr auto Clear = Vec4<float>{0.03125f, 0.0625f, 0.09375f, 0.125f};
std::atomic_bool gpuError = false;
u32 width = 64, height = 48;
std::filesystem::path artifacts;

struct Fixture {
  gx::PipelineConfig config{};
  std::vector<u8> vertices;
  std::vector<u8> attributes;
  std::array<u32, gx::MaxIndexAttr> arrays{};
  u32 verticesCount = 3;
  u32 instances = 1;
  bool texture = false;
  bool mipTexture = false;
  bool arbitraryMips = false;
  bool linearTexture = false;
  float minLod = 0, maxLod = 3, lodBias = 0;
  Vec4<float> tevColor{0.17f,0.36f,0.7f,0.8f};
  std::array<Vec4<float>,GX_MAX_KCOLOR> kcolors{{{0.23f,0.48f,0.71f,0.37f}}};
  bool persistent = false;
  bool interpolation = false;
  bool production = false;
  bool expectedVisible = true;
  u32 zBias = 0x23456;
  size_t metadataPrefix = 0;
  float translate = 0;
};
void be32(std::vector<u8>& out, u32 value) {
  for (int shift : {24,16,8,0}) out.push_back(u8(value >> shift));
}
void f32be(std::vector<u8>& out, float value) { be32(out, std::bit_cast<u32>(value)); }
Fixture triangle() {
  Fixture f;
  auto& c = f.config;
  c.depthFunc = GX_LEQUAL; c.cullMode = GX_CULL_NONE; c.blendMode = GX_BM_NONE;
  c.blendFacSrc = GX_BL_ONE; c.blendFacDst = GX_BL_ZERO; c.blendOp = GX_LO_CLEAR;
  c.dstAlpha = UINT32_MAX;
  c.depthCompare = c.depthUpdate = c.colorUpdate = c.alphaUpdate = true;
  auto& shader = c.shaderConfig;
  shader.tevStageCount = 1;
  shader.tevStages[0].colorPass.d = GX_CC_C0;
  shader.tevStages[0].alphaPass.d = GX_CA_A0;
  shader.attrs[GX_VA_POS] = {.attrType=GX_DIRECT, .cnt=3, .compType=GX_F32, .le=false};
  shader.vtxStride = 12;
  for (auto xy : {Vec2<float>{-0.9f,-0.8f}, Vec2<float>{0.9f,-0.8f}, Vec2<float>{0.0f,0.9f}}) {
    f32be(f.vertices,xy.x); f32be(f.vertices,xy.y); f32be(f.vertices,0.25f);
  }
  return f;
}
void textured(Fixture& f) {
  f.texture = true;
  auto& c = f.config.shaderConfig;
  c.attrs[GX_VA_TEX0] = {.attrType=GX_DIRECT,.cnt=2,.compType=GX_F32,.offset=12,.le=false};
  c.vtxStride = 20;
  c.tcgs[0].src = GX_TG_TEX0;
  c.tevStages[0].texMapId = GX_TEXMAP0; c.tevStages[0].texCoordId = GX_TEXCOORD0;
  c.tevStages[0].colorPass.d = GX_CC_TEXC; c.tevStages[0].alphaPass.d = GX_CA_TEXA;
  const auto positions = f.vertices; f.vertices.clear();
  for (u32 i=0;i<3;++i) {
    f.vertices.insert(f.vertices.end(),positions.begin()+i*12,positions.begin()+(i+1)*12);
    f32be(f.vertices, i == 1 ? 1.1371f : -0.0373f);
    f32be(f.vertices, i == 2 ? 1.0571f : 0.0021f);
  }
}
constexpr std::array<std::array<u8,4>,4> MipColors{{
  {224,32,64,192}, {36,208,60,144}, {232,136,28,112}, {156,44,208,80},
}};
constexpr std::array<u8,4> MipRight{32,96,192,128};
Fixture mip_triangle(float u, float v) {
  auto f=triangle();textured(f);f.mipTexture=true;
  f.vertices.clear();
  for(auto xy:{Vec2<float>{-0.9f,-0.8f},Vec2<float>{0.9f,-0.8f},Vec2<float>{0,0.9f}}) {
    f32be(f.vertices,xy.x);f32be(f.vertices,xy.y);f32be(f.vertices,0.25f);
    f32be(f.vertices,u);f32be(f.vertices,v);
  }
  return f;
}
webgpu::TextureWithSampler target(wgpu::TextureFormat format, const char* label, u32 samples=1) {
  const wgpu::TextureDescriptor desc{
    .label=label,
    .usage=samples==1 ? (wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::CopySrc |
           wgpu::TextureUsage::CopyDst | wgpu::TextureUsage::TextureBinding)
           : (wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::TextureBinding),
    .size={width,height,1}, .format=format, .sampleCount=samples,
  };
  auto texture = webgpu::g_device.CreateTexture(&desc);
  auto view = texture.CreateView();
  return {.texture=std::move(texture),.view=std::move(view),.size={width,height,1},.format=format};
}
struct Image { std::vector<u8> color, normal, depth; std::vector<Image> replay; };
void expect_pixel(const Image& image,u32 x,u32 y,const std::array<u8,4>& expected) {
  ASSERT_EQ(image.color.size(),size_t(width)*height*4);
  const auto offset=(size_t(y)*width+x)*4;
  for(u32 channel=0;channel<4;++channel)
    EXPECT_EQ(image.color[offset+channel],expected[channel])<<x<<","<<y<<" channel "<<channel;
}
float depth_at(const Image& image,u32 x,u32 y,u32 sample=0) {
  const auto offset=((size_t(sample)*height+y)*width+x)*4;
  if(offset+4>image.depth.size()) { ADD_FAILURE()<<"missing depth sample "<<sample;return -1; }
  float value;std::memcpy(&value,image.depth.data()+offset,4);return value;
}
struct CaptureTicket {
  wgpu::Buffer read;
  std::future<void> submitted;
  u32 pitch=0,plane=0,depthSamples=1;
  bool normal=false;
  std::shared_ptr<u32> replaySlots;
};
void save(const std::string& name, const std::vector<u8>& data) {
  if (artifacts.empty()) return;
  std::ofstream out(artifacts / name,std::ios::binary);
  out.write(reinterpret_cast<const char*>(data.data()),std::streamsize(data.size()));
  if(name.find("rgba")!=std::string::npos && data.size()==width*height*4) {
    png_image image{};image.version=PNG_IMAGE_VERSION;image.width=width;image.height=height;image.format=PNG_FORMAT_RGBA;
    auto pngPath=artifacts/name;pngPath.replace_extension("png");
    EXPECT_TRUE(png_image_write_to_file(&image,pngPath.string().c_str(),0,data.data(),0,nullptr));
    png_image_free(&image);
  }
}
bool same(const Image& a,const Image& b) { return a.color==b.color && a.normal==b.normal && a.depth==b.depth; }
void compare(const Image& reference,const Image& actual,const std::string& name) {
  EXPECT_FALSE(reference.color.empty());
  const auto check = [&](const auto& a,const auto& b,const char* attachment) {
    ASSERT_EQ(a.size(),b.size());
    size_t mismatches=0,first=0; int maxDelta=0;
    for(size_t i=0;i<a.size();++i) if(a[i]!=b[i]) {
      if(!mismatches) first=i; ++mismatches;
      maxDelta=std::max(maxDelta,std::abs(int(a[i])-int(b[i])));
    }
    if(mismatches) {
      save(name+"-"+attachment+"-reference.bin",a); save(name+"-"+attachment+"-actual.bin",b);
      std::vector<u8> diff(a.size()); for(size_t i=0;i<a.size();++i) diff[i]=u8(std::abs(int(a[i])-int(b[i])));
      save(name+"-"+attachment+"-diff.bin",diff);
    }
    EXPECT_EQ(mismatches,0u) << name << " " << attachment << " max delta " << maxDelta
      << " first pixel " << (first/4)%width << "," << ((first/4)/width)%height
      << " sample " << (first/4)/(size_t(width)*height) << " byte " << first%4;
  };
  check(reference.color,actual.color,"rgba"); check(reference.normal,actual.normal,"normal"); check(reference.depth,actual.depth,"depth32");
  ASSERT_EQ(reference.replay.size(),actual.replay.size());
  for(size_t slot=0;slot<reference.replay.size();++slot) compare(reference.replay[slot],actual.replay[slot],name+"-slot-"+std::to_string(slot));
}
struct Gate {
  struct State {
    std::mutex mutex;std::condition_variable cv;
    gfx::PipelineRef target=0;
    bool entered=false,released=false,timedOut=false,failGeneric=false,failSpecialized=false,all=false;
  };
  std::shared_ptr<State> state=std::make_shared<State>();
  static inline std::shared_ptr<State> active;
  Gate() { std::atomic_store(&active,state);gfx::testing::set_pipeline_create_hook(hook); }
  ~Gate() { release();gfx::testing::set_pipeline_create_hook(nullptr);std::atomic_store(&active,std::shared_ptr<State>{}); }
  static bool hook(gfx::PipelineRef ref,bool generic) {
    const auto gate=std::atomic_load(&active);
    if(!gate) return true;
    std::unique_lock lock(gate->mutex);
    if(generic) return !gate->failGeneric;
    if(ref!=gate->target && !gate->all) return !gate->failSpecialized;
    gate->entered=true;gate->cv.notify_all();
    if(!gate->cv.wait_for(lock,10s,[&]{return gate->released;})) gate->timedOut=true;
    return !gate->failSpecialized;
  }
  void target(gfx::PipelineRef ref) { std::lock_guard lock(state->mutex);state->target=ref; }
  void all_specializations() { std::lock_guard lock(state->mutex);state->all=true; }
  void failures(bool generic,bool specialized) { std::lock_guard lock(state->mutex);state->failGeneric=generic;state->failSpecialized=specialized; }
  void release() { std::lock_guard lock(state->mutex);state->released=true;state->cv.notify_all(); }
  bool timed_out() { std::lock_guard lock(state->mutex);return state->timedOut; }
  bool wait() { std::unique_lock lock(state->mutex);return state->cv.wait_for(lock,5s,[&]{return state->entered;}); }
  static void complete_after_prepass() {
    const auto gate=std::atomic_load(&active);
    if(!gate) { ADD_FAILURE()<<"prepass gate missing";return; }
    gfx::PipelineRef ref;
    { std::lock_guard lock(gate->mutex);ref=gate->target;gate->released=true;gate->cv.notify_all(); }
    gfx::CompiledPipeline complete;EXPECT_TRUE(gfx::get_pipeline_ready(ref,complete));
  }
};
class GxShaderGpu : public ::testing::Test {
protected:
  virtual bool limited_compatibility_device() const { return false; }
  void SetUp() override {
    gpuError=false;width=64;height=48;
    const std::string_view resolution=std::getenv("GX_SHADER_TEST_RESOLUTION")
      ? std::getenv("GX_SHADER_TEST_RESOLUTION") : "small";
    ASSERT_TRUE(resolution=="small" || resolution=="native")<<resolution;
    if(resolution=="native") { width=640;height=528; }
    const std::string_view scale=std::getenv("GX_SHADER_TEST_SCALE") ? std::getenv("GX_SHADER_TEST_SCALE") : "1";
    ASSERT_TRUE(scale=="1" || scale=="1.5")<<scale;
    if(scale=="1.5") { width=width*3/2;height=height*3/2; }
    const auto logical=vi::configured_fb_size();
    ASSERT_EQ(logical.x,640u);ASSERT_EQ(logical.y,528u);
    std::cout<<"fixture physical "<<width<<"x"<<height<<" logical "<<logical.x<<"x"<<logical.y<<'\n';
    if(const auto* dir=std::getenv("GX_SHADER_TEST_ARTIFACTS")) artifacts=dir;
    else artifacts=std::filesystem::temp_directory_path()/"aurora-shader-parity";
    std::filesystem::create_directories(artifacts);
    const std::array features{wgpu::InstanceFeatureName::TimedWaitAny};
    const wgpu::InstanceDescriptor desc{.requiredFeatureCount=features.size(),.requiredFeatures=features.data()};
    webgpu::g_instance=wgpu::CreateInstance(&desc); ASSERT_TRUE(webgpu::g_instance);
    wgpu::BackendType backend=wgpu::BackendType::Vulkan;
    if(const auto* name=std::getenv("GX_SHADER_TEST_BACKEND")) {
      if(std::string_view(name)=="d3d12") backend=wgpu::BackendType::D3D12;
      else if(std::string_view(name)=="metal") backend=wgpu::BackendType::Metal;
      else ASSERT_EQ(std::string_view(name),"vulkan");
    }
    const bool limited=limited_compatibility_device();
    const wgpu::RequestAdapterOptions options{
      .featureLevel=limited?wgpu::FeatureLevel::Compatibility:wgpu::FeatureLevel::Core,.backendType=backend};
    const auto adapter=std::make_shared<wgpu::Adapter>();
    const auto request=webgpu::g_instance.RequestAdapter(&options,wgpu::CallbackMode::WaitAnyOnly,
      [adapter](wgpu::RequestAdapterStatus status,wgpu::Adapter result,wgpu::StringView message) {
        EXPECT_EQ(status,wgpu::RequestAdapterStatus::Success) << std::string_view(message); *adapter=std::move(result);
      });
    ASSERT_EQ(webgpu::g_instance.WaitAny(request,TimeoutNs),wgpu::WaitStatus::Success); ASSERT_TRUE(*adapter);
    ASSERT_EQ(adapter->GetInfo(&webgpu::g_adapterInfo),wgpu::Status::Success);
    const auto& info=webgpu::g_adapterInfo;
    ASSERT_EQ(info.backendType,backend);
    ASSERT_NE(info.adapterType,wgpu::AdapterType::CPU); ASSERT_NE(info.backendType,wgpu::BackendType::Null);
    std::cout << "adapter " << std::string_view(info.device) << " driver " << std::string_view(info.description)
              << " vendor " << info.vendorID << " device " << info.deviceID << " backend " << int(backend) << '\n';
    webgpu::g_backendType=backend;
    wgpu::Limits supported; ASSERT_EQ(adapter->GetLimits(&supported),wgpu::Status::Success);
    ASSERT_GE(supported.maxStorageBuffersPerShaderStage,3u); ASSERT_GE(supported.maxImmediateSize,64u);
    wgpu::CompatibilityModeLimits compatibility{wgpu::CompatibilityModeLimits::Init{
      .maxStorageBuffersInVertexStage=2,.maxStorageBuffersInFragmentStage=2}};
    const wgpu::Limits limits{.nextInChain=limited?&compatibility:nullptr,
      .maxStorageBuffersPerShaderStage=limited?2u:3u,.maxImmediateSize=64};
    std::vector<wgpu::FeatureName> deviceFeatures;
    if(!limited && adapter->HasFeature(wgpu::FeatureName::CoreFeaturesAndLimits)) deviceFeatures.push_back(wgpu::FeatureName::CoreFeaturesAndLimits);
    webgpu::g_dualSourceBlendingSupported=adapter->HasFeature(wgpu::FeatureName::DualSourceBlending);
    if(webgpu::g_dualSourceBlendingSupported) deviceFeatures.push_back(wgpu::FeatureName::DualSourceBlending);
    wgpu::DeviceDescriptor deviceDesc({.requiredFeatureCount=deviceFeatures.size(),.requiredFeatures=deviceFeatures.data(),.requiredLimits=&limits});
    deviceDesc.SetUncapturedErrorCallback([](const wgpu::Device&,wgpu::ErrorType type,wgpu::StringView message) {
      gpuError=true; ADD_FAILURE()<<"Dawn error "<<int(type)<<": "<<std::string_view(message);
    });
    deviceDesc.SetDeviceLostCallback(wgpu::CallbackMode::AllowSpontaneous,
      [](const wgpu::Device&,wgpu::DeviceLostReason reason,wgpu::StringView message) {
        if(reason!=wgpu::DeviceLostReason::Destroyed) { gpuError=true; ADD_FAILURE()<<"Device lost "<<std::string_view(message); }
      });
    const auto deviceRequest=adapter->RequestDevice(&deviceDesc,wgpu::CallbackMode::WaitAnyOnly,
      [](wgpu::RequestDeviceStatus status,wgpu::Device device,wgpu::StringView message) {
        EXPECT_EQ(status,wgpu::RequestDeviceStatus::Success)<<std::string_view(message); webgpu::g_device=std::move(device);
      });
    ASSERT_EQ(webgpu::g_instance.WaitAny(deviceRequest,TimeoutNs),wgpu::WaitStatus::Success); ASSERT_TRUE(webgpu::g_device);
    wgpu::Limits actual;ASSERT_EQ(webgpu::g_device.GetLimits(&actual),wgpu::Status::Success);
    ASSERT_GE(actual.maxTextureDimension2D,std::max(width,height));
    ASSERT_GE(actual.maxBufferSize,u64(AURORA_ALIGN(width*4,256))*height*12);
    ASSERT_GE(actual.maxStorageBufferBindingSize,u64(AURORA_ALIGN(width*4,256))*height*4);
    ASSERT_GE(actual.maxComputeWorkgroupsPerDimension,(std::max(width,height)+7)/8);
    webgpu::g_queue=webgpu::g_device.GetQueue();
    webgpu::g_hasCoreFeatures=webgpu::g_device.HasFeature(wgpu::FeatureName::CoreFeaturesAndLimits);
    cache=artifacts/("cache-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(cache); cacheText=cache.string();
    g_config.cachePath=cacheText.c_str();
    g_config.userPath=cacheText.c_str(); g_config.resourcesPath=cacheText.c_str();
    webgpu::g_graphicsConfig.surfaceConfiguration.format=ColorFormat;
    webgpu::g_graphicsConfig.depthFormat=DepthFormat; webgpu::g_graphicsConfig.msaaSamples=1;
    webgpu::g_frameBuffer=target(ColorFormat,"Parity EFB"); webgpu::g_depthBuffer=target(DepthFormat,"Parity depth");
    webgpu::g_graphicsConfig.normalBuffer=false;
    aurora_set_shader_mode(AURORA_SHADER_HYBRID); gfx::initialize(); initialized=true;
    ASSERT_EQ(bool(gfx::detail::resources().uberStateBuffer),!limited);
    ASSERT_EQ(gfx::detail::resources().hybridResources,!limited);
    gx::fifo::init(); fifoInitialized=true;
    ASSERT_TRUE(gfx::begin_frame());
    gx::fifo::begin_frame();
    GXInit(nullptr,0);gx::fifo::drain();gx::fifo::end_frame();gfx::finish();
    gfx::end_frame([](wgpu::CommandEncoder& encoder,std::vector<gfx::AfterSubmitCallback> callbacks,const gfx::InterpolationReplay&) {
      const auto commands=encoder.Finish();webgpu::g_queue.Submit(1,&commands);
      for(auto& callback:callbacks) callback();
    });
    gfx::render_worker::synchronize();
    gx::testing::reset_hybrid_counters();
  }
  void TearDown() override {
    gx::testing::set_after_prepass_hook(nullptr);
    gx::testing::invalidate_generic_source(false);
    gx::set_frame_interpolation_fps(0);
    if(fifoInitialized) gx::fifo::shutdown();
    if(initialized) gfx::shutdown();
    webgpu::g_frameBuffer={}; webgpu::g_frameBufferResolved={}; webgpu::g_depthBuffer={}; webgpu::g_normalBuffer={};
    depthSamplePipeline={};texture.reset();mipTextures={};
    webgpu::g_queue={}; webgpu::g_device={}; webgpu::g_instance={};
    g_config.cachePath=g_config.userPath=g_config.resourcesPath=nullptr;
    aurora_set_shader_mode(AURORA_SHADER_SPECIALIZED);
    EXPECT_FALSE(gpuError.load());
  }
  void uniforms(const Fixture& f) {
    auto& state=gx::g_gxState;
    state.currentPnMtx=0;
    state.clearColor=Clear; state.clearDepth=0xFFFFFF;
    const auto logical=vi::configured_fb_size();
    state.renderViewport={0,0,float(width),float(height),0,1};
    state.logicalViewport={0,0,float(logical.x),float(logical.y),0,1};
    state.renderScissor={0,0,int(width),int(height)};
    state.logicalScissor={0,0,int(logical.x),int(logical.y)};
    state.proj={{1,0,0,0},{0,1,0,0},{0,0,-1,0},{0,0,0,1}};
    const Mat3x4<float> identity{{1,0,0,0},{0,1,0,0},{0,0,1,0}};
    for(auto& m:state.pnMtx) { m.pos=identity; m.nrm=identity; m.pos.m0[3]=f.translate; }
    for(auto& m:state.texMtxs) m=identity;
    for(auto& m:state.ptTexMtxs) m=identity;
    if(!f.production) {
      state.colorRegs[0]={-0.3f,1.2f,0.17f,0.49f};
      state.colorRegs[1]=f.tevColor;
      state.colorRegs[2]={0.83f,0.1f,0.25f,0.3f};
      state.colorRegs[3]={1.23f,-0.2f,0.8f,1.5f};
    }
    state.kcolors=f.kcolors;
    state.colorChannelConfig=f.config.shaderConfig.colorChannels;
    for(auto& cc:state.colorChannelState) { cc.matColor={0.4f,0.6f,0.8f,0.7f}; cc.ambColor={0.1f,0.2f,0.3f,0.4f}; cc.lightMask=1; }
    state.lights[0].pos={0,0,3}; state.lights[0].dir={0,0,-1}; state.lights[0].color={0.5f,0.7f,0.3f,0.6f};
    state.lights[0].cosAtt={1,0,0}; state.lights[0].distAtt={1,0,0};
    state.fog.a=0.3f; state.fog.b=1.5f; state.fog.c=0.01f; state.fog.color={0.8f,0.3f,0.1f,1};
    state.zTextureBias=f.zBias;
    for(auto& matrix:state.indTexMtxs) { matrix.mtx={{0.25f,0.125f},{-0.125f,0.25f},{0.0625f,-0.0625f}};matrix.scaleExp=1; }
    state.lineHalfAspect=false;state.lineTexOffset=GX_TO_HALF;state.pointTexOffset=GX_TO_FOURTH;
    state.lineWidth=120; state.pointSize=180;
    for(auto& scale:state.texCoordScales) { scale.scaleS=6; scale.scaleT=4; scale.lineOffset=true; scale.pointOffset=true; }
    if(f.texture && !f.mipTexture && !texture) {
      const wgpu::TextureDescriptor desc{.label="Parity 7x5 texture",.usage=wgpu::TextureUsage::TextureBinding|wgpu::TextureUsage::CopyDst,.size={7,5,1},.format=ColorFormat};
      auto tex=webgpu::g_device.CreateTexture(&desc); auto view=tex.CreateView();
      texture=std::make_shared<gfx::TextureRef>(tex,view,view,wgpu::Extent3D{7,5,1},ColorFormat,1,GX_TF_RGBA8);
      std::vector<u8> pixels(7*5*4);
      for(u32 y=0;y<5;++y) for(u32 x=0;x<7;++x) {
        const auto i=(y*7+x)*4; pixels[i]=u8(x*37);pixels[i+1]=u8(y*59);pixels[i+2]=u8(x*23+y*17);pixels[i+3]=u8(47+x*19+y*9);
      }
      const wgpu::TexelCopyTextureInfo dst{.texture=tex};
      const wgpu::TexelCopyBufferLayout bytes{.bytesPerRow=28,.rowsPerImage=5};
      const wgpu::Extent3D size{7,5,1}; webgpu::g_queue.WriteTexture(&dst,pixels.data(),pixels.size(),&bytes,&size);
    }
    if(f.mipTexture) {
      if(!mipTextures[0]) {
        const wgpu::TextureDescriptor desc{.label="Parity four distinct mips",
          .usage=wgpu::TextureUsage::TextureBinding|wgpu::TextureUsage::CopyDst,
          .size={8,8,1},.format=ColorFormat,.mipLevelCount=4};
        auto tex=webgpu::g_device.CreateTexture(&desc);auto view=tex.CreateView();
        mipTextures[0]=std::make_shared<gfx::TextureRef>(tex,view,view,wgpu::Extent3D{8,8,1},ColorFormat,4,GX_TF_RGBA8);
        for(u32 level=0;level<4;++level) {
          const u32 size=8u>>level;std::vector<u8> pixels(size*size*4);
          for(u32 y=0;y<size;++y)for(u32 x=0;x<size;++x) {
            const auto& color=level==0 && x>=size/2 ? MipRight : MipColors[level];
            std::copy(color.begin(),color.end(),pixels.begin()+(y*size+x)*4);
          }
          const wgpu::TexelCopyTextureInfo dst{.texture=tex,.mipLevel=level};
          const wgpu::TexelCopyBufferLayout bytes{.bytesPerRow=size*4,.rowsPerImage=size};
          const wgpu::Extent3D extent{size,size,1};
          webgpu::g_queue.WriteTexture(&dst,pixels.data(),pixels.size(),&bytes,&extent);
        }
        mipTextures[1]=std::make_shared<gfx::TextureRef>(*mipTextures[0]);
        mipTextures[1]->hasArbitraryMips=true;
      }
      GXTexObj object;
      GXInitTexObj(&object,nullptr,8,8,GX_TF_RGBA8,GX_CLAMP,GX_CLAMP,GX_TRUE);
      GXInitTexObjLOD(&object,f.linearTexture?GX_LIN_MIP_NEAR:GX_NEAR_MIP_NEAR,
        f.linearTexture?GX_LINEAR:GX_NEAR,f.minLod,f.maxLod,f.lodBias,GX_FALSE,GX_FALSE,GX_ANISO_1);
      state.textures[0]=gfx::TextureBind(*reinterpret_cast<const GXTexObj_*>(&object),mipTextures[f.arbitraryMips?1:0]);
    } else if(f.texture) {
      GXTexObj_ object;object.mWidth=7;object.mHeight=5;object.mFormat=GX_TF_RGBA8;
      state.textures[0]=gfx::TextureBind(object,texture);
    }
  }
  void multisample_targets() {
    gfx::render_worker::synchronize();
    webgpu::g_graphicsConfig.msaaSamples=4;
    webgpu::g_frameBuffer=target(ColorFormat,"Parity 4x EFB",4);
    webgpu::g_frameBufferResolved=target(ColorFormat,"Parity resolved EFB");
    webgpu::g_depthBuffer=target(DepthFormat,"Parity 4x depth",4);
    constexpr auto source=R"(
@group(0) @binding(0) var src: texture_depth_multisampled_2d;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
@compute @workgroup_size(8,8) fn cs_main(@builtin(global_invocation_id) id: vec3u) {
  let size=textureDimensions(src);
  if(id.x>=size.x || id.y>=size.y) { return; }
  let stride=((size.x+63u)/64u)*64u;
  for(var sample=0u;sample<4u;sample++) {
    dst[sample*stride*size.y+id.y*stride+id.x]=bitcast<u32>(textureLoad(src,vec2i(id.xy),i32(sample)));
  }
}
)";
    const wgpu::ShaderSourceWGSL wgsl{wgpu::ShaderSourceWGSL::Init{.code=source}};
    const wgpu::ShaderModuleDescriptor moduleDesc{.nextInChain=&wgsl,.label="Parity every depth sample"};
    const auto module=webgpu::g_device.CreateShaderModule(&moduleDesc);
    const wgpu::ComputePipelineDescriptor desc{.label="Parity every depth sample",.compute={.module=module,.entryPoint="cs_main"}};
    depthSamplePipeline=webgpu::g_device.CreateComputePipeline(&desc);
    ASSERT_TRUE(depthSamplePipeline);
    const auto layout=gfx::scene_render_target_layout();
    ASSERT_EQ(layout.sampleCount,4u);ASSERT_EQ(layout.colorAttachmentCount,1u);
    ASSERT_EQ(layout.depthStencilFormat,DepthFormat);
    ASSERT_EQ(webgpu::g_frameBuffer.texture.GetSampleCount(),4u);
    ASSERT_EQ(webgpu::g_depthBuffer.texture.GetSampleCount(),4u);
    ASSERT_EQ(webgpu::g_frameBufferResolved.texture.GetSampleCount(),1u);
  }
  CaptureTicket submit(Fixture f,gx::DrawShaderSelection selection,gfx::PipelineRef specialization=0,bool rejectMetadata=false) {
    const u32 samples=webgpu::g_graphicsConfig.msaaSamples;
    if(samples!=1 && (samples!=4 || f.interpolation || f.persistent || webgpu::g_normalBuffer.texture)) {
      ADD_FAILURE()<<"multisample capture requires the scene color/depth layout without replay or persistent copy";return {};
    }
    uniforms(f);
    if(!gfx::begin_frame()) { ADD_FAILURE()<<"begin_frame failed";return {}; }
    gx::fifo::begin_frame();
    if(f.metadataPrefix) {
      std::vector<u8> poison(f.metadataPrefix,0xdc);
      EXPECT_TRUE(gfx::try_push_uber_state(poison));
    }
    if(f.production) {
      gx::testing::set_shader_selection(selection==gx::DrawShaderSelection::GenericOnly
        ? gx::testing::ShaderSelection::Generic : (selection==gx::DrawShaderSelection::Hybrid
        ? gx::testing::ShaderSelection::Automatic : gx::testing::ShaderSelection::Specialized));
      GXClearVtxDesc();GXSetVtxDesc(GX_VA_POS,GX_DIRECT);
      GXSetVtxAttrFmt(GX_VTXFMT0,GX_VA_POS,GX_POS_XYZ,GX_F32,0);
      GXSetNumChans(0);GXSetNumTexGens(0);GXSetNumTevStages(1);
      GXSetTevOrder(GX_TEVSTAGE0,GX_TEXCOORD_NULL,GX_TEXMAP_NULL,GX_COLOR_NULL);
      GXSetTevColor(GX_TEVREG0,{43,92,179,204});
      GXSetTevColorIn(GX_TEVSTAGE0,GX_CC_ZERO,GX_CC_ZERO,GX_CC_ZERO,GX_CC_C0);
      GXSetTevAlphaIn(GX_TEVSTAGE0,GX_CA_ZERO,GX_CA_ZERO,GX_CA_ZERO,GX_CA_A0);
      GXSetTevColorOp(GX_TEVSTAGE0,GX_TEV_ADD,GX_TB_ZERO,GX_CS_SCALE_1,GX_TRUE,GX_TEVPREV);
      GXSetTevAlphaOp(GX_TEVSTAGE0,GX_TEV_ADD,GX_TB_ZERO,GX_CS_SCALE_1,GX_TRUE,GX_TEVPREV);
      GXSetAlphaCompare(GX_ALWAYS,0,GX_AOP_AND,GX_ALWAYS,0);
      GXSetZMode(GX_TRUE,GX_LEQUAL,GX_TRUE);GXSetCullMode(GX_CULL_NONE);
      GXSetBlendMode(GX_BM_NONE,GX_BL_ONE,GX_BL_ZERO,f.config.blendOp);
      GXSetColorUpdate(GX_TRUE);GXSetAlphaUpdate(GX_TRUE);GXSetDstAlpha(GX_FALSE,0);
      GXSetPixelFmt(GX_PF_RGBA6_Z24,GX_ZC_LINEAR);
      if(f.config.shaderConfig.zTextureOp!=GX_ZT_DISABLE) {
        for(u32 i=0;i<16;++i) {
          zTexels[i*2]=u8(47+i*9);zTexels[i*2+1]=u8(17+i*13);
          zTexels[32+i*2]=u8(29+i*7);zTexels[32+i*2+1]=u8(61+i*11);
        }
        GXTexObj object;
        GXInitTexObj(&object,zTexels.data(),4,4,GX_TF_RGBA8,GX_CLAMP,GX_CLAMP,GX_FALSE);
        GXLoadTexObj(&object,GX_TEXMAP0);
        GXSetNumTexGens(1);
        GXSetTexCoordGen2(GX_TEXCOORD0,GX_TG_MTX2x4,GX_TG_POS,GX_IDENTITY,GX_FALSE,GX_PTIDENTITY);
        GXSetTevOrder(GX_TEVSTAGE0,GX_TEXCOORD0,GX_TEXMAP0,GX_COLOR_NULL);
      }
      const std::array zFormats{GX_TF_Z8,GX_TF_Z16,GX_TF_Z24X8};
      GXSetZTexture(static_cast<GXZTexOp>(f.config.shaderConfig.zTextureOp),zFormats[f.config.shaderConfig.zTextureFormat],f.zBias);
      GXBegin(GX_TRIANGLES,GX_VTXFMT0,3);
      GXPosition3f32(-0.9f,-0.8f,0.25f);GXPosition3f32(0.9f,-0.8f,0.25f);GXPosition3f32(0,0.9f,0.25f);GXEnd();
      gx::fifo::drain();
      gx::testing::set_shader_selection(gx::testing::ShaderSelection::Automatic);
    } else {
    const auto info=gx::build_shader_info(f.config.shaderConfig); const auto layout=gfx::get_render_target_layout();
    if(!specialization && selection!=gx::DrawShaderSelection::GenericOnly) specialization=gfx::find_pipeline(f.config,layout,true);
    gfx::PipelineRef generic=0;
    if(selection==gx::DrawShaderSelection::Hybrid || selection==gx::DrawShaderSelection::GenericOnly) generic=gfx::find_uber_pipeline(f.config,layout);
    gx::DrawImmediateData immediate{};
    const auto verts=gfx::push_verts(f.vertices.data(),f.vertices.size(),4); immediate.vtxStart=verts.offset;
    if(!f.attributes.empty()) { const auto attrs=gfx::push_storage(f.attributes.data(),f.attributes.size()); for(u32 i=0;i<f.arrays.size();++i) immediate.arrayStart[i]=attrs.offset+f.arrays[i]; }
    if(f.config.shaderConfig.fogRangeEnabled) {
      std::vector<float> lut(width,1.0f);
      for(u32 x=0;x<width;++x) lut[x]=1.0f+float(x)/float(width);
      const auto range=gfx::push_storage(reinterpret_cast<const u8*>(lut.data()),lut.size()*sizeof(float));
      immediate.fogRangeBase=range.offset/4;
    }
    const auto uniform=gx::build_uniform(info);
    if(generic) {
      if(rejectMetadata) { std::vector<u8> fill(gfx::UberStateBufferSize,0xa7); EXPECT_TRUE(gfx::try_push_uber_state(fill)); }
      const auto state=gx::build_uber_shader_state(f.config.shaderConfig,info);
      const auto range=gfx::try_push_uber_state({reinterpret_cast<const u8*>(&state),sizeof(state)});
      if(!range) { EXPECT_TRUE(rejectMetadata); selection=gx::DrawShaderSelection::RequiredSpecialized;generic=0; }
      else immediate.uberStateOffset=range->offset;
    }
    gx::DrawData draw{.pipeline=specialization,.fallbackPipeline=generic,.bindGroups=gx::build_bind_groups(info),
      .vertSize=verts.size,.uniformOffset=uniform.offset,.uniformSize=u16(uniform.size),.selection=selection,
      .immediateData=immediate,.vtxCount=f.verticesCount,.indexCount=0,.instanceCount=f.instances,.dstAlpha=f.config.dstAlpha};
    if(f.config.shaderConfig.lineMode) {
      const std::array<u16,6> indices{0,1,3,3,2,0}; draw.idxRange=gfx::push_indices(reinterpret_cast<const u8*>(indices.data()),sizeof(indices),4); draw.indexCount=6;
    }
    if(f.interpolation) {
      Mat4x4<float> projection;
      std::memcpy(&projection,gfx::uniform_data(uniform)+info.offsets.projection,sizeof(projection));
      const auto ranges=gx::record_interpolation_draw({0x123456,0x123456,0x123456,0x123456},projection,1,
        {.sourceUniformData=gfx::uniform_data(uniform),.uniformSize=uniform.size,
         .projectionOffset=info.offsets.projection,.positionOffset=info.offsets.postex,.normalOffset=info.offsets.nrm,
         .currentMatrix=0,.indexedMatrices=false});
      if(ranges[0].size) draw.interpUniformBase=ranges[0].offset;
    }
    gfx::push_draw_command(draw);
    }
    gfx::TextureHandle persistent;
    if(f.persistent) {
      const auto& source=webgpu::g_frameBuffer;
      const wgpu::TextureDescriptor desc{.usage=wgpu::TextureUsage::RenderAttachment|wgpu::TextureUsage::TextureBinding|wgpu::TextureUsage::CopySrc,.size=source.size,.format=ColorFormat};
      const auto tex=webgpu::g_device.CreateTexture(&desc); const auto view=tex.CreateView();
      persistent=std::make_shared<gfx::TextureRef>(tex,view,view,source.size,ColorFormat,1,GX_TF_RGBA8);
      gfx::resolve_pass_into(persistent,{0,0,int(width),int(height)},false,false,false,{},0,GX_TF_RGBA8,GX_PF_RGB8_Z24,{.persistent=true});
    }
    gx::fifo::drain();gx::fifo::end_frame();
    gfx::finish();
    const u32 pitch=AURORA_ALIGN(width*4,256), plane=pitch*height;
    const bool normal=bool(webgpu::g_normalBuffer.texture);
    const wgpu::BufferDescriptor readDesc{.usage=wgpu::BufferUsage::CopyDst|wgpu::BufferUsage::MapRead,.size=u64(plane)*12};
    auto read=webgpu::g_device.CreateBuffer(&readDesc);
    const auto done=std::make_shared<std::promise<void>>();auto submitted=done->get_future();
    const auto replaySlots=std::make_shared<u32>(0);
    const auto depthPipeline=depthSamplePipeline;
    gfx::end_frame([read,done,persistent,pitch,plane,normal,replaySlots,samples,depthPipeline](wgpu::CommandEncoder& cmd,std::vector<gfx::AfterSubmitCallback> callbacks,const gfx::InterpolationReplay& replay) {
      const auto copy=[&](wgpu::CommandEncoder& encoder,const wgpu::Texture& source,u32 offset,wgpu::TextureAspect aspect) {
        const wgpu::TexelCopyTextureInfo src{.texture=source,.aspect=aspect};
        const wgpu::TexelCopyBufferInfo dst{.layout={.offset=offset,.bytesPerRow=pitch,.rowsPerImage=height},.buffer=read};
        const wgpu::Extent3D size{width,height,1};encoder.CopyTextureToBuffer(&src,&dst,&size);
      };
      copy(cmd,persistent?persistent->texture:(samples>1?webgpu::g_frameBufferResolved.texture:webgpu::g_frameBuffer.texture),0,wgpu::TextureAspect::All);
      if(samples==1) copy(cmd,webgpu::g_depthBuffer.texture,plane,wgpu::TextureAspect::DepthOnly);
      else {
        const wgpu::BufferDescriptor desc{.label="Parity four depth planes",
          .usage=wgpu::BufferUsage::Storage|wgpu::BufferUsage::CopySrc,.size=u64(plane)*4};
        const auto storage=webgpu::g_device.CreateBuffer(&desc);
        const std::array entries{
          wgpu::BindGroupEntry{.binding=0,.textureView=webgpu::g_depthBuffer.view},
          wgpu::BindGroupEntry{.binding=1,.buffer=storage,.size=u64(plane)*4},
        };
        const auto layout=depthPipeline.GetBindGroupLayout(0);
        const wgpu::BindGroupDescriptor bindDesc{.layout=layout,.entryCount=entries.size(),.entries=entries.data()};
        const auto bind=webgpu::g_device.CreateBindGroup(&bindDesc);
        const auto pass=cmd.BeginComputePass();pass.SetPipeline(depthPipeline);pass.SetBindGroup(0,bind);
        pass.DispatchWorkgroups((width+7)/8,(height+7)/8);pass.End();
        cmd.CopyBufferToBuffer(storage,0,read,plane,u64(plane)*4);
      }
      if(normal) copy(cmd,webgpu::g_normalBuffer.texture,plane*2,wgpu::TextureAspect::All);
      auto commands=cmd.Finish();webgpu::g_queue.Submit(1,&commands);for(auto& callback:callbacks) callback();
      *replaySlots=replay.slots;
      for(u32 slot=0;slot<replay.slots;++slot) {
        auto encoder=webgpu::g_device.CreateCommandEncoder();
        replay.encode(encoder,slot);
        const u32 offset=(slot+1)*plane*3;
        copy(encoder,webgpu::g_frameBuffer.texture,offset,wgpu::TextureAspect::All);
        copy(encoder,webgpu::g_depthBuffer.texture,offset+plane,wgpu::TextureAspect::DepthOnly);
        if(normal) copy(encoder,webgpu::g_normalBuffer.texture,offset+plane*2,wgpu::TextureAspect::All);
        auto replayCommands=encoder.Finish();webgpu::g_queue.Submit(1,&replayCommands);
      }
      done->set_value();
    });
    return {std::move(read),std::move(submitted),pitch,plane,samples,normal,replaySlots};
  }
  Image read_ticket(CaptureTicket ticket) {
    if(!ticket.submitted.valid() || ticket.submitted.wait_for(15s)!=std::future_status::ready) {
      ADD_FAILURE()<<"renderer submission timed out";return {};
    }
    const auto read=ticket.read;
    const auto pitch=ticket.pitch,plane=ticket.plane,samples=ticket.depthSamples;
    const bool normal=ticket.normal;
    const auto slots=*ticket.replaySlots;
    auto result=std::make_shared<Image>();auto mapped=std::make_shared<std::atomic_bool>(false);
    const auto map=read.MapAsync(wgpu::MapMode::Read,0,u64(plane)*12,wgpu::CallbackMode::WaitAnyOnly,
      [result,mapped,read,pitch,plane,normal,slots,samples](wgpu::MapAsyncStatus status,wgpu::StringView message) {
        EXPECT_EQ(status,wgpu::MapAsyncStatus::Success)<<std::string_view(message);
        if(status!=wgpu::MapAsyncStatus::Success) return;
        const auto* data=static_cast<const u8*>(read.GetConstMappedRange(0,u64(plane)*12));
        const auto unpack=[&](std::vector<u8>& out,u32 offset) { out.resize(width*height*4); for(u32 y=0;y<height;++y) std::memcpy(out.data()+y*width*4,data+offset+y*pitch,width*4); };
        unpack(result->color,0);
        result->depth.resize(size_t(width)*height*4*samples);
        for(u32 sample=0;sample<samples;++sample)for(u32 y=0;y<height;++y)
          std::memcpy(result->depth.data()+((size_t(sample)*height+y)*width)*4,
            data+(sample+1)*plane+y*pitch,width*4);
        if(normal) unpack(result->normal,plane*2);
        result->replay.resize(slots);
        for(u32 slot=0;slot<slots;++slot) {
          const u32 offset=(slot+1)*plane*3;
          unpack(result->replay[slot].color,offset);unpack(result->replay[slot].depth,offset+plane);
          if(normal)unpack(result->replay[slot].normal,offset+plane*2);
        }
        read.Unmap();mapped->store(true);
      });
    EXPECT_EQ(webgpu::g_instance.WaitAny(map,TimeoutNs),wgpu::WaitStatus::Success);EXPECT_TRUE(mapped->load());
    EXPECT_FALSE(gpuError.load());return *result;
  }
  Image capture(Fixture f,gx::DrawShaderSelection selection,gfx::PipelineRef specialization=0,bool rejectMetadata=false) {
    return read_ticket(submit(std::move(f),selection,specialization,rejectMetadata));
  }
  void parity(Fixture f,const std::string& name) {
    const auto reference=capture(f,gx::DrawShaderSelection::RequiredSpecialized);
    const std::array<u8,4> clear{8,16,24,32};
    bool visible=false;
    for(size_t i=0;i<reference.color.size();++i) visible |= reference.color[i]!=clear[i%4];
    EXPECT_EQ(visible,f.expectedVisible)<<name<<" reference semantic probe";
    const auto before=gx::testing::hybrid_counters();
    const auto actual=capture(f,gx::DrawShaderSelection::GenericOnly);
    EXPECT_EQ(gx::testing::hybrid_counters().generic,before.generic+1);compare(reference,actual,name);
  }
  Image probed_parity(Fixture f,const std::string& name) {
    const auto before=gx::testing::hybrid_counters();
    const auto reference=capture(f,gx::DrawShaderSelection::RequiredSpecialized);
    EXPECT_EQ(gx::testing::hybrid_counters().specialized,before.specialized+1);
    const auto info=gx::build_shader_info(f.config.shaderConfig);
    const auto coverage=gx::uber_coverage(f.config.shaderConfig,info,gfx::scene_render_target_layout());
    if(coverage!=gx::GenericCoverageReason::Covered) {
      ADD_FAILURE()<<name<<" fixture is outside verified generic coverage: "<<int(coverage);return {};
    }
    const auto actual=capture(f,gx::DrawShaderSelection::GenericOnly);
    EXPECT_EQ(gx::testing::hybrid_counters().generic,before.generic+1);
    compare(reference,actual,name);return reference;
  }
  bool initialized=false,fifoInitialized=false;
  std::filesystem::path cache;std::string cacheText;
  gfx::TextureHandle texture;
  alignas(32) std::array<u8,64> zTexels{};
  std::array<gfx::TextureHandle,2> mipTextures;
  wgpu::ComputePipeline depthSamplePipeline;
};

class GxShaderGpuCompatibility : public GxShaderGpu {
protected:
  bool limited_compatibility_device() const override { return true; }
};
TEST_F(GxShaderGpuCompatibility, LimitedVertexStorageKeepsSpecializedLayout) {
  wgpu::CompatibilityModeLimits compatibility{};
  wgpu::Limits limits{.nextInChain=&compatibility};
  ASSERT_EQ(webgpu::g_device.GetLimits(&limits),wgpu::Status::Success);
  EXPECT_FALSE(webgpu::g_hasCoreFeatures);
  EXPECT_GE(limits.maxStorageBuffersPerShaderStage,3u);
  EXPECT_EQ(compatibility.maxStorageBuffersInVertexStage,2u);
  const auto& resources=gfx::detail::resources();
  EXPECT_EQ(resources.shaderMode,AURORA_SHADER_HYBRID);
  EXPECT_FALSE(resources.hybridResources);EXPECT_FALSE(resources.uberStateBuffer);
  EXPECT_EQ(resources.limits.nextInChain,nullptr);
  auto f=triangle();f.tevColor={43.f/255,92.f/255,179.f/255,204.f/255};
  const auto reference=capture(f,gx::DrawShaderSelection::RequiredSpecialized);
  const auto before=gx::testing::hybrid_counters();
  const auto actual=capture(f,gx::DrawShaderSelection::Hybrid);
  compare(reference,actual,"compatibility-specialized-layout");
  expect_pixel(actual,width/2,height/2,{43,92,179,204});
  EXPECT_EQ(gx::testing::hybrid_counters().generic,before.generic);
  EXPECT_EQ(gx::testing::hybrid_counters().specialized,before.specialized+1);
}
TEST_F(GxShaderGpu, PhysicalResolutionKeepsLogicalViAndVisibleColorDepth) {
  const bool native=std::getenv("GX_SHADER_TEST_RESOLUTION") && std::string_view(std::getenv("GX_SHADER_TEST_RESOLUTION"))=="native";
  const bool fractional=std::getenv("GX_SHADER_TEST_SCALE") && std::string_view(std::getenv("GX_SHADER_TEST_SCALE"))=="1.5";
  EXPECT_EQ(width,native?(fractional?960u:640u):(fractional?96u:64u));
  EXPECT_EQ(height,native?(fractional?792u:528u):(fractional?72u:48u));
  EXPECT_EQ(webgpu::g_frameBuffer.texture.GetWidth(),width);
  EXPECT_EQ(webgpu::g_frameBuffer.texture.GetHeight(),height);
  auto f=triangle();f.tevColor={43.f/255,92.f/255,179.f/255,204.f/255};
  const auto image=probed_parity(f,"physical-resolution");
  EXPECT_EQ(gx::g_gxState.logicalViewport.width,640.f);
  EXPECT_EQ(gx::g_gxState.logicalViewport.height,528.f);
  EXPECT_EQ(gx::g_gxState.renderViewport.width,float(width));
  EXPECT_EQ(gx::g_gxState.renderViewport.height,float(height));
  expect_pixel(image,width/2,height/2,{43,92,179,204});
  expect_pixel(image,0,0,{8,16,24,32});
  ASSERT_EQ(image.depth.size(),size_t(width)*height*4);
  const float center=depth_at(image,width/2,height/2),clear=gx::UseReversedZ?0.f:1.f;
  EXPECT_TRUE(std::isfinite(center));EXPECT_GT(center,0.f);EXPECT_LT(center,1.f);
  EXPECT_EQ(depth_at(image,0,0),clear);
}
TEST_F(GxShaderGpu, MultisampleColorAndEveryDepthSampleParity) {
  multisample_targets();ASSERT_FALSE(::testing::Test::HasFatalFailure());
  const auto check_depth=[&](const Image& image) {
    ASSERT_EQ(image.depth.size(),size_t(width)*height*4*4);
    const float clear=gx::UseReversedZ?0.f:1.f;
    const auto center=depth_at(image,width/2,height/2);
    EXPECT_TRUE(std::isfinite(center));EXPECT_GT(center,0.f);EXPECT_LT(center,1.f);
    for(u32 sample=0;sample<4;++sample) {
      EXPECT_EQ(depth_at(image,width/2,height/2,sample),center);
      EXPECT_EQ(depth_at(image,0,0,sample),clear);
    }
    u32 mixed=0;
    for(u32 y=0;y<height;++y)for(u32 x=0;x<width;++x) {
      u32 covered=0;
      for(u32 sample=0;sample<4;++sample) covered+=depth_at(image,x,y,sample)!=clear;
      mixed+=covered>0 && covered<4;
    }
    EXPECT_GT(mixed,0u)<<"triangle edges must exercise distinct covered depth samples";
  };
  auto f=triangle();f.tevColor={43.f/255,92.f/255,179.f/255,204.f/255};
  const auto reference=probed_parity(f,"msaa4");
  expect_pixel(reference,width/2,height/2,{43,92,179,204});check_depth(reference);
  auto noDepth=f;noDepth.config.depthUpdate=false;
  const auto unwritten=capture(noDepth,gx::DrawShaderSelection::GenericOnly);
  EXPECT_EQ(reference.color,unwritten.color);
  EXPECT_NE(reference.depth,unwritten.depth)<<"depth-write negative control must be independent of color";
  const float clear=gx::UseReversedZ?0.f:1.f;
  for(u32 sample=0;sample<4;++sample) EXPECT_EQ(depth_at(unwritten,width/2,height/2,sample),clear);
  f.config.dstAlpha=83;
  const auto replaced=probed_parity(f,"msaa4-destination-alpha-replace");
  expect_pixel(replaced,width/2,height/2,{43,92,179,83});check_depth(replaced);
  struct RestoreDualSource {
    bool value=webgpu::g_dualSourceBlendingSupported;
    ~RestoreDualSource() { webgpu::g_dualSourceBlendingSupported=value; }
  } restore;
  webgpu::g_dualSourceBlendingSupported=false;
  f.config.blendMode=GX_BM_BLEND;f.config.blendFacSrc=GX_BL_SRCALPHA;f.config.blendFacDst=GX_BL_INVSRCALPHA;
  EXPECT_TRUE(gx::make_pipeline_plan(f.config,gfx::scene_render_target_layout()).hasPrepass);
  const auto blended=probed_parity(f,"msaa4-destination-alpha-two-pass");
  ASSERT_EQ(blended.color.size(),size_t(width)*height*4);
  EXPECT_EQ(blended.color[(height/2*width+width/2)*4+3],83);
  EXPECT_NE(blended.color,reference.color);check_depth(blended);
  if(restore.value) {
    webgpu::g_dualSourceBlendingSupported=true;
    f.config.dstAlpha=84;
    EXPECT_EQ(gx::make_pipeline_plan(f.config,gfx::scene_render_target_layout()).main.dstAlphaMode,gx::DstAlphaMode::DualSource);
    const auto dual=probed_parity(f,"msaa4-destination-alpha-dual-source");
    ASSERT_EQ(dual.color.size(),size_t(width)*height*4);
    EXPECT_EQ(dual.color[(height/2*width+width/2)*4+3],84);check_depth(dual);
  }
}
TEST_F(GxShaderGpu, DistinctMipLevelsAndSignedLodBias) {
  for(u32 level=1;level<4;++level) {
    auto f=mip_triangle(0.25f,0.5f);f.minLod=f.maxLod=float(level);
    const auto image=probed_parity(f,"clamped-mip-"+std::to_string(level));
    expect_pixel(image,width/2,height/2,MipColors[level]);
    auto base=f;base.minLod=base.maxLod=0;
    const auto zero=capture(base,gx::DrawShaderSelection::GenericOnly);
    expect_pixel(zero,width/2,height/2,MipColors[0]);
    EXPECT_NE(image.color,zero.color)<<"forcing mip zero must change visible output";
  }
  for(bool arbitrary:{false,true})for(int bias:{-1,0,1}) {
    auto f=mip_triangle(0,0);f.arbitraryMips=arbitrary;f.lodBias=float(bias);
    const auto logical=vi::configured_fb_size();
    const float sx=float(arbitrary?logical.x:width)/4.f,sy=float(arbitrary?logical.y:height)/4.f;
    f.vertices.clear();
    for(auto xy:{Vec2<float>{-0.9f,-0.8f},Vec2<float>{0.9f,-0.8f},Vec2<float>{0,0.9f}}) {
      f32be(f.vertices,xy.x);f32be(f.vertices,xy.y);f32be(f.vertices,0.25f);
      f32be(f.vertices,(xy.x+1.f)*sx);f32be(f.vertices,(1.f-xy.y)*sy);
    }
    const auto image=probed_parity(f,"implicit-lod-"+std::to_string(arbitrary)+"-"+std::to_string(bias));
    expect_pixel(image,width/2,height/2,MipColors[2+bias]);
  }
}
TEST_F(GxShaderGpu, MultiMipDirectSubtexelEdgesKeepNearestAndLinearSamples) {
  for(bool linear:{false,true})for(int side:{-1,1}) {
    auto f=mip_triangle(0.5f+float(side)/512.f,0.5f);
    f.linearTexture=linear;f.minLod=f.maxLod=0;
    const auto image=probed_parity(f,"mip0-subtexel-"+std::to_string(linear)+"-"+std::to_string(side));
    const std::array<u8,4> expected=linear ? (side<0?std::array<u8,4>{131,63,126,161}:std::array<u8,4>{125,65,130,159})
      : (side<0?MipColors[0]:MipRight);
    expect_pixel(image,width/2,height/2,expected);
    expect_pixel(image,width/2,height/3,expected);
  }
}

TEST_F(GxShaderGpu, NontrivialReferenceAndMetadataNegativeControl) {
  auto f=triangle();const auto reference=capture(f,gx::DrawShaderSelection::RequiredSpecialized);
  ASSERT_EQ(reference.color.size(),width*height*4);
  const auto center=(height/2*width+width/2)*4;
  EXPECT_NEAR(reference.color[center],43,1);EXPECT_NEAR(reference.color[center+1],92,1);EXPECT_NEAR(reference.color[center+2],179,1);EXPECT_NEAR(reference.color[center+3],204,1);
  compare(reference,capture(f,gx::DrawShaderSelection::GenericOnly),"basic");
  f.config.shaderConfig.tevStages[0].colorPass.d=GX_CC_C1;
  EXPECT_FALSE(same(reference,capture(f,gx::DrawShaderSelection::GenericOnly)))<<"metadata mutation must change visible pixels";
}
TEST_F(GxShaderGpu, ProductionFifoUsesImmutableForcedGenericSelection) {
  auto f=triangle();f.production=true;
  const auto reference=capture(f,gx::DrawShaderSelection::RequiredSpecialized);
  ASSERT_EQ(reference.color.size(),width*height*4);
  const auto center=(height/2*width+width/2)*4;
  EXPECT_EQ(reference.color[center],43);EXPECT_EQ(reference.color[center+1],92);
  EXPECT_EQ(reference.color[center+2],179);EXPECT_EQ(reference.color[center+3],204);
  const auto before=gx::testing::hybrid_counters();
  const auto actual=capture(f,gx::DrawShaderSelection::GenericOnly);
  compare(reference,actual,"production-fifo");
  EXPECT_EQ(gx::testing::hybrid_counters().generic,before.generic+1);
}
TEST_F(GxShaderGpu, StartupModeHasNoOptionalResourcesAndRestartRecreatesThem) {
  auto f=triangle();const auto reference=capture(f,gx::DrawShaderSelection::RequiredSpecialized);
  const auto generation=gfx::detail::resources().deviceGeneration;
  gx::fifo::shutdown();fifoInitialized=false;gfx::shutdown();initialized=false;
  aurora_set_shader_mode(AURORA_SHADER_SPECIALIZED);gfx::initialize();initialized=true;gx::fifo::init();fifoInitialized=true;
  EXPECT_FALSE(gfx::detail::resources().uberStateBuffer);
  EXPECT_EQ(gfx::detail::allocated_staging_buffer_size(),gfx::detail::StagingBufferSize);
  EXPECT_GT(gfx::detail::resources().deviceGeneration,generation);
  compare(reference,capture(f,gx::DrawShaderSelection::RequiredSpecialized),"restart-specialized");
  aurora_set_shader_mode(AURORA_SHADER_HYBRID);
  EXPECT_EQ(gfx::detail::resources().shaderMode,AURORA_SHADER_SPECIALIZED);
  EXPECT_FALSE(gfx::detail::resources().uberStateBuffer);
  gx::fifo::shutdown();fifoInitialized=false;gfx::shutdown();initialized=false;
  gfx::initialize();initialized=true;gx::fifo::init();fifoInitialized=true;
  EXPECT_TRUE(gfx::detail::resources().uberStateBuffer);
  compare(reference,capture(f,gx::DrawShaderSelection::GenericOnly),"restart-hybrid");
}
TEST_F(GxShaderGpu, NormalAttachmentAndHighMatrixIndexBits) {
  webgpu::g_normalBuffer=target(webgpu::NormalBufferFormat,"Parity normals");
  webgpu::g_graphicsConfig.normalBuffer=true;
  auto f=triangle();const auto old=f.vertices;f.vertices.clear();
  auto& c=f.config.shaderConfig;
  c.attrs[GX_VA_PNMTXIDX]={.attrType=GX_DIRECT,.cnt=1,.compType=GX_U8,.offset=0,.le=false};
  c.attrs[GX_VA_POS].offset=1;
  c.attrs[GX_VA_NRM]={.attrType=GX_DIRECT,.cnt=3,.compType=GX_F32,.offset=13,.le=false};c.vtxStride=25;
  for(u32 i=0;i<3;++i) {
    f.vertices.push_back(u8(0xc0u));f.vertices.insert(f.vertices.end(),old.begin()+i*12,old.begin()+(i+1)*12);
    f32be(f.vertices,0.2f);f32be(f.vertices,0.5f);f32be(f.vertices,0.8f);
  }
  parity(f,"normal-high-pn-bits");
  f.config.shaderConfig.attrs[GX_VA_NRM].attrType=GX_NONE;parity(f,"neutral-normal");
}
TEST_F(GxShaderGpu, KonstSelectorsKeepDefinedAndReservedOutput) {
  constexpr std::array<std::array<u8,4>,4> colors{{
    {32,64,96,128},{48,80,112,144},{80,112,144,176},{96,128,160,192},
  }};
  constexpr std::array<u8,8> fractions{255,223,191,159,128,96,64,32};
  std::vector<u32> selectors;
  for(u32 selector=0;selector<32;++selector) selectors.push_back(selector);
  selectors.push_back(32);selectors.push_back(UINT32_MAX);
  for(bool alpha:{false,true})for(u32 selector:selectors) {
    SCOPED_TRACE((alpha?"alpha ":"color ")+std::to_string(selector));
    auto f=triangle();f.tevColor={43.f/255,92.f/255,179.f/255,204.f/255};
    auto& config=f.config.shaderConfig;config.tevStageCount=5;
    for(u32 i=0;i<4;++i) {
      const auto& bytes=colors[i];
      f.kcolors[i]={bytes[0]/255.f,bytes[1]/255.f,bytes[2]/255.f,bytes[3]/255.f};
      auto& stage=config.tevStages[i];
      stage.colorPass.d=GX_CC_KONST;stage.kcSel=GXTevKColorSel(GX_TEV_KCSEL_K0+i);
      stage.alphaPass.d=GX_CA_KONST;stage.kaSel=GXTevKAlphaSel(GX_TEV_KASEL_K0_A+i);
    }
    auto& last=config.tevStages[4];
    last.colorPass.d=alpha?GX_CC_C0:GX_CC_KONST;
    last.alphaPass.d=alpha?GX_CA_KONST:GX_CA_A0;
    last.kcSel=GXTevKColorSel(selector);last.kaSel=GXTevKAlphaSel(selector);
    EXPECT_EQ(gx::build_shader_info(config).sampledKColors.count(),4u);
    std::array<u8,4> expected{43,92,179,204};
    if(alpha) {
      expected[3]=selector<8?fractions[selector]:selector>=16 && selector<32?
        colors[(selector-16)%4][(selector-16)/4]:0;
    } else {
      std::array<u8,3> rgb{};
      if(selector<8) rgb.fill(fractions[selector]);
      else if(selector>=12 && selector<16) std::copy_n(colors[selector-12].begin(),3,rgb.begin());
      else if(selector>=16 && selector<32) rgb.fill(colors[(selector-16)%4][(selector-16)/4]);
      std::copy(rgb.begin(),rgb.end(),expected.begin());
    }
    const auto reference=probed_parity(f,"konst-"+std::to_string(alpha)+"-"+std::to_string(selector));
    expect_pixel(reference,width/2,height/2,expected);
    const auto depth=depth_at(reference,width/2,height/2);
    EXPECT_TRUE(std::isfinite(depth));EXPECT_GT(depth,0.f);EXPECT_LT(depth,1.f);
  }
}
TEST_F(GxShaderGpu, FloatingTevFeedbackAndColorComparisons) {
  for(auto op:{GX_TEV_ADD,GX_TEV_SUB,GX_TEV_COMP_R8_GT,GX_TEV_COMP_R8_EQ,GX_TEV_COMP_GR16_GT,GX_TEV_COMP_GR16_EQ,GX_TEV_COMP_BGR24_GT,GX_TEV_COMP_BGR24_EQ,GX_TEV_COMP_RGB8_GT,GX_TEV_COMP_RGB8_EQ}) {
    for(auto scale:{GX_CS_SCALE_1,GX_CS_SCALE_2,GX_CS_SCALE_4,GX_CS_DIVIDE_2}) {
      auto f=triangle();auto& c=f.config.shaderConfig;c.tevStageCount=3;
      auto& first=c.tevStages[0];first.colorPass={GX_CC_C0,GX_CC_C1,GX_CC_HALF,GX_CC_C2};first.colorOp={.op=op,.bias=GX_TB_SUBHALF,.scale=scale,.outReg=GX_TEVREG1,.clamp=false};
      c.tevStages[1].colorPass={GX_CC_C1,GX_CC_C0,GX_CC_KONST,GX_CC_ZERO};c.tevStages[1].colorOp.clamp=true;c.tevStages[1].alphaPass.d=GX_CA_A1;
      c.tevStages[2].colorPass.d=GX_CC_CPREV;c.tevStages[2].alphaPass={GX_CA_A0,GX_CA_A1,GX_CA_KONST,GX_CA_ZERO};
      parity(f,"tev-"+std::to_string(op)+"-"+std::to_string(scale));
    }
  }
  for(auto op:{GX_TEV_COMP_R8_GT,GX_TEV_COMP_R8_EQ,GX_TEV_COMP_GR16_GT,GX_TEV_COMP_GR16_EQ,GX_TEV_COMP_BGR24_GT,GX_TEV_COMP_BGR24_EQ,GX_TEV_COMP_A8_GT,GX_TEV_COMP_A8_EQ}) {
    auto f=triangle();auto& stage=f.config.shaderConfig.tevStages[0];
    stage.colorPass={GX_CC_C0,GX_CC_C1,GX_CC_HALF,GX_CC_ZERO};
    stage.alphaPass={GX_CA_A0,GX_CA_A1,GX_CA_KONST,GX_CA_A2};stage.alphaOp.op=op;
    parity(f,"packed-alpha-"+std::to_string(op));
  }
  for(u32 count:{1u,4u,16u}) { auto f=triangle();f.config.shaderConfig.tevStageCount=count;for(u32 i=1;i<count;++i){ f.config.shaderConfig.tevStages[i].colorPass.d=GX_CC_CPREV;f.config.shaderConfig.tevStages[i].alphaPass.d=GX_CA_APREV; }parity(f,"stages-"+std::to_string(count)); }
}
TEST_F(GxShaderGpu, DirectSubtexelIndirectFogAndZTexture) {
  auto f=triangle();textured(f);parity(f,"raw-direct-7x5");
  for(auto format:{GX_ITF_8,GX_ITF_5,GX_ITF_4,GX_ITF_3}) {
    auto indirect=f;auto& c=indirect.config.shaderConfig;c.numIndStages=1;c.indStages[0]={GX_TEXCOORD0,GX_TEXMAP0,GX_ITS_2,GX_ITS_4};
    c.tevStages[0].indTexMtxId=GX_ITM_0;c.tevStages[0].indTexFormat=format;c.tevStages[0].indTexBiasSel=GX_ITB_STU;c.tevStages[0].indTexWrapS=GX_ITW_16;
    c.tevStageCount=2;c.tevStages[1]=c.tevStages[0];c.tevStages[1].indTexAddPrev=true;parity(indirect,"indirect-"+std::to_string(format));
  }
  for(auto type:{GX_FOG_PERSP_LIN,GX_FOG_PERSP_EXP,GX_FOG_PERSP_EXP2,GX_FOG_PERSP_REVEXP,GX_FOG_PERSP_REVEXP2,GX_FOG_ORTHO_LIN}) { auto fog=f;fog.config.shaderConfig.fogType=type;fog.config.shaderConfig.fogRangeEnabled=true;parity(fog,"fog-"+std::to_string(type)); }
  if(webgpu::g_backendType!=wgpu::BackendType::Vulkan)
    for(u32 format:{0u,1u,2u})for(auto op:{GX_ZT_ADD,GX_ZT_REPLACE}) {auto z=f;z.config.shaderConfig.zTextureOp=op;z.config.shaderConfig.zTextureFormat=format;parity(z,"z-"+std::to_string(format)+"-"+std::to_string(op));}
}
TEST_F(GxShaderGpu, DepthTextureFallbackPreservesProductionFifoOutput) {
  for(u32 format:{0u,1u,2u})for(auto op:{GX_ZT_ADD,GX_ZT_REPLACE}) {
    auto f=triangle();f.production=true;
    f.config.shaderConfig.zTextureOp=op;f.config.shaderConfig.zTextureFormat=format;
    const auto reference=capture(f,gx::DrawShaderSelection::RequiredSpecialized);
    ASSERT_EQ(reference.color.size(),width*height*4);
    const auto center=(height/2*width+width/2)*4;
    EXPECT_EQ(reference.color[center],43);EXPECT_EQ(reference.color[center+1],92);
    EXPECT_EQ(reference.color[center+2],179);EXPECT_EQ(reference.color[center+3],204);
    float depth;std::memcpy(&depth,reference.depth.data()+center,4);EXPECT_GT(depth,0.0f);EXPECT_LT(depth,1.0f);
    const auto before=gx::testing::hybrid_counters();
    Image actual;
    if(webgpu::g_backendType==wgpu::BackendType::Vulkan) {
      Gate gate;gate.all_specializations();
      auto release=std::async(std::launch::async,[&] {
        const bool entered=gate.wait();
        const auto deadline=std::chrono::steady_clock::now()+5s;
        while(gx::testing::hybrid_counters().unsupportedFallback==before.unsupportedFallback &&
              std::chrono::steady_clock::now()<deadline) std::this_thread::sleep_for(1ms);
        const bool fallback=gx::testing::hybrid_counters().unsupportedFallback==before.unsupportedFallback+1;
        gate.release();return entered && fallback;
      });
      f.config.blendOp=GX_LO_OR;
      actual=capture(f,gx::DrawShaderSelection::Hybrid);
      EXPECT_TRUE(release.get());EXPECT_FALSE(gate.timed_out());
      const auto after=gx::testing::hybrid_counters();
      EXPECT_EQ(after.generic,before.generic);EXPECT_EQ(after.specialized,before.specialized+1);
      EXPECT_EQ(after.unsupportedFallback,before.unsupportedFallback+1);
    } else actual=capture(f,gx::DrawShaderSelection::Hybrid);
    compare(reference,actual,"production-z-"+std::to_string(format)+"-"+std::to_string(op));
  }
}
TEST_F(GxShaderGpu, NumericVertexFormatsCrossWordsAndIndexedPositionsUseArrayEndianness) {
  for(auto comp:{GX_U8,GX_S8,GX_U16,GX_S16,GX_F32}) {
    auto f=triangle();auto& c=f.config.shaderConfig;f.vertices.clear();
    const bool wide=comp==GX_U16||comp==GX_S16;
    const bool signedValue=comp==GX_S8||comp==GX_S16;
    c.attrs[GX_VA_PNMTXIDX]={.attrType=GX_DIRECT,.cnt=1,.compType=GX_U8,.offset=0,.le=false};
    c.attrs[GX_VA_POS].offset=1;c.attrs[GX_VA_POS].compType=u8(comp);c.attrs[GX_VA_POS].frac=comp==GX_F32?0:(wide?12:6);
    c.vtxStride=u8(1+3*(comp==GX_F32?4:(wide?2:1)));
    for(u32 vertex=0;vertex<3;++vertex) {
      f.vertices.push_back(0xc0);
      if(comp==GX_F32) {
        f32be(f.vertices,vertex==0?-0.875f:(vertex==1?0.875f:0));
        f32be(f.vertices,vertex==2?0.875f:-0.75f);f32be(f.vertices,0.25f);
      } else {
        const int scale=wide?64:1;
        const int x=(vertex==0?(signedValue?-56:8):(vertex==1?56:(signedValue?0:32)))*scale;
        const int y=(vertex==2?56:(signedValue?-48:8))*scale;
        for(int value:{x,y,16*scale}) {
          if(wide) f.vertices.push_back(u8(u16(value)>>8));f.vertices.push_back(u8(value));
        }
      }
    }
    parity(f,"vertex-format-"+std::to_string(comp));
  }
  for(bool index16:{false,true})for(bool little:{false,true}) {
    auto f=triangle();const auto raw=f.vertices;f.vertices.clear();
    auto& c=f.config.shaderConfig;c.vtxStride=index16?2:1;
    c.attrs[GX_VA_POS]={.attrType=u8(index16?GX_INDEX16:GX_INDEX8),.cnt=3,.compType=GX_F32,.offset=0,.stride=12,.le=little};
    for(u32 vertex=0;vertex<3;++vertex) {if(index16)f.vertices.push_back(0);f.vertices.push_back(u8(vertex));}
    f.attributes=raw;
    if(little)for(size_t i=0;i<f.attributes.size();i+=4)std::reverse(f.attributes.begin()+i,f.attributes.begin()+i+4);
    parity(f,"indexed-position-"+std::to_string(index16)+"-"+std::to_string(little));
  }
}
TEST_F(GxShaderGpu, PackedVertexColorsRemainNonwhiteAndKeepAlpha) {
  for(auto comp:{GX_RGB565,GX_RGB8,GX_RGBX8,GX_RGBA4,GX_RGBA6,GX_RGBA8}) {
    auto f=triangle();auto& c=f.config.shaderConfig;
    const auto raw=f.vertices;f.vertices.clear();
    const u32 bytes=(comp==GX_RGB565||comp==GX_RGBA4)?2:((comp==GX_RGB8||comp==GX_RGBA6)?3:4);
    c.vtxStride=u8(12+bytes);c.attrs[GX_VA_CLR0]={.attrType=GX_DIRECT,.cnt=4,.compType=u8(comp),.offset=12,.le=false};
    c.tevStages[0].channelId=GX_COLOR0A0;c.tevStages[0].colorPass.d=GX_CC_RASC;c.tevStages[0].alphaPass.d=GX_CA_RASA;
    c.colorChannels[0].matSrc=GX_SRC_VTX;c.colorChannels[GX_ALPHA0].matSrc=GX_SRC_VTX;
    for(u32 vertex=0;vertex<3;++vertex) {
      f.vertices.insert(f.vertices.end(),raw.begin()+vertex*12,raw.begin()+(vertex+1)*12);
      const u32 value=comp==GX_RGB565?(vertex==0?0xf800:(vertex==1?0x07e0:0x001f)):
        (comp==GX_RGBA4?(0xf135u-vertex*0x212u):0xa94f26u+vertex*0x175b31u);
      if(comp==GX_RGB565||comp==GX_RGBA4||comp==GX_RGBA6) {
        for(int shift=int(bytes-1)*8;shift>=0;shift-=8)f.vertices.push_back(u8(value>>shift));
      } else {
        f.vertices.push_back(u8(240-vertex*83));f.vertices.push_back(u8(20+vertex*91));f.vertices.push_back(u8(40+vertex*71));
        if(bytes==4)f.vertices.push_back(u8(32+vertex*99));
      }
    }
    parity(f,"vertex-color-"+std::to_string(comp));
  }
}
TEST_F(GxShaderGpu, IndexedNbt3UsesCurrentGroupOffsetsAndArrayEndianness) {
  for(bool index16:{false,true}) for(bool little:{false,true}) for(auto source:{GX_TG_BINRM,GX_TG_TANGENT}) {
    auto f=triangle();textured(f);
    auto& c=f.config.shaderConfig;c.tcgs[0].src=source;
    c.attrs[GX_VA_NRM]={.attrType=u8(index16?GX_INDEX16:GX_INDEX8),.cnt=9,.compType=GX_F32,.offset=20,.stride=36,.le=little,.nbt3=true};
    c.vtxStride=index16?26:23;
    const auto old=f.vertices;f.vertices.clear();
    for(u32 i=0;i<3;++i) {
      f.vertices.insert(f.vertices.end(),old.begin()+i*20,old.begin()+(i+1)*20);
      for(u32 group:{0u,1u,2u}) { if(index16)f.vertices.push_back(0);f.vertices.push_back(u8(group)); }
    }
    for(u32 entry=0;entry<3;++entry) for(u32 group=0;group<3;++group) {
      for(float value:{float(entry)*0.21f+float(group)*0.11f,float(group)*0.19f,0.75f}) {
        const auto bits=std::bit_cast<u32>(value);
        if(little)for(int shift:{0,8,16,24}) f.attributes.push_back(u8(bits>>shift));else be32(f.attributes,bits);
      }
    }
    parity(f,"nbt3-"+std::to_string(index16)+"-"+std::to_string(little)+"-"+std::to_string(source));
  }
}
TEST_F(GxShaderGpu, TextureMatrixMasksAndEmbossUseCurrentGeneratedCoordinates) {
  auto masked=triangle();textured(masked);auto& m=masked.config.shaderConfig;
  m.attrs[GX_VA_TEX0MTXIDX]={.attrType=GX_DIRECT,.cnt=1,.compType=GX_U8,.offset=0,.le=false};
  m.attrs[GX_VA_POS].offset=1;m.attrs[GX_VA_TEX0].offset=13;m.vtxStride=21;
  const auto old=masked.vertices;masked.vertices.clear();
  for(u32 i=0;i<3;++i) { masked.vertices.push_back(0xde);masked.vertices.insert(masked.vertices.end(),old.begin()+i*20,old.begin()+(i+1)*20); }
  parity(masked,"texture-matrix-high-bits");
  auto f=triangle();textured(f);auto& c=f.config.shaderConfig;
  c.attrs[GX_VA_NRM]={.attrType=GX_DIRECT,.cnt=9,.compType=GX_F32,.offset=20,.le=false};c.vtxStride=56;
  c.tcgs[0].postMtx=GX_PTTEXMTX19;
  c.tcgs[1]={.type=GX_TG_BUMP0,.src=GX_TG_TEX0,.mtx=GX_IDENTITY,.postMtx=GX_PTIDENTITY,.normalize=false,.embossSrc=0};
  c.tevStages[0].texCoordId=GX_TEXCOORD1;
  const auto base=f.vertices;f.vertices.clear();
  for(u32 i=0;i<3;++i) {
    f.vertices.insert(f.vertices.end(),base.begin()+i*20,base.begin()+(i+1)*20);
    for(float value:{0.0f,0.0f,1.0f,0.0f,2.5f,0.0f,1.75f,0.0f,0.0f}) f32be(f.vertices,value);
  }
  parity(f,"emboss-nbt-post-matrix");
  c.lineMode=3;f.verticesCount=4;f.instances=3;parity(f,"emboss-point-offset-order");
}
TEST_F(GxShaderGpu, IndependentRasterLightingAndGeneratedColorTexgen) {
  auto f=triangle();textured(f);auto& c=f.config.shaderConfig;
  c.tevStages[0].colorPass.a=GX_CC_RASC;c.tevStages[0].colorPass.b=GX_CC_TEXC;c.tevStages[0].colorPass.c=GX_CC_HALF;c.tevStages[0].colorPass.d=GX_CC_ZERO;
  c.tevStages[0].channelId=GX_COLOR0A0;
  c.tevStages[0].alphaPass.d=GX_CA_RASA;
  c.colorChannels[0]={.matSrc=GX_SRC_REG,.ambSrc=GX_SRC_REG,.diffFn=GX_DF_CLAMP,.attnFn=GX_AF_NONE,.lightingEnabled=true};
  c.colorChannels[GX_ALPHA0]={.matSrc=GX_SRC_REG,.ambSrc=GX_SRC_REG,.diffFn=GX_DF_NONE,.attnFn=GX_AF_SPOT,.lightingEnabled=true};
  c.tcgs[0].type=GX_TG_SRTG;c.tcgs[0].src=GX_TG_COLOR0;
  parity(f,"independent-lighting-color-texgen");
}
TEST_F(GxShaderGpu, AlphaRejectionLinesPointsAndDestinationAlpha) {
  for(u32 ref:{0u,1u,127u,128u,254u,255u}) { auto f=triangle();f.config.shaderConfig.alphaCompare={.comp0=GX_GREATER,.ref0=ref,.op=GX_AOP_AND,.comp1=GX_ALWAYS,.ref1=0};f.expectedVisible=ref<204;parity(f,"alpha-"+std::to_string(ref)); }
  for(u32 mode:{1u,2u,3u}) { auto f=triangle();f.config.shaderConfig.lineMode=mode;f.verticesCount=4;f.instances=mode==3?3:1;parity(f,"line-point-"+std::to_string(mode)); }
  for(u32 alpha:{0u,127u,UINT32_MAX}) {auto f=triangle();f.config.dstAlpha=alpha;f.config.blendMode=GX_BM_BLEND;f.config.blendFacSrc=GX_BL_SRCALPHA;f.config.blendFacDst=GX_BL_INVSRCALPHA;parity(f,"dest-alpha-"+std::to_string(alpha));}
  webgpu::g_dualSourceBlendingSupported=false;
  auto f=triangle();f.config.dstAlpha=83;f.config.blendMode=GX_BM_BLEND;f.config.blendFacSrc=GX_BL_SRCALPHA;parity(f,"dest-alpha-two-pass");
}
TEST_F(GxShaderGpu, PendingSpecializationDoesNotBlockPersistentGenericAndThenConverges) {
  auto referenceFixture=triangle();referenceFixture.persistent=true;
  const auto reference=capture(referenceFixture,gx::DrawShaderSelection::RequiredSpecialized);
  auto pending=referenceFixture;pending.config.msaaSamples=923;
  const auto layout=gfx::scene_render_target_layout();
  const auto cacheKey=xxh3_hash(pending.config,static_cast<HashType>(gfx::ShaderType::GX));
  const auto ref=xxh3_hash(layout.key,cacheKey);
  Gate gate;gate.target(ref);
  ASSERT_EQ(gfx::find_deferred_specialization(pending.config,layout),ref);ASSERT_TRUE(gate.wait());
  ASSERT_EQ(gfx::pipeline_build_state(ref),gfx::PipelineBuildState::Pending);
  gfx::CompiledPipeline completed;
  const auto start=std::chrono::steady_clock::now();
  { gfx::RequireReadyPipelinesScope require(true);EXPECT_FALSE(gfx::try_get_pipeline(ref,completed)); }
  EXPECT_LT(std::chrono::steady_clock::now()-start,100ms);
  const auto before=gx::testing::hybrid_counters();
  compare(reference,capture(pending,gx::DrawShaderSelection::Hybrid,ref),"pending-persistent");
  EXPECT_EQ(gfx::pipeline_build_state(ref),gfx::PipelineBuildState::Pending);
  EXPECT_EQ(gx::testing::hybrid_counters().generic,before.generic+1);
  gate.release();ASSERT_TRUE(gfx::get_pipeline_ready(ref,completed));EXPECT_FALSE(gate.timed_out());
  const auto ready=gx::testing::hybrid_counters();
  compare(reference,capture(pending,gx::DrawShaderSelection::Hybrid,ref),"ready-specialized");
  EXPECT_EQ(gx::testing::hybrid_counters().specialized,ready.specialized+1);
}
TEST_F(GxShaderGpu, CapacityFailureUsesReadySpecializationAndEpochsReuseRealStaging) {
  auto f=triangle();const auto reference=capture(f,gx::DrawShaderSelection::RequiredSpecialized);
  auto full=f;full.attributes.resize(gfx::StorageBufferSize,0x5a);
  compare(reference,capture(full,gx::DrawShaderSelection::Hybrid,0,true),"metadata-capacity");
  compare(reference,capture(full,gx::DrawShaderSelection::GenericOnly),"full-attribute-region");
  for(u32 frame=0;frame<12;++frame) {
    auto changed=f;changed.config.shaderConfig.forceAlphaOne=(frame&1)!=0;
    parity(changed,"staging-epoch-"+std::to_string(frame));
  }
}
TEST_F(GxShaderGpu, DelayedPacketsKeepTheirOwnMetadataAcrossRealFrameAndStagingSlots) {
  auto a=triangle();auto b=a;b.config.shaderConfig.tevStages[0].colorPass.d=GX_CC_C1;
  const auto referenceA=capture(a,gx::DrawShaderSelection::RequiredSpecialized);
  const auto referenceB=capture(b,gx::DrawShaderSelection::RequiredSpecialized);
  for(u32 batch=0;batch<6;++batch) {
    struct Delay { std::mutex mutex;std::condition_variable cv;bool release=false; };
    const auto delay=std::make_shared<Delay>();
    gfx::render_worker::enqueue_work([delay] {
      std::unique_lock lock(delay->mutex);
      EXPECT_TRUE(delay->cv.wait_for(lock,10s,[&]{return delay->release;}));
    });
    a.metadataPrefix=4+batch*64;b.metadataPrefix=128+batch*32;
    auto first=submit(a,gx::DrawShaderSelection::GenericOnly);
    auto second=submit(b,gx::DrawShaderSelection::GenericOnly);
    { std::lock_guard lock(delay->mutex);delay->release=true;delay->cv.notify_all(); }
    compare(referenceA,read_ticket(std::move(first)),"delayed-a-"+std::to_string(batch));
    compare(referenceB,read_ticket(std::move(second)),"delayed-b-"+std::to_string(batch));
  }
}
TEST_F(GxShaderGpu, RealInterpolationReplayUsesTheSameUniformCopies) {
  auto f=triangle();f.interpolation=true;
  gx::set_frame_interpolation_fps(240);
  capture(f,gx::DrawShaderSelection::RequiredSpecialized);
  f.translate=0.2f;
  const auto reference=capture(f,gx::DrawShaderSelection::RequiredSpecialized);
  ASSERT_FALSE(reference.replay.empty());
  gx::set_frame_interpolation_fps(0);capture(triangle(),gx::DrawShaderSelection::RequiredSpecialized);
  gx::set_frame_interpolation_fps(240);f.translate=0;
  capture(f,gx::DrawShaderSelection::GenericOnly);
  f.translate=0.2f;
  const auto actual=capture(f,gx::DrawShaderSelection::GenericOnly);
  ASSERT_FALSE(actual.replay.empty());compare(reference,actual,"interpolation");
  EXPECT_FALSE(same(reference,reference.replay.front()))<<"interpolation must change visible geometry";
  gx::set_frame_interpolation_fps(0);
}
TEST_F(GxShaderGpu, OneGenericPairSurvivesCompletionBetweenAlphaPrepassAndMain) {
  webgpu::g_dualSourceBlendingSupported=false;
  auto f=triangle();f.config.dstAlpha=79;f.config.blendMode=GX_BM_BLEND;f.config.blendFacSrc=GX_BL_SRCALPHA;
  const auto reference=capture(f,gx::DrawShaderSelection::RequiredSpecialized);
  f.config.msaaSamples=781;
  const auto layout=gfx::scene_render_target_layout();
  const auto ref=xxh3_hash(layout.key,xxh3_hash(f.config,static_cast<HashType>(gfx::ShaderType::GX)));
  Gate gate;gate.target(ref);gfx::find_deferred_specialization(f.config,layout);ASSERT_TRUE(gate.wait());
  gx::testing::set_after_prepass_hook(Gate::complete_after_prepass);
  const auto before=gx::testing::hybrid_counters();
  const auto actual=capture(f,gx::DrawShaderSelection::Hybrid,ref);
  gx::testing::set_after_prepass_hook(nullptr);
  compare(reference,actual,"pair-completion");
  EXPECT_EQ(gx::testing::hybrid_counters().generic,before.generic+1);
  EXPECT_EQ(gfx::pipeline_build_state(ref),gfx::PipelineBuildState::Ready);
}
TEST_F(GxShaderGpu, BothFailedPipelinesAreReportedAsFatal) {
  GTEST_FLAG_SET(death_test_style,"threadsafe");
  EXPECT_DEATH({
    Gate gate;gate.failures(true,true);
    auto f=triangle();f.config.blendOp=GX_LO_OR;
    capture(f,gx::DrawShaderSelection::Hybrid);
  }, "No usable GX pipeline");
}
TEST_F(GxShaderGpu, ActualGenericValidationFailureUsesSpecialization) {
  auto f=triangle();f.config.dstAlpha=81;
  const auto reference=capture(f,gx::DrawShaderSelection::RequiredSpecialized);
  gx::testing::invalidate_generic_source(true);
  const auto generic=gfx::find_uber_pipeline(f.config,gfx::scene_render_target_layout());
  gfx::CompiledPipeline failed;
  EXPECT_FALSE(gfx::get_pipeline_ready(generic,failed));
  EXPECT_EQ(gfx::pipeline_build_state(generic),gfx::PipelineBuildState::Failed);
  gx::testing::invalidate_generic_source(false);
  EXPECT_FALSE(gpuError.load());
  compare(reference,capture(f,gx::DrawShaderSelection::Hybrid),"actual-generic-validation-fallback");
}
TEST_F(GxShaderGpu, GenericPipelineFailureFallsBackAndFailedSpecializationStillDraws) {
  auto f=triangle();const auto reference=capture(f,gx::DrawShaderSelection::RequiredSpecialized);
  Gate gate;gate.failures(true,false);
  auto fixed=f;fixed.config.blendOp=GX_LO_XOR;
  compare(reference,capture(fixed,gx::DrawShaderSelection::Hybrid),"generic-failure-fallback");
  gate.failures(false,true);
  auto material=f;material.config.msaaSamples=998;
  const auto ref=gfx::find_deferred_specialization(material.config,gfx::scene_render_target_layout());
  gfx::CompiledPipeline failed;EXPECT_FALSE(gfx::get_pipeline_ready(ref,failed));EXPECT_EQ(gfx::pipeline_build_state(ref),gfx::PipelineBuildState::Failed);
  compare(reference,capture(material,gx::DrawShaderSelection::Hybrid,ref),"specialization-failure-generic");
}
} // namespace
