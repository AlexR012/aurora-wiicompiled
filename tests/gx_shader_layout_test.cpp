#include <gtest/gtest.h>
#include "gx/ubershader.hpp"
#include "gx/shader_info.hpp"
#include "gfx/frame.hpp"
#include "gfx/frame_packet.hpp"
#include "gfx/recording.hpp"
#include "gfx/resources.hpp"
#include <algorithm>
#include <cstring>

namespace {
using namespace aurora;
gfx::RenderTargetLayout layout() {
  gfx::RenderTargetLayout out{};
  out.colorAttachmentCount = 1;
  out.colorAttachments[0] = {gfx::ColorAttachmentSemantic::SceneColor, wgpu::TextureFormat::RGBA8Unorm};
  out.depthStencilFormat = wgpu::TextureFormat::Depth32Float;
  gfx::detail::finalize_render_target_layout(out);
  return out;
}
gx::PipelineConfig config() {
  gx::PipelineConfig out{};
  out.shaderConfig.tevStageCount = 1;
  out.shaderConfig.tevStages[0].colorPass.d = GX_CC_C0;
  out.shaderConfig.tevStages[0].alphaPass.d = GX_CA_A0;
  out.depthFunc = GX_LEQUAL;
  out.cullMode = GX_CULL_NONE;
  out.blendMode = GX_BM_NONE;
  out.blendFacSrc = GX_BL_ONE;
  out.blendFacDst = GX_BL_ZERO;
  out.blendOp = GX_LO_CLEAR;
  out.dstAlpha = UINT32_MAX;
  out.colorUpdate = out.alphaUpdate = out.depthUpdate = out.depthCompare = true;
  return out;
}
TEST(GxShaderLayout, ImmediateAndDrawPayloadStayWithinExistingLimits) {
  EXPECT_EQ(sizeof(gx::DrawImmediateData), 64u);
  EXPECT_EQ(offsetof(gx::DrawImmediateData, uberStateOffset), 12u);
  EXPECT_EQ(offsetof(gx::DrawImmediateData, arrayStart), 16u);
  EXPECT_LE(sizeof(gx::DrawData), 128u);
  EXPECT_EQ(gfx::StorageBufferSize, 8u * 1024 * 1024);
  EXPECT_EQ(gfx::detail::UberStateStagingOffset, gfx::detail::StagingBufferSize);
}
TEST(GxShaderLayout, HybridResourcesRespectBothStorageLimitSets) {
  const wgpu::Limits limits{.maxStorageBuffersPerShaderStage=8,
      .maxStorageBufferBindingSize=128u*1024*1024,.maxInterStageShaderVariables=15};
  wgpu::CompatibilityModeLimits compatibility{wgpu::CompatibilityModeLimits::Init{
      .maxStorageBuffersInVertexStage=2,.maxStorageBuffersInFragmentStage=4}};
  EXPECT_FALSE(gfx::detail::supports_hybrid_resources(limits,compatibility));
  compatibility.maxStorageBuffersInVertexStage=3;
  EXPECT_TRUE(gfx::detail::supports_hybrid_resources(limits,compatibility));
  compatibility.maxStorageBuffersInFragmentStage=1;
  EXPECT_FALSE(gfx::detail::supports_hybrid_resources(limits,compatibility));
  compatibility.maxStorageBuffersInFragmentStage=2;
  EXPECT_TRUE(gfx::detail::supports_hybrid_resources(limits,compatibility));
  for(u32 invalid:{0u,wgpu::kLimitU32Undefined}) {
    auto missing=compatibility;missing.maxStorageBuffersInVertexStage=invalid;
    EXPECT_FALSE(gfx::detail::supports_hybrid_resources(limits,missing));
    missing=compatibility;missing.maxStorageBuffersInFragmentStage=invalid;
    EXPECT_FALSE(gfx::detail::supports_hybrid_resources(limits,missing));
  }
  auto reduced=limits;reduced.maxStorageBuffersPerShaderStage=2;
  EXPECT_FALSE(gfx::detail::supports_hybrid_resources(reduced,compatibility));
  reduced=limits;reduced.maxStorageBufferBindingSize=gfx::UberStateBufferSize-1;
  EXPECT_FALSE(gfx::detail::supports_hybrid_resources(reduced,compatibility));
  reduced=limits;reduced.maxInterStageShaderVariables=10;
  EXPECT_FALSE(gfx::detail::supports_hybrid_resources(reduced,compatibility));
}
TEST(GxShaderLayout, OffsetsFollowCurrentCompactUniforms) {
  auto c = config().shaderConfig;
  const auto base = gx::build_shader_info(c);
  EXPECT_EQ(base.offsets.projection, 16u);
  EXPECT_EQ(base.offsets.postex, 80u);
  EXPECT_EQ(base.offsets.nrm, 1040u);
  EXPECT_EQ(base.offsets.tevreg[GX_TEVREG0], 1520u);
  EXPECT_EQ(base.offsets.lights, gx::UniformOffsets::Absent);
  EXPECT_EQ(base.offsets.fog, gx::UniformOffsets::Absent);
  EXPECT_EQ(base.offsets.texcoordScale[7] - base.offsets.texcoordScale[0], 7u * 16);
  EXPECT_EQ(base.uniformSize, gfx::align_uniform(base.offsets.end));
  c.lineMode = 3;
  c.fogType = GX_FOG_PERSP_EXP2;
  c.fogRangeEnabled = true;
  c.zTextureOp = GX_ZT_REPLACE;
  c.tevStages[0].texMapId = GX_TEXMAP0;
  c.tevStages[0].texCoordId = GX_TEXCOORD0;
  c.tevStages[0].colorPass.a = GX_CC_RASC;
  c.tevStages[0].channelId = GX_COLOR0A0;
  c.colorChannels[0].lightingEnabled = true;
  c.tcgs[0].postMtx = GX_PTTEXMTX0;
  c.numIndStages = 1;
  c.indStages[0] = {GX_TEXCOORD0, GX_TEXMAP0, GX_ITS_1, GX_ITS_2};
  c.tevStages[0].indTexMtxId = GX_ITM_0;
  const auto full = gx::build_shader_info(c);
  EXPECT_EQ(full.offsets.projection, 32u);
  EXPECT_EQ(full.offsets.lights + sizeof(gx::Light) * GX::MaxLights, full.offsets.lightState);
  EXPECT_EQ(full.offsets.postmtx + gx::MaxPTTexMtx * 48, full.offsets.fog);
  EXPECT_EQ(full.offsets.fog + sizeof(gx::Fog), full.offsets.ztexture);
  EXPECT_EQ(full.offsets.ztexture + 16, full.offsets.texcoordScale[0]);
  EXPECT_EQ(full.offsets.texcoordScale[7] + 16, full.offsets.indmtx);
  EXPECT_EQ(full.offsets.indmtx + 96, full.offsets.texSizeBias[0]);
  EXPECT_EQ(full.offsets.texSizeBias[0] + 16, full.offsets.end);
  EXPECT_LE(full.uniformSize, gx::MaxUniformSize);
  const auto snapshot = gx::build_uber_shader_state(c, full);
  EXPECT_EQ(snapshot.flags & 7u, 7u);
  EXPECT_EQ((snapshot.flags >> 3u) & 3u, GX_ZT_REPLACE);
  EXPECT_EQ(std::memcmp(&snapshot.uniforms, &full.offsets, sizeof(full.offsets)), 0);
}
TEST(GxShaderKey, CoversEveryFixedFieldAndExcludesMaterialState) {
  const auto c = config();
  const auto l = layout();
  const auto key = gx::uber_pipeline_key(c, l);
  const auto differs = [&](const gx::PipelineConfig& changed, const gfx::RenderTargetLayout& target) {
    const auto other = gx::uber_pipeline_key(changed, target);
    EXPECT_NE(std::memcmp(&key, &other, sizeof(key)), 0);
  };
#define CHANGE(field, value) { auto changed = c; changed.field = value; differs(changed, l); }
  CHANGE(depthFunc, GX_GREATER); CHANGE(cullMode, GX_CULL_BACK); CHANGE(blendMode, GX_BM_BLEND);
  CHANGE(blendFacSrc, GX_BL_SRCALPHA); CHANGE(blendFacDst, GX_BL_ONE); CHANGE(blendOp, GX_LO_XOR);
  CHANGE(dstAlpha, 0); CHANGE(polygonOffsetBits, 1); CHANGE(polygonOffsetScaleBits, 1); CHANGE(polygonOffsetClampBits, 1);
  CHANGE(depthCompare, false); CHANGE(depthUpdate, false); CHANGE(colorUpdate, false); CHANGE(alphaUpdate, false);
  CHANGE(shaderConfig.zTextureOp, GX_ZT_REPLACE);
#undef CHANGE
  for (u32 sample : {2u, 4u, 8u}) { auto target = l; target.sampleCount = sample; gfx::detail::finalize_render_target_layout(target); differs(c, target); }
  auto target = l;
  target.colorAttachments[0].format = wgpu::TextureFormat::BGRA8Unorm;
  gfx::detail::finalize_render_target_layout(target); differs(c, target);
  target = l; target.depthStencilFormat = wgpu::TextureFormat::Depth24Plus;
  gfx::detail::finalize_render_target_layout(target); differs(c, target);
  target = l; target.colorAttachmentCount = 2;
  target.colorAttachments[1] = {gfx::ColorAttachmentSemantic::Normal, wgpu::TextureFormat::RGB10A2Unorm};
  gfx::detail::finalize_render_target_layout(target); differs(c, target);
  auto material = c;
  material.msaaSamples = 123;
  material.shaderConfig.tevStages[0].colorPass.d = GX_CC_KONST;
  material.shaderConfig.attrs[GX_VA_POS].frac = 4;
  material.shaderConfig.tcgs[0].mtx = GX_TEXMTX1;
  auto shared = gx::uber_pipeline_key(material, l);
  EXPECT_EQ(std::memcmp(&key, &shared, sizeof(key)), 0);
  target = l; target.colorAttachments[0].width = 640; target.colorAttachments[0].height = 480;
  gfx::detail::finalize_render_target_layout(target);
  shared = gx::uber_pipeline_key(c, target);
  EXPECT_EQ(std::memcmp(&key, &shared, sizeof(key)), 0);
}
TEST(GxShaderKey, DestinationAlphaPlanMatchesBothFactories) {
  auto c = config(); auto l = layout();
  const bool dual = webgpu::g_dualSourceBlendingSupported;
  c.dstAlpha = 127;
  EXPECT_EQ(gx::make_pipeline_plan(c,l).main.dstAlphaMode, gx::DstAlphaMode::Replace);
  c.blendMode = GX_BM_BLEND; c.blendFacSrc = GX_BL_SRCALPHA;
  webgpu::g_dualSourceBlendingSupported = false;
  const auto twoPass = gx::make_pipeline_plan(c,l);
  EXPECT_TRUE(twoPass.hasPrepass);
  EXPECT_FALSE(twoPass.prepass.depthUpdate);
  EXPECT_FALSE(twoPass.prepass.colorUpdate);
  EXPECT_FALSE(twoPass.main.alphaUpdate);
  webgpu::g_dualSourceBlendingSupported = true;
  EXPECT_EQ(gx::make_pipeline_plan(c,l).main.dstAlphaMode, gx::DstAlphaMode::DualSource);
  l.colorAttachmentCount = 2;
  EXPECT_TRUE(gx::make_pipeline_plan(c,l).hasPrepass);
  webgpu::g_dualSourceBlendingSupported = dual;
}
class GxShaderArena : public ::testing::Test {
protected:
  void SetUp() override {
    gfx::detail::resources().shaderMode = AURORA_SHADER_HYBRID;
    gfx::detail::resources().limits.minUniformBufferOffsetAlignment = 256;
    gfx::detail::resources().limits.minStorageBufferOffsetAlignment = 256;
    webgpu::g_graphicsConfig.surfaceConfiguration.format = wgpu::TextureFormat::RGBA8Unorm;
    webgpu::g_graphicsConfig.depthFormat = wgpu::TextureFormat::Depth32Float;
    webgpu::g_graphicsConfig.msaaSamples = 1;
    webgpu::g_frameBuffer.size = webgpu::g_depthBuffer.size = {64,64,1};
    metadata.resize(gfx::UberStateBufferSize); attributes.resize(gfx::StorageBufferSize);
    gfx::detail::testing::suppress_render_worker(true);
    begin();
  }
  void begin() {
    frame.uberStates = ByteBuffer(metadata.data(), metadata.size());
    frame.storage = ByteBuffer(attributes.data(), attributes.size());
    gfx::detail::begin_recording(frame,0);
  }
  void end() { gfx::finish(); gfx::detail::end_recording(); }
  void TearDown() override {
    end(); gfx::detail::shutdown_recording();
    gfx::detail::resources().shaderMode = AURORA_SHADER_SPECIALIZED;
    gfx::detail::testing::suppress_render_worker(false);
  }
  gfx::detail::FramePacket frame;
  std::vector<u8> metadata, attributes;
};
TEST_F(GxShaderArena, RejectsWithoutMutationAndPreservesEntireAttributeCapacity) {
  const std::array<u8,3> a{1,2,3};
  const auto first = gfx::try_push_uber_state(a); ASSERT_TRUE(first);
  EXPECT_EQ(first->offset, 0u);
  const auto again = gfx::try_push_uber_state(a); ASSERT_TRUE(again); EXPECT_EQ(*first,*again);
  std::vector<u8> tail(gfx::UberStateBufferSize - 4, 7);
  const auto last = gfx::try_push_uber_state(tail); ASSERT_TRUE(last); EXPECT_EQ(last->offset,4u);
  const auto size = frame.uberStates.size(); const auto before = metadata;
  const std::array<u8,1> rejected{9}; EXPECT_FALSE(gfx::try_push_uber_state(rejected));
  EXPECT_EQ(size, frame.uberStates.size()); EXPECT_EQ(before, metadata);
  std::vector<u8> all(gfx::StorageBufferSize, 0xa5);
  const auto full = gfx::push_storage(all.data(),all.size());
  EXPECT_EQ(full.offset,0u); EXPECT_EQ(full.size,gfx::StorageBufferSize); EXPECT_EQ(attributes,all);
  const auto oldEpoch = gfx::recording_upload_epoch(); end(); frame = {}; begin();
  EXPECT_GT(gfx::recording_upload_epoch(), oldEpoch);
  EXPECT_EQ(frame.frameIndex,0u);
  const auto fresh = gfx::try_push_uber_state(a); ASSERT_TRUE(fresh); EXPECT_EQ(fresh->offset,0u);
}
TEST_F(GxShaderArena, SharedOffsetsAddressTheActualUniformWriter) {
  auto c=config().shaderConfig;
  c.lineMode=3;c.fogType=GX_FOG_PERSP_LIN;c.zTextureOp=GX_ZT_ADD;
  auto& stage=c.tevStages[0];stage.texMapId=GX_TEXMAP0;stage.texCoordId=GX_TEXCOORD0;
  stage.colorPass={GX_CC_RASC,GX_CC_KONST,GX_CC_HALF,GX_CC_C0};stage.channelId=GX_COLOR0A0;
  stage.kcSel=GX_TEV_KCSEL_K0;c.tcgs[0].postMtx=GX_PTTEXMTX0;
  c.colorChannels[0].lightingEnabled=true;
  c.numIndStages=1;c.indStages[0]={GX_TEXCOORD0,GX_TEXMAP0,GX_ITS_1,GX_ITS_1};stage.indTexMtxId=GX_ITM_0;
  auto& state=gx::g_gxState;
  state.colorChannelConfig=c.colorChannels;
  state.renderViewport={0,0,64,48,0,1};state.logicalViewport=state.renderViewport;
  state.pointSize=18;
  state.proj={{1,2,3,4},{5,6,7,8},{9,10,11,12},{13,14,15,16}};
  state.pnMtx[0].pos={{21,22,23,24},{25,26,27,28},{29,30,31,32}};
  state.pnMtx[0].nrm={{31,32,33,34},{35,36,37,38},{39,40,41,42}};
  state.colorRegs[1]={0.1f,0.2f,0.3f,0.4f};state.kcolors[0]={0.6f,0.7f,0.8f,0.9f};
  state.fog.a=0.125f;state.fog.b=1.25f;state.fog.c=0.25f;state.zTextureBias=0x876543;
  state.colorChannelState[0].ambColor={0.2f,0.3f,0.4f,0.5f};
  state.colorChannelState[0].matColor={0.3f,0.4f,0.5f,0.6f};
  state.texCoordScales[0].scaleS=12;state.texCoordScales[0].scaleT=18;
  const auto info=gx::build_shader_info(c);
  const auto uniform=gx::build_uniform(info);
  const auto* bytes=gfx::uniform_data(uniform);
  const auto scalar=[&](u32 offset) { float value;std::memcpy(&value,bytes+offset,4);return value; };
  EXPECT_EQ(uniform.size,info.offsets.end);
  EXPECT_EQ(scalar(0),64);EXPECT_EQ(scalar(16),3);
  EXPECT_EQ(scalar(info.offsets.projection),1);
  EXPECT_EQ(scalar(info.offsets.projection+32),-9);
  EXPECT_EQ(scalar(info.offsets.postex),21);EXPECT_EQ(scalar(info.offsets.nrm),31);
  EXPECT_EQ(scalar(info.offsets.tevreg[1]),0.1f);
  EXPECT_EQ(scalar(info.offsets.ambient[0]),0.2f);EXPECT_EQ(scalar(info.offsets.material[0]),0.3f);
  EXPECT_EQ(scalar(info.offsets.kcolor[0]),0.6f);
  EXPECT_EQ(scalar(info.offsets.fog+16),0.125f);
  u32 bias;std::memcpy(&bias,bytes+info.offsets.ztexture,4);EXPECT_EQ(bias,0x876543u);
  EXPECT_EQ(scalar(info.offsets.texcoordScale[0]),13);EXPECT_EQ(scalar(info.offsets.texcoordScale[0]+4),19);
}
TEST(GxShaderMode, InvalidAndNextStartRequestsDoNotProvisionResources) {
  aurora_set_shader_mode(static_cast<AuroraShaderMode>(99));EXPECT_EQ(aurora_get_shader_mode(),AURORA_SHADER_SPECIALIZED);
  aurora_set_shader_mode(AURORA_SHADER_HYBRID);EXPECT_EQ(aurora_get_shader_mode(),AURORA_SHADER_HYBRID);
  EXPECT_FALSE(gfx::detail::resources().uberStateBuffer);
  EXPECT_EQ(gfx::detail::resources().shaderMode,AURORA_SHADER_SPECIALIZED);
  aurora_set_shader_mode(AURORA_SHADER_SPECIALIZED);
}
TEST(GxShaderState, MissingLookupDoesNotWaitInRequiredScope) {
  gfx::CompiledPipeline result;
  gfx::RequireReadyPipelinesScope required(true);
  EXPECT_EQ(gfx::pipeline_build_state(0), gfx::PipelineBuildState::Missing);
  EXPECT_FALSE(gfx::try_get_pipeline(0,result));
  EXPECT_FALSE(gfx::get_pipeline_ready(0,result));
}
} // namespace
