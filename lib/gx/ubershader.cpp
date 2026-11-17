#include "ubershader.hpp"
#include "shader_info.hpp"
#include "shader_common_wgsl.hpp"
#include "../gfx/recording.hpp"
#include "../gfx/resources.hpp"
#include "../gfx/frame_packet.hpp"
#include "../webgpu/gpu.hpp"
#include <atomic>
#include <cstddef>
#include <chrono>
#include <mutex>
namespace aurora::gx {
namespace {
constexpr Module Log("aurora::gx::uber");
std::mutex moduleMutex;
// Two depth signatures, three alpha signatures, and bounded attachment locations.
constexpr size_t ModuleCount = 6 * (gfx::MaxColorAttachments + 1);
std::array<wgpu::ShaderModule, ModuleCount> modules;
std::atomic<testing::ShaderSelection> selection{testing::ShaderSelection::Automatic};
std::array<std::atomic<u64>, 5> counters{};
std::array<bool, ModuleCount> moduleAttempted{};
std::atomic_bool invalidGenericSource = false;
class CreationErrors {
  bool finished = false;
public:
  CreationErrors(const CreationErrors&) = delete;
  CreationErrors& operator=(const CreationErrors&) = delete;
  CreationErrors() {
    for (auto filter : {wgpu::ErrorFilter::OutOfMemory, wgpu::ErrorFilter::Internal, wgpu::ErrorFilter::Validation})
      webgpu::g_device.PushErrorScope(filter);
  }
  bool finish() {
    finished = true;
    bool success = true;
    for (int i = 0; i < 3; ++i) {
      const auto valid = std::make_shared<std::atomic_bool>(false);
      const auto future = webgpu::g_device.PopErrorScope(wgpu::CallbackMode::WaitAnyOnly,
          [valid](wgpu::PopErrorScopeStatus status, wgpu::ErrorType type, wgpu::StringView message) {
            const bool ok = status == wgpu::PopErrorScopeStatus::Success && type == wgpu::ErrorType::NoError;
            valid->store(ok);
            if (!ok) Log.warn("GX pipeline creation failed: {}", message);
          });
      const auto status = webgpu::g_instance.WaitAny(future, 5'000'000'000);
      if (status != wgpu::WaitStatus::Success || !valid->load()) success = false;
    }
    return success;
  }
  ~CreationErrors() { if (!finished) finish(); }
};
template <typename Create>
auto checked_creation(Create&& create) {
  CreationErrors errors;
  auto result = create();
  if (!errors.finish()) return decltype(result){};
  return result;
}
std::string declarations() {
  std::string text = "diagnostic(off, derivative_uniformity);\n";
#define OFFSET(type, field) text += fmt::format("const " #type "_" #field ":u32 = {}u;\n", offsetof(type, field))
#define SIZE(type) text += fmt::format("const " #type "_size:u32 = {}u;\n", sizeof(type))
  OFFSET(ShaderConfig, fogType);
  OFFSET(ShaderConfig, vtxStride);
  OFFSET(ShaderConfig, attrs);
  OFFSET(ShaderConfig, tevSwapTable);
  OFFSET(ShaderConfig, tevStages);
  OFFSET(ShaderConfig, tevStageCount);
  OFFSET(ShaderConfig, colorChannels);
  OFFSET(ShaderConfig, tcgs);
  OFFSET(ShaderConfig, alphaCompare);
  OFFSET(ShaderConfig, indStages);
  OFFSET(ShaderConfig, numIndStages);
  SIZE(ShaderConfig);
  OFFSET(AttrConfig, attrType);
  OFFSET(AttrConfig, cnt);
  OFFSET(AttrConfig, compType);
  OFFSET(AttrConfig, offset);
  OFFSET(AttrConfig, stride);
  OFFSET(AttrConfig, frac);
  OFFSET(AttrConfig, le);
  OFFSET(AttrConfig, nbt3);
  SIZE(AttrConfig);
  using TevPass = aurora::gx::TevPass<GXTevColorArg, GX_CC_ZERO>;
  OFFSET(TevPass, a);
  OFFSET(TevPass, b);
  OFFSET(TevPass, c);
  OFFSET(TevPass, d);
  SIZE(TevPass);
  OFFSET(TevOp, op);
  OFFSET(TevOp, bias);
  OFFSET(TevOp, scale);
  OFFSET(TevOp, outReg);
  OFFSET(TevOp, clamp);
  SIZE(TevOp);
  OFFSET(TevStage, colorPass);
  OFFSET(TevStage, alphaPass);
  OFFSET(TevStage, colorOp);
  OFFSET(TevStage, alphaOp);
  OFFSET(TevStage, kcSel);
  OFFSET(TevStage, kaSel);
  OFFSET(TevStage, texCoordId);
  OFFSET(TevStage, texMapId);
  OFFSET(TevStage, channelId);
  OFFSET(TevStage, tevSwapRas);
  OFFSET(TevStage, tevSwapTex);
  OFFSET(TevStage, indTexStage);
  OFFSET(TevStage, indTexFormat);
  OFFSET(TevStage, indTexBiasSel);
  OFFSET(TevStage, indTexAlphaSel);
  OFFSET(TevStage, indTexMtxId);
  OFFSET(TevStage, indTexWrapS);
  OFFSET(TevStage, indTexWrapT);
  OFFSET(TevStage, indTexUseOrigLOD);
  OFFSET(TevStage, indTexAddPrev);
  SIZE(TevStage);
  OFFSET(ColorChannelConfig, matSrc);
  OFFSET(ColorChannelConfig, ambSrc);
  OFFSET(ColorChannelConfig, diffFn);
  OFFSET(ColorChannelConfig, attnFn);
  OFFSET(ColorChannelConfig, lightingEnabled);
  SIZE(ColorChannelConfig);
  OFFSET(TcgConfig, type);
  OFFSET(TcgConfig, src);
  OFFSET(TcgConfig, embossSrc);
  OFFSET(TcgConfig, mtx);
  OFFSET(TcgConfig, postMtx);
  OFFSET(TcgConfig, normalize);
  OFFSET(TcgConfig, inputFormAB11);
  SIZE(TcgConfig);
  OFFSET(TevSwap, red);
  OFFSET(TevSwap, green);
  OFFSET(TevSwap, blue);
  OFFSET(TevSwap, alpha);
  SIZE(TevSwap);
  OFFSET(AlphaCompare, comp0);
  OFFSET(AlphaCompare, ref0);
  OFFSET(AlphaCompare, op);
  OFFSET(AlphaCompare, comp1);
  OFFSET(AlphaCompare, ref1);
  SIZE(AlphaCompare);
  OFFSET(IndStage, texCoordId);
  OFFSET(IndStage, texMapId);
  OFFSET(IndStage, scaleS);
  OFFSET(IndStage, scaleT);
  SIZE(IndStage);
  OFFSET(UberShaderState, uniforms);
  OFFSET(UberUniformOffsets, projection);
  OFFSET(UberUniformOffsets, postex);
  OFFSET(UberUniformOffsets, nrm);
  OFFSET(UberUniformOffsets, tevreg);
  OFFSET(UberUniformOffsets, lights);
  OFFSET(UberUniformOffsets, lightState);
  OFFSET(UberUniformOffsets, ambient);
  OFFSET(UberUniformOffsets, material);
  OFFSET(UberUniformOffsets, kcolor);
  OFFSET(UberUniformOffsets, postmtx);
  OFFSET(UberUniformOffsets, fog);
  OFFSET(UberUniformOffsets, indmtx);
  OFFSET(UberUniformOffsets, texcoordScale);
  OFFSET(UberUniformOffsets, texSizeBias);
  OFFSET(UberUniformOffsets, ztexture);
  OFFSET(UberShaderState, flags);
  OFFSET(UberShaderState, sampledTexCoords);
  OFFSET(UberShaderState, sampledColorChannels);
  OFFSET(UberShaderState, usedIndStages);
  SIZE(UberUniformOffsets);
#undef SIZE
#undef OFFSET
  text += fmt::format("struct UberUniformBuffer {{ words:array<vec4u,{}> }};\n", MaxUniformSize / 16);
  text += R"(
@group(0) @binding(0) var<storage,read> vbuf:array<u32>;
@group(0) @binding(1) var<storage,read> abuf:array<u32>;
@group(0) @binding(2) var<storage,read> sbuf:array<u32>;
struct Immediate {
  vtx_start:u32, current_pnmtx:u32, fog_range_base:u32, uber_state_offset:u32,
  array_start0:vec4u, array_start1:vec4u, array_start2:vec4u,
};
var<immediate> imm:Immediate;
fn array_start(index:u32)->u32 {
  switch index / 4u {
    case 0u: { return imm.array_start0[index & 3u]; }
    case 1u: { return imm.array_start1[index & 3u]; }
    default: { return imm.array_start2[index & 3u]; }
  }
}
@group(1) @binding(0) var<uniform> ubuf:UberUniformBuffer;
fn uniform_u32(offset:u32)->u32 {
  if (offset == 0xffffffffu) { return 0u; }
  return ubuf.words[offset >> 4u][(offset & 15u) >> 2u];
}
fn uniform_f32(offset:u32)->f32 { return bitcast<f32>(uniform_u32(offset)); }
fn uniform_vec4(offset:u32)->vec4f {
  if (offset == 0xffffffffu) { return vec4f(0.0); }
  return bitcast<vec4f>(ubuf.words[offset >> 4u]);
}
fn uniform_mat3x4(offset:u32)->mat3x4f {
  return mat3x4f(uniform_vec4(offset), uniform_vec4(offset+16u), uniform_vec4(offset+32u));
}
fn uniform_mat4x4(offset:u32)->mat4x4f {
  return mat4x4f(uniform_vec4(offset), uniform_vec4(offset+16u), uniform_vec4(offset+32u), uniform_vec4(offset+48u));
}
fn cfg_u32(offset:u32)->u32 { return sbuf[(imm.uber_state_offset+offset) >> 2u]; }
fn cfg_u8(offset:u32)->u32 { return (cfg_u32(offset & ~3u) >> ((offset & 3u)*8u)) & 255u; }
fn uniform_offset(field:u32)->u32 { return cfg_u32(UberShaderState_uniforms+field); }
struct UberVertexOutput {
  @builtin(position) pos:vec4f,
  @location(0) cc0:vec4f,
  @location(1) cc1:vec4f,
  @location(2) tex0:vec3f,
  @location(3) tex1:vec3f,
  @location(4) tex2:vec3f,
  @location(5) tex3:vec3f,
  @location(6) tex4:vec3f,
  @location(7) tex5:vec3f,
  @location(8) tex6:vec3f,
  @location(9) tex7:vec3f,
  @location(10) mv_nrm:vec3f,
};
)";
  for (u32 i = 0; i < MaxTextures; ++i) {
    text += fmt::format(
        "@group(2) @binding({}) var tex{}:texture_2d<f32>;\n@group(2) @binding({}) var tex{}_samp:sampler;\n", i * 2, i,
        i * 2 + 1, i);
  }
  return text;
}
} // namespace
UberShaderState build_uber_shader_state(const ShaderConfig& config, const ShaderInfo& info) {
  return {config, info.offsets,
          u32(config.lineMode) | (u32(config.fogRangeEnabled) << 2) | (u32(config.zTextureOp) << 3) |
              (u32(config.zTextureFormat) << 5) | (u32(config.forceAlphaOne) << 7) |
              (u32(config.attrs[GX_VA_NRM].attrType != GX_NONE) << 8),
          u32(info.sampledTexCoords.to_ulong()), u32(info.sampledColorChannels.to_ulong()),
          u32(info.usedIndStages.to_ulong())};
}
GenericCoverageReason uber_coverage(const ShaderConfig& config, const ShaderInfo& info,
                                   const gfx::RenderTargetLayout& layout) noexcept {
  if (!gfx::detail::resources().uberStateBuffer) return GenericCoverageReason::UnsupportedDevice;
  if (config.tevStageCount == 0 || config.tevStageCount > MaxTevStages || info.uniformSize > MaxUniformSize)
    return GenericCoverageReason::UnsupportedConfig;
  if (config.zTextureOp > GX_ZT_REPLACE || config.fogType == 8 || config.fogType > 15)
    return GenericCoverageReason::UnsupportedConfig;
  if (config.zTextureOp != GX_ZT_DISABLE && webgpu::g_backendType == wgpu::BackendType::Vulkan)
    return GenericCoverageReason::UnsupportedOutput;
  for (u32 i = 0; i < config.tevStageCount; ++i) {
    const auto& stage = config.tevStages[i];
    const auto validOp = [](const TevOp& op) {
      return (op.op == GX_TEV_ADD || op.op == GX_TEV_SUB ||
              (op.op >= GX_TEV_COMP_R8_GT && op.op <= GX_TEV_COMP_RGB8_EQ)) &&
             op.bias <= GX_TB_SUBHALF && op.scale <= GX_CS_DIVIDE_2 && op.outReg <= GX_TEVREG2;
    };
    if (!validOp(stage.colorOp) || !validOp(stage.alphaOp) ||
        stage.indTexWrapS > GX_ITW_0 || stage.indTexWrapT > GX_ITW_0)
      return GenericCoverageReason::UnsupportedConfig;
  }
  const auto plan = make_pipeline_plan(PipelineConfig{.shaderConfig = config}, layout);
  // The current specialized normal output has no varying when lighting is enabled.
  if (plan.normalAttachment != UINT32_MAX && info.lightingEnabled && config.attrs[GX_VA_NRM].attrType != GX_NONE)
    return GenericCoverageReason::UnsupportedOutput;
  for (u32 i = 0; i < MaxTexCoord; ++i) {
    if (!info.sampledTexCoords[i]) continue;
    const auto& tcg = config.tcgs[i];
    const bool emboss = tcg.type >= GX_TG_BUMP0 && tcg.type <= GX_TG_BUMP7;
    if (emboss && (tcg.embossSrc >= i || !info.sampledTexCoords[tcg.embossSrc] ||
                  config.attrs[GX_VA_NRM].cnt != 9 ||
                  (config.tcgs[tcg.embossSrc].type >= GX_TG_BUMP0 && config.tcgs[tcg.embossSrc].type <= GX_TG_BUMP7))) return GenericCoverageReason::UnsupportedConfig;
    if (tcg.src >= GX_TG_TEXCOORD0 && tcg.src <= GX_TG_TEXCOORD6) {
      const auto source = tcg.src - GX_TG_TEXCOORD0;
      if (source >= i || !info.sampledTexCoords[source] ||
          (config.tcgs[source].type >= GX_TG_BUMP0 && config.tcgs[source].type <= GX_TG_BUMP7))
        return GenericCoverageReason::UnsupportedConfig;
    }
    if ((tcg.src == GX_TG_COLOR0 && !info.sampledColorChannels[0]) ||
        (tcg.src == GX_TG_COLOR1 && !info.sampledColorChannels[1])) return GenericCoverageReason::UnsupportedConfig;
    if ((tcg.src == GX_TG_BINRM || tcg.src == GX_TG_TANGENT) && config.attrs[GX_VA_NRM].cnt != 9)
      return GenericCoverageReason::UnsupportedConfig;
  }
  return GenericCoverageReason::Covered;
}
UberPipelineKey uber_pipeline_key(const PipelineConfig& config, const gfx::RenderTargetLayout& layout) {
  const auto plan = make_pipeline_plan(config, layout);
  return {layout.key, UberShaderABI, u32(config.depthFunc), u32(config.cullMode), u32(config.blendMode),
          u32(config.blendFacSrc), u32(config.blendFacDst), u32(config.blendOp), config.dstAlpha,
          config.polygonOffsetBits, config.polygonOffsetScaleBits, config.polygonOffsetClampBits,
          u32(config.depthCompare) | (u32(config.depthUpdate) << 1) | (u32(config.alphaUpdate) << 2) |
              (u32(config.colorUpdate) << 3) | (u32(config.shaderConfig.zTextureOp != GX_ZT_DISABLE) << 4) |
              (u32(plan.hasPrepass) << 5),
          u32(plan.main.dstAlphaMode), u32(plan.prepass.dstAlphaMode), plan.normalAttachment};
}
std::string build_uber_shader_source(const UberModuleKey& key) {
  auto source = declarations();
  if (key.dstAlphaMode == DstAlphaMode::DualSource) source.insert(0, "enable dual_source_blending;\n");
  source += ShaderCommonWGSL;
  source += uber_vertex_wgsl();
  source += uber_fragment_wgsl(key);
  return source;
}
wgpu::ShaderModule build_uber_shader(const UberModuleKey& key) {
  if (key.depthOutput > 1 || u32(key.dstAlphaMode) > 2 ||
      (key.normalAttachment != UINT32_MAX && (key.normalAttachment == 0 || key.normalAttachment >= gfx::MaxColorAttachments)) ||
      (key.normalAttachment != UINT32_MAX && key.dstAlphaMode == DstAlphaMode::DualSource)) return {};
  const u32 attachment = key.normalAttachment == UINT32_MAX ? 0 : key.normalAttachment + 1;
  const auto index = (attachment * 3 + u32(key.dstAlphaMode)) * 2 + key.depthOutput;
  std::lock_guard lock{moduleMutex};
  if (!moduleAttempted[index]) {
    moduleAttempted[index] = true;
    const auto source = build_uber_shader_source(key);
    wgpu::ShaderSourceWGSL wgsl{};
    wgsl.code = invalidGenericSource.load() ? "invalid generic shader source" : source.c_str();
    const auto label = fmt::format("GX uber ABI {} signature {}", UberShaderABI, index);
    const wgpu::ShaderModuleDescriptor descriptor{.nextInChain = &wgsl, .label = label.c_str()};
    modules[index] = checked_creation([&] { return webgpu::g_device.CreateShaderModule(&descriptor); });
  }
  return modules[index];
}
gfx::CompiledPipeline create_uber_pipeline(const PipelineConfig& config, const gfx::RenderTargetLayout& layout) {
  return checked_creation([&]() -> gfx::CompiledPipeline {
    const auto plan = make_pipeline_plan(config, layout);
    const auto create = [&](const PipelineOptions& options) {
      const auto shader = build_uber_shader({u32(config.shaderConfig.zTextureOp != GX_ZT_DISABLE),
                                            options.dstAlphaMode, {}, plan.normalAttachment});
      if (!shader) return wgpu::RenderPipeline{};
      return build_pipeline(config, layout, options, {}, shader, "GX uber pipeline ABI 1");
    };
    gfx::CompiledPipeline result;
    if (plan.hasPrepass) {
      result.prepass = create(plan.prepass);
      if (!result.prepass) return {};
    }
    result.main = create(plan.main);
    return result.main ? result : gfx::CompiledPipeline{};
  });
}
gfx::CompiledPipeline create_hybrid_specialization(const PipelineConfig& config, const gfx::RenderTargetLayout& layout) {
  return checked_creation([&] { return create_pipeline(config, layout); });
}
void initialize_ubershaders() {
  if (!gfx::detail::resources().uberStateBuffer) return;
  const auto started = std::chrono::steady_clock::now();
  const auto scene = gfx::scene_render_target_layout();
  auto offscreen = scene;
  offscreen.colorAttachmentCount = 1;
  offscreen.sampleCount = 1;
  gfx::detail::finalize_render_target_layout(offscreen);
  PipelineConfig config{};
  config.depthFunc = GX_LEQUAL;
  config.cullMode = GX_CULL_BACK;
  config.blendMode = GX_BM_NONE;
  config.blendFacSrc = GX_BL_ONE;
  config.blendFacDst = GX_BL_ZERO;
  config.blendOp = GX_LO_CLEAR;
  config.dstAlpha = UINT32_MAX;
  config.depthCompare = config.depthUpdate = config.colorUpdate = true;
  for (const auto& layout : {scene, offscreen}) {
    for (u32 depth : {0u, 1u}) {
      config.shaderConfig.zTextureOp = depth ? GX_ZT_REPLACE : GX_ZT_DISABLE;
      gfx::CompiledPipeline pipeline;
      if (!gfx::get_pipeline_ready(gfx::find_uber_pipeline(config, layout), pipeline))
        Log.warn("Generic startup pipeline unavailable; draws will require specialization");
    }
  }
  const auto elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
  Log.info("Generic startup warmup: {:.2f} ms", elapsed);
}
void shutdown_ubershaders() {
  std::lock_guard lock{moduleMutex};
  modules = {};
  moduleAttempted.fill(false);
  invalidGenericSource.store(false);
  selection.store(testing::ShaderSelection::Automatic);
  testing::set_after_prepass_hook(nullptr);
}
namespace testing {
void invalidate_generic_source(bool invalid) noexcept { invalidGenericSource.store(invalid); }
void set_shader_selection(ShaderSelection value) noexcept { selection.store(value); }
ShaderSelection shader_selection() noexcept { return selection.load(); }
HybridCounters hybrid_counters() noexcept { return {counters[0].load(), counters[1].load(), counters[2].load(), counters[3].load(), counters[4].load()}; }
void reset_hybrid_counters() noexcept { for (auto& counter : counters) counter.store(0); }
void count_specialized() noexcept { ++counters[0]; }
void count_generic() noexcept { ++counters[1]; }
void count_capacity_fallback() noexcept { ++counters[2]; }
void count_unsupported_fallback() noexcept { ++counters[3]; }
void count_failed_draw() noexcept { ++counters[4]; }
}
}
