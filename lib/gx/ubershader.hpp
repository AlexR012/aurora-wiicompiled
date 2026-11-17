#pragma once
#include "pipeline.hpp"
#include <string>
namespace aurora::gx {
inline constexpr u32 UberShaderABI = 1;
using UberUniformOffsets = UniformOffsets;
struct UberShaderState {
  ShaderConfig config;
  UberUniformOffsets uniforms;
  u32 flags, sampledTexCoords, sampledColorChannels, usedIndStages;
};
static_assert(std::has_unique_object_representations_v<UberShaderState>);
static_assert(std::is_standard_layout_v<UberShaderState> && offsetof(UberShaderState, config) == 0);
struct UberModuleKey {
  u32 depthOutput;
  DstAlphaMode dstAlphaMode;
  u8 reserved[3]{};
  u32 normalAttachment = UINT32_MAX;
};
static_assert(std::has_unique_object_representations_v<UberModuleKey>);
struct UberPipelineKey {
  u64 layout;
  u32 abi, depthFunc, cullMode, blendMode, blendFacSrc, blendFacDst, blendOp, dstAlpha;
  u32 polygonOffsetBits, polygonOffsetScaleBits, polygonOffsetClampBits;
  u32 flags, mainAlphaMode, prepassAlphaMode, normalAttachment, reserved = 0;
};
static_assert(std::has_unique_object_representations_v<UberPipelineKey>);
enum class GenericCoverageReason : u8 { Covered, UnsupportedConfig, UnsupportedOutput, UnsupportedDevice, MetadataCapacity };
UberShaderState build_uber_shader_state(const ShaderConfig&, const ShaderInfo&);
GenericCoverageReason uber_coverage(const ShaderConfig&, const ShaderInfo&, const gfx::RenderTargetLayout&) noexcept;
UberPipelineKey uber_pipeline_key(const PipelineConfig&, const gfx::RenderTargetLayout&);
std::string uber_vertex_wgsl();
std::string uber_fragment_wgsl(const UberModuleKey&);
std::string build_uber_shader_source(const UberModuleKey&);
wgpu::ShaderModule build_uber_shader(const UberModuleKey&);
gfx::CompiledPipeline create_hybrid_specialization(const PipelineConfig&, const gfx::RenderTargetLayout&);
gfx::CompiledPipeline create_uber_pipeline(const PipelineConfig&, const gfx::RenderTargetLayout&);
void initialize_ubershaders();
void shutdown_ubershaders();
namespace testing {
using AfterPrepassHook = void (*)();
void set_after_prepass_hook(AfterPrepassHook) noexcept;
void invalidate_generic_source(bool) noexcept;
enum class ShaderSelection : u8 { Automatic, Specialized, Generic };
void set_shader_selection(ShaderSelection) noexcept;
ShaderSelection shader_selection() noexcept;
struct HybridCounters { u64 specialized, generic, capacityFallback, unsupportedFallback, failedDraw; };
HybridCounters hybrid_counters() noexcept;
void reset_hybrid_counters() noexcept;
void count_specialized() noexcept;
void count_generic() noexcept;
void count_capacity_fallback() noexcept;
void count_unsupported_fallback() noexcept;
void count_failed_draw() noexcept;
}
}
