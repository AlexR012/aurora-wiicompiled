#pragma once

#include "../gfx/pipeline_cache.hpp"
#include "gx.hpp"

namespace aurora::gx {
enum class DrawShaderSelection : u8 { Specialized, Hybrid, GenericOnly, RequiredSpecialized };
struct DrawData {
  gfx::PipelineRef pipeline;
  gfx::PipelineRef fallbackPipeline = 0;
  GXBindGroups bindGroups;
  uint32_t vertSize;
  gfx::Range idxRange;
  uint32_t uniformOffset;
  uint16_t uniformSize;
  DrawShaderSelection selection = DrawShaderSelection::Specialized;
  uint8_t reserved = 0;
  // First interpolated copy of the uniform block; slot k follows at k aligned block sizes.
  uint32_t interpUniformBase = UINT32_MAX;
  DrawImmediateData immediateData;
  uint32_t vtxCount;
  uint32_t indexCount;
  uint32_t instanceCount;
  uint32_t dstAlpha;
};

static_assert(std::is_trivially_copyable_v<DrawData> && std::is_aggregate_v<DrawData>);
static_assert(sizeof(DrawData) <= gfx::InlineDrawPayloadSize);

constexpr uint32_t GXPipelineConfigVersion = 14;
struct PipelineConfig {
  uint32_t version = GXPipelineConfigVersion;
  uint32_t msaaSamples = 1; // deprecated
  ShaderConfig shaderConfig;
  GXCompare depthFunc;
  GXCullMode cullMode;
  GXBlendMode blendMode;
  GXBlendFactor blendFacSrc, blendFacDst;
  GXLogicOp blendOp;
  uint32_t dstAlpha;
  uint32_t polygonOffsetBits;
  uint32_t polygonOffsetScaleBits;
  uint32_t polygonOffsetClampBits;
  bool depthCompare, depthUpdate, alphaUpdate, colorUpdate;
};
static_assert(std::has_unique_object_representations_v<PipelineConfig>);

struct PipelineOptions {
  DstAlphaMode dstAlphaMode = DstAlphaMode::None;
  bool colorUpdate = false;
  bool alphaUpdate = false;
  bool depthUpdate = false;
};

struct PipelinePlan {
  PipelineOptions main, prepass;
  bool hasPrepass = false;
  u32 normalAttachment = UINT32_MAX;
};
PipelinePlan make_pipeline_plan(const PipelineConfig&, const gfx::RenderTargetLayout&);
gfx::CompiledPipeline create_pipeline(const PipelineConfig& config, const gfx::RenderTargetLayout& layout);
void render(const DrawData& data, const wgpu::RenderPassEncoder& pass);

void queue_surface(const u8* dlStart, uint32_t dlSize, bool bigEndian) noexcept;
} // namespace aurora::gx
