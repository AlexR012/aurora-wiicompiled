#include "pipeline.hpp"
#include "ubershader.hpp"
#include "frame_interpolation.hpp"

#include "../gfx/encoding.hpp"
#include "../gfx/recording.hpp"
#include "../gfx/resources.hpp"
#include "../gfx/pipeline_cache.hpp"
#include "../gfx/resource_cache.hpp"

#include "gx_fmt.hpp"
#include "shader_info.hpp"

#include <tracy/Tracy.hpp>
#include <atomic>

namespace aurora::gx {
namespace { std::atomic<testing::AfterPrepassHook> afterPrepassHook{nullptr}; }
namespace testing {
void set_after_prepass_hook(AfterPrepassHook hook) noexcept { afterPrepassHook.store(hook); }
}


PipelinePlan make_pipeline_plan(const PipelineConfig& config, const gfx::RenderTargetLayout& layout) {
  PipelinePlan plan;
  plan.main = {.colorUpdate = config.colorUpdate, .alphaUpdate = config.alphaUpdate, .depthUpdate = config.depthUpdate};
  for (u32 i = 0; i < layout.colorAttachmentCount; ++i)
    if (layout.colorAttachments[i].semantic == gfx::ColorAttachmentSemantic::Normal) plan.normalAttachment = i;
  if (config.alphaUpdate && config.dstAlpha != UINT32_MAX && config.dstAlpha != 0) {
    const auto usesSourceAlpha = [](GXBlendFactor factor) {
      return factor == GX_BL_SRCALPHA || factor == GX_BL_INVSRCALPHA;
    };
    const bool needsSourceAlpha = config.colorUpdate && config.blendMode == GX_BM_BLEND &&
                                 (usesSourceAlpha(config.blendFacSrc) || usesSourceAlpha(config.blendFacDst));
    if (!needsSourceAlpha) {
      plan.main.dstAlphaMode = DstAlphaMode::Replace;
    } else if (webgpu::g_dualSourceBlendingSupported && layout.colorAttachmentCount == 1) {
      plan.main.dstAlphaMode = DstAlphaMode::DualSource;
    } else {
      plan.hasPrepass = true;
      plan.prepass = plan.main;
      plan.prepass.dstAlphaMode = DstAlphaMode::Replace;
      plan.prepass.colorUpdate = false;
      plan.prepass.depthUpdate = false;
      plan.main.alphaUpdate = false;
    }
  }
  return plan;
}

gfx::CompiledPipeline create_pipeline(const PipelineConfig& config, const gfx::RenderTargetLayout& layout) {
  ZoneScoped;
  const auto plan = make_pipeline_plan(config, layout);
  const auto hash = xxh3_hash(layout.key, xxh3_hash(config, static_cast<HashType>(gfx::ShaderType::GX)));
  const auto create = [&](const PipelineOptions& options, const char* passName) {
    const auto shader = build_shader(config.shaderConfig, layout, options.dstAlphaMode);
    const auto label = fmt::format("GX Pipeline {:x} {}", hash, passName);
    return build_pipeline(config, layout, options, {}, shader, label.c_str());
  };
  gfx::CompiledPipeline result;
  if (plan.hasPrepass) {
    result.prepass = create(plan.prepass, "alpha");
    if (!result.prepass) return {};
  }
  result.main = create(plan.main, "main");
  return result.main ? result : gfx::CompiledPipeline{};
}

void render(const DrawData& data, const wgpu::RenderPassEncoder& pass) {
  gfx::CompiledPipeline pipeline;
  bool generic = false;
  switch (data.selection) {
  case DrawShaderSelection::Specialized:
    if (!gfx::get_pipeline(data.pipeline, pipeline)) return;
    break;
  case DrawShaderSelection::RequiredSpecialized:
    if (!gfx::get_pipeline_ready(data.pipeline, pipeline)) {
      testing::count_failed_draw();
      Module("aurora::gx").fatal("Required GX specialization failed");
    }
    break;
  case DrawShaderSelection::Hybrid:
  case DrawShaderSelection::GenericOnly:
    if (data.selection == DrawShaderSelection::Hybrid && gfx::try_get_pipeline(data.pipeline, pipeline)) break;
    if (data.fallbackPipeline && gfx::get_pipeline_ready(data.fallbackPipeline, pipeline)) {
      generic = true;
      break;
    }
    if (data.selection == DrawShaderSelection::GenericOnly || !gfx::get_pipeline_ready(data.pipeline, pipeline)) {
      testing::count_failed_draw();
      Module("aurora::gx").fatal("No usable GX pipeline for recorded draw");
    }
    break;
  }
  if (data.selection != DrawShaderSelection::Specialized) {
    if (generic) testing::count_generic(); else testing::count_specialized();
  }
  const auto& resources = gfx::detail::resources();
  pass.SetImmediates(0, &data.immediateData, sizeof(data.immediateData));
  uint32_t uniformOffset = data.uniformOffset;
  if (const int slot = g_interpolationEncodeSlot; slot >= 0 && data.interpUniformBase != UINT32_MAX) {
    uniformOffset = data.interpUniformBase + static_cast<uint32_t>(slot) * gfx::align_uniform(data.uniformSize);
  }
  const std::array offsets{uniformOffset};
  pass.SetBindGroup(1, resources.uniformBindGroup, offsets.size(), offsets.data());
  if (data.bindGroups.textureBindGroup) {
    pass.SetBindGroup(2, gfx::find_bind_group(data.bindGroups.textureBindGroup));
  } else if (generic) {
    pass.SetBindGroup(2, g_emptyTextureBindGroup);
  }
  pass.SetIndexBuffer(resources.indexBuffer, wgpu::IndexFormat::Uint16, data.idxRange.offset, data.idxRange.size);
  if (data.dstAlpha != UINT32_MAX) {
    const wgpu::Color color{0.f, 0.f, 0.f, data.dstAlpha / 255.f};
    pass.SetBlendConstant(&color);
  }
  const auto draw = [&] {
    if (data.indexCount == 0) {
      pass.Draw(data.vtxCount, data.instanceCount);
    } else {
      pass.DrawIndexed(data.indexCount, data.instanceCount);
    }
  };
  if (pipeline.prepass) {
    gfx::bind_pipeline(pipeline.prepass, pass);
    draw();
    if (const auto hook = afterPrepassHook.load()) hook();
  }
  gfx::bind_pipeline(pipeline.main, pass);
  draw();
}

} // namespace aurora::gx
