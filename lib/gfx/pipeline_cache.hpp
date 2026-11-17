#pragma once

#include "types.hpp"

namespace aurora::gfx::clear {
struct PipelineConfig;
} // namespace aurora::gfx::clear

namespace aurora::gx {
struct PipelineConfig;
} // namespace aurora::gx

namespace aurora::rmlui {
struct PipelineConfig;
} // namespace aurora::rmlui

namespace aurora::gfx {

enum class ShaderType : uint8_t {
  Clear = 0,
  GX = 1,
  Rml = 2,
};

struct CompiledPipeline {
  wgpu::RenderPipeline main;
  wgpu::RenderPipeline prepass;

  [[nodiscard]] uint32_t pipeline_count() const {
    return static_cast<uint32_t>(static_cast<bool>(main)) + static_cast<uint32_t>(static_cast<bool>(prepass));
  }
};

void initialize_pipeline_cache();
void shutdown_pipeline_cache();
void begin_pipeline_frame();
void end_pipeline_frame();
void rebuild_pipeline_cache();

PipelineRef find_pipeline(const gx::PipelineConfig& config, const RenderTargetLayout& layout, bool blocking = false);
PipelineRef find_pipeline(const clear::PipelineConfig& config, const RenderTargetLayout& layout, bool blocking = false);
PipelineRef find_pipeline(const rmlui::PipelineConfig& config);

enum class PipelineBuildState : uint8_t { Missing, Pending, Ready, Failed };
PipelineBuildState pipeline_build_state(PipelineRef ref) noexcept;
bool try_get_pipeline(PipelineRef ref, CompiledPipeline& pipeline) noexcept;
bool get_pipeline_ready(PipelineRef ref, CompiledPipeline& pipeline);
PipelineRef find_deferred_specialization(const gx::PipelineConfig&, const RenderTargetLayout&);
PipelineRef find_uber_pipeline(const gx::PipelineConfig&, const RenderTargetLayout&);
namespace testing {
using PipelineCreateHook = bool (*)(PipelineRef, bool generic);
// Install only while no jobs are running. The hook runs before the real creator.
void set_pipeline_create_hook(PipelineCreateHook) noexcept;
}
bool get_pipeline(PipelineRef ref, CompiledPipeline& pipeline);

// While alive, get_pipeline on this thread waits for pending pipelines instead of skipping the draw.
struct RequireReadyPipelinesScope {
  explicit RequireReadyPipelinesScope(bool enabled) noexcept;
  ~RequireReadyPipelinesScope() noexcept;
  RequireReadyPipelinesScope(const RequireReadyPipelinesScope&) = delete;
  RequireReadyPipelinesScope& operator=(const RequireReadyPipelinesScope&) = delete;

private:
  bool m_previous;
};

} // namespace aurora::gfx
