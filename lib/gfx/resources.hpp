#pragma once

#include "types.hpp"

namespace aurora::gfx {
inline constexpr bool UseTextureBuffer = true;
inline constexpr uint64_t UniformBufferSize = 25165824; // 24 MiB
inline constexpr uint64_t VertexBufferSize = 5242880;   // 5 MiB
inline constexpr uint64_t IndexBufferSize = 2097152;    // 2 MiB
inline constexpr uint64_t UberStateBufferSize = 1048576; // 1 MiB
inline constexpr uint64_t StorageBufferSize = 8388608;  // 8 MiB
inline constexpr uint64_t TextureUploadSize = 25165824; // 24 MiB

namespace detail {
inline bool supports_hybrid_resources(const wgpu::Limits& limits,
                                     const wgpu::CompatibilityModeLimits& compatibility) noexcept {
  return limits.maxStorageBuffersPerShaderStage >= 3 &&
         limits.maxStorageBufferBindingSize >= UberStateBufferSize &&
         limits.maxInterStageShaderVariables >= 11 &&
         compatibility.maxStorageBuffersInVertexStage >= 3 &&
         compatibility.maxStorageBuffersInVertexStage != wgpu::kLimitU32Undefined &&
         compatibility.maxStorageBuffersInFragmentStage >= 2 &&
         compatibility.maxStorageBuffersInFragmentStage != wgpu::kLimitU32Undefined;
}

struct Resources {
  wgpu::Buffer vertexBuffer;
  wgpu::Buffer uniformBuffer;
  wgpu::Buffer indexBuffer;
  wgpu::Buffer storageBuffer;
  wgpu::Buffer uberStateBuffer;
  uint64_t deviceGeneration = 0;
  bool hybridResources = false;
  AuroraShaderMode shaderMode = AURORA_SHADER_SPECIALIZED;
  wgpu::BindGroupLayout staticBindGroupLayout;
  wgpu::BindGroup staticBindGroup;
  wgpu::BindGroupLayout uniformBindGroupLayout;
  wgpu::BindGroup uniformBindGroup;
  wgpu::Limits limits;
  AuroraStats stats{};
};

Resources& resources() noexcept;
} // namespace detail
} // namespace aurora::gfx
