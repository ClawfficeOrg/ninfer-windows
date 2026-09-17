#include "ops/linear_attention/gated_delta_net/chunked/launch.h"
#include "ops/linear_attention/gated_delta_net/chunked/output.cuh"

#include "core/device.h"

namespace ninfer::ops::detail::gated_delta_net::chunked {
namespace {

namespace kernel = output;

// The launch packs logical (chunk, head) jobs so that the grid covers at most one resident wave,
// which keeps the MULTI_JOB grid-stride loop from over-subscribing the device. The wave was the
// 170-SM RTX 5090 at four CTAs per SM (680). Resolved from the bound device instead, so the wave
// tracks the part in use; the kernel is a grid-stride loop and is correct at any grid size, so a
// wrong value only costs throughput.
//
// NOTE (RTX 5070 Ti / GB203, 70 SMs): four CTAs per SM was measured on the 5090 for this kernel's
// shared-memory footprint. Re-check the per-SM residency on the 5070 Ti before treating 4 as
// final; the fallback below preserves the 5090 behaviour if the attribute cannot be read.
constexpr std::int64_t kCtasPerSm          = 4;
constexpr std::int64_t kTargetCtasFallback = 680;

std::int64_t target_ctas() noexcept {
    const std::int32_t sm_count = current_device_multiprocessor_count();
    if (sm_count <= 0) { return kTargetCtasFallback; }
    return static_cast<std::int64_t>(sm_count) * kCtasPerSm;
}

template <bool MULTI_JOB>
cudaError_t launch_fixed(const chunk_output_config& cfg, dim3 grid, head_map qk_map, int chunks) {
    constexpr int smem_bytes = kernel::kernel_dims::SMEM_BYTES;

    cudaError_t err = cudaFuncSetAttribute(kernel::output_kernel<MULTI_JOB>,
                                           cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    if (err != cudaSuccess) { return err; }

    const dim3 block(kernel::THREADS, 1, 1);

    kernel::output_kernel<MULTI_JOB><<<grid, block, smem_bytes, cfg.stream>>>(
        cfg.q, cfg.k, cfg.v_new, cfg.g_cumsum, cfg.h_chunk, cfg.attn_out, qk_map, cfg.scale,
        chunks);
    return cudaGetLastError();
}

} // namespace

cudaError_t launch_output(const chunk_output_config& cfg) {
    stage_validator v{"launch_output", cfg.H_qk, cfg.H_v, cfg.L};
    NINFER_GATED_DELTA_NET_PROPAGATE(v.check_shape());
    NINFER_GATED_DELTA_NET_PROPAGATE(v.check_full_chunks());
    if (cfg.q == nullptr || cfg.k == nullptr || cfg.v_new == nullptr || cfg.g_cumsum == nullptr ||
        cfg.h_chunk == nullptr || cfg.attn_out == nullptr) {
        return cudaErrorInvalidValue;
    }

    const auto qk_map     = head_map::of((int)cfg.H_qk, (int)cfg.H_v);
    const std::int64_t NT = cfg.L / BT;

    // Keep at most one resident wave and distribute chunks evenly across it. Small grids retain
    // one logical job per CTA.
    const std::int64_t wave           = target_ctas();
    const std::int64_t logical_jobs   = NT * cfg.H_v;
    const std::int64_t jobs_per_block = (logical_jobs + wave - 1) / wave;
    const std::int64_t grid_chunks    = (NT + jobs_per_block - 1) / jobs_per_block;
    NINFER_GATED_DELTA_NET_PROPAGATE(v.check_grid(grid_chunks, cfg.H_v));

    const dim3 grid(static_cast<unsigned>(grid_chunks), static_cast<unsigned>(cfg.H_v), 1);
    if (jobs_per_block == 1) {
        return launch_fixed<false>(cfg, grid, qk_map, static_cast<int>(NT));
    }
    return launch_fixed<true>(cfg, grid, qk_map, static_cast<int>(NT));
}

} // namespace ninfer::ops::detail::gated_delta_net::chunked
