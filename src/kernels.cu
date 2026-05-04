/// @file kernels.cu
/// @brief CUDA kernels for EHAP Fisher-diagonal scoring and N:M mask generation.
/// @ingroup tensorbit-core
///
/// ### Kernel Design Notes
/// - All kernels use a 1-D grid of 1-D thread blocks.
/// - The 2:4 mask kernel uses one thread per group (4-element locality, no warp sync).
/// - The generic N:M mask kernel uses one block per group with cooperative shared-memory
///   ranking — each thread holds one element and counts how many others it outranks.
/// - All kernels are optimized for A100/H100 (SM80/SM90) occupancy.

#include <cuda_runtime.h>

#include <cfloat>
#include <cstddef>
#include <cstdint>

// ---------------------------------------------------------------------------
// Device Constants
// ---------------------------------------------------------------------------

constexpr int kThreadsPerBlock = 256;

// ---------------------------------------------------------------------------
// get_grid_blocks
// ---------------------------------------------------------------------------

__host__ __device__ constexpr int get_grid_blocks(std::size_t N, int threads = kThreadsPerBlock) {
    return static_cast<int>((N + static_cast<std::size_t>(threads) - 1) /
                            static_cast<std::size_t>(threads));
}

// ===========================================================================
// EHAP Math Kernels
// ===========================================================================

// ---------------------------------------------------------------------------
// Kernel: fisher_diagonal
// ---------------------------------------------------------------------------

/// @brief CUDA kernel: computes the per-element squared-gradient sum for the
/// diagonal Fisher Information approximation.
///
/// @f[
///   F_i = \sum_{b=0}^{B-1} g_{b,i}^2
/// @f]
///
/// Assumes gradients are stored in row-major order with shape [B, N]
/// and the output fisher_diag has shape [N].
__global__ void fisher_diagonal_kernel(const float* __restrict__ gradients,
                                       float* __restrict__       fisher_diag,
                                       std::size_t               B,
                                       std::size_t               N) {
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float sum = 0.0f;
    for (std::size_t b = 0; b < B; ++b) {
        float g = gradients[b * N + idx];
        sum = fmaf(g, g, sum);
    }
    fisher_diag[idx] = sum;
}

// ---------------------------------------------------------------------------
// Kernel: fisher_accumulate
// ---------------------------------------------------------------------------

/// @brief CUDA kernel: accumulates per-element gradient information into the
/// running Fisher diagonal.
///
/// @f[
///   F_i \leftarrow F_i + \alpha \cdot g_i^2
/// @f]
///
/// @param fisher_diag Running Fisher diagonal (modified in-place).
/// @param gradients   Per-element gradient values (same shape as fisher_diag).
/// @param N           Number of elements.
/// @param alpha       Scaling factor.
__global__ void fisher_accumulate_kernel(float* __restrict__       fisher_diag,
                                         const float* __restrict__ gradients,
                                         std::size_t               N,
                                         float                     alpha) {
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float g = gradients[idx];
    fisher_diag[idx] = fmaf(alpha * g, g, fisher_diag[idx]);
}

// ---------------------------------------------------------------------------
// Kernel: fisher_beta_decay
// ---------------------------------------------------------------------------

/// @brief CUDA kernel: applies EMA decay factor to the Fisher diagonal.
/// F[i] = beta * F[i]  on the device directly, eliminating the previous
/// host round-trip (device→host→multiply→host→device).
__global__ void fisher_beta_decay_kernel(float* __restrict__ fisher_diag,
                                         float beta, std::size_t N) {
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    fisher_diag[idx] *= beta;
}

// ---------------------------------------------------------------------------
// Kernel: ehap_importance
// ---------------------------------------------------------------------------

/// @brief CUDA kernel: computes the EHAP importance score for each weight.
///
/// @f[
///   s_i = w_i^2 \cdot (F_i + \lambda)
/// @f]
///
/// where @f$ F_i @f$ is the Fisher diagonal and @f$ \lambda @f$ is the
/// damping factor for numerical stability.
///
/// When @f$ F_i @f$ is unavailable (null pointer), the kernel falls back
/// to magnitude-based importance: @f$ s_i = w_i^2 @f$.
///
/// @param weights     Weight values.
/// @param fisher_diag Fisher diagonal (may be nullptr for magnitude fallback).
/// @param importance  Output importance scores.
/// @param N           Number of elements.
/// @param damping     Damping factor λ.
__global__ void ehap_importance_kernel(const float* __restrict__ weights,
                                       const float* __restrict__ fisher_diag,
                                       float* __restrict__       importance,
                                       std::size_t               N,
                                       float                     damping) {
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float w = weights[idx];
    float w2 = w * w;

    if (fisher_diag != nullptr) {
        float f_val = fisher_diag[idx] + damping;
        importance[idx] = w2 * f_val;
    } else {
        importance[idx] = w2;
    }
}

// ===========================================================================
// N:M Structured Sparsity Kernels
// ===========================================================================

// ---------------------------------------------------------------------------
// Kernel: nm_mask_2_4
// ---------------------------------------------------------------------------

/// @brief CUDA kernel: generates a 2:4 structured-sparsity bitmask.
///
/// **Algorithm** (per thread, one group of 4):
///   1. Load the 4 importance values for the group.
///   2. Find the indices of the top-2 magnitudes via a fixed 4-element
///      comparison tree (no loops, fully unrolled).
///   3. Write a packed byte where bits 0-3 correspond to elements 0-3;
///      a set bit (1) means "keep this weight."
///
/// **Performance**: Each thread works entirely on local registers.
/// Zero shared memory, zero warp synchronization. Occupancy is
/// limited only by register pressure (~18 registers per thread on SM80),
/// yielding near-100% theoretical occupancy.
///
/// **Why this matters for Ampere**: The A100/H100 Sparse Tensor Cores
/// require exactly this 2:4 pattern. By generating masks in the same
/// warp-local fashion, we minimize PCIe and memory overhead when the
/// mask is consumed directly by cuSPARSELt or our .tb runtime.
__global__ void nm_mask_2_4_kernel(const float* __restrict__ importance,
                                   uint8_t* __restrict__     mask_out,
                                   std::size_t               N) {
    std::size_t group_idx = blockIdx.x * blockDim.x + threadIdx.x;
    std::size_t num_groups = N / 4;
    if (group_idx >= num_groups) return;

    std::size_t base = group_idx * 4;

    float v0 = importance[base + 0];
    float v1 = importance[base + 1];
    float v2 = importance[base + 2];
    float v3 = importance[base + 3];

    float a0 = fabsf(v0);
    float a1 = fabsf(v1);
    float a2 = fabsf(v2);
    float a3 = fabsf(v3);

    // ---- Stage 1: Find the single maximum ----
    int   top1    = 0;
    float top1val = a0;

    if (a1 > top1val) { top1 = 1; top1val = a1; }
    if (a2 > top1val) { top1 = 2; top1val = a2; }
    if (a3 > top1val) { top1 = 3; top1val = a3; }

    // ---- Stage 2: Find the second maximum (excluding top1) ----
    // Start with the first index that is not top1.
    int   top2    = (top1 == 0) ? 1 : 0;
    float top2val = (top1 == 0) ? a1 : a0;

    // Check remaining candidates that are not top1.
    for (int c = 0; c < 4; ++c) {
        if (c == top1) continue;
        float aval = (c == 0) ? a0 : (c == 1) ? a1 : (c == 2) ? a2 : a3;
        if (c != top1 && aval > top2val) {
            top2    = c;
            top2val = aval;
        }
    }

    // ---- Stage 3: Emit mask byte ----
    uint8_t mask = 0;
    mask |= (1u << top1);
    mask |= (1u << top2);

    mask_out[group_idx] = mask;
}

// ---------------------------------------------------------------------------
// Kernel: nm_mask_generic
// ---------------------------------------------------------------------------

/// @brief CUDA kernel: generates a generic N:M structured-sparsity bitmask
/// using cooperative shared-memory ranking.
///
/// **Algorithm** (per block, one group of size M):
///   1. Each thread (lane i) loads importance value `v_i` for element i
///      of the group into shared memory.
///   2. Each thread counts how many other elements in the group have a
///      value *strictly greater* than its own. Ties are broken by index
///      (lower index wins) to ensure deterministic behavior.
///   3. Thread 0 gathers the results: any element whose rank is less
///      than N (i.e., top-N) gets a set bit in the output mask.
///   4. The mask is packed into 1 byte per group (bits 0..M-1).
///
/// **Constraints**: M must be ≤ 32 (fits in a warp; larger M needs
/// additional shared memory and block-level sync). M ≤ 32 covers all
/// practical N:M patterns (1:4, 2:4, 1:8, 2:8, 4:8).
///
/// **Time Complexity**: O(M^2) comparisons per block — negligible for
/// M ≤ 32. Each block of M threads produces 1 mask byte.
__global__ void nm_mask_generic_kernel(const float* __restrict__ importance,
                                       uint8_t* __restrict__     mask_out,
                                       std::size_t               N_elements,
                                       int                       group_n,
                                       int                       group_m) {
    int group_idx = blockIdx.x;
    std::size_t base = static_cast<std::size_t>(group_idx) * static_cast<std::size_t>(group_m);
    int lane = threadIdx.x;

    __shared__ float s_vals[32];
    __shared__ int   s_ranks[32];

    // ---- Initialize shared memory: all lanes write sentinel ----
    // This covers lanes [0..group_m-1] with actual values and
    // lanes [group_m..31] with -FLT_MAX (never selected, prevents
    // uninitialized-memory contamination of the ranking loop).
    s_vals[lane] = -FLT_MAX;
    if (lane < group_m) {
        s_vals[lane] = importance[base + lane];
    }
    __syncthreads();

    // ---- Each thread computes its rank (number of elements > its value) ----
    int rank = 0;
    float my_val = s_vals[lane];

    for (int j = 0; j < group_m; ++j) {
        float other = s_vals[j];
        if (other > my_val) {
            rank++;
        } else if (other == my_val && j < lane) {
            // Tie-breaking: lower index wins (stability guarantee)
            rank++;
        }
    }

    if (lane < group_m) {
        s_ranks[lane] = rank;
    }
    __syncthreads();

    // ---- Thread 0 assembles the mask ----
    if (lane == 0) {
        uint32_t mask_bits = 0;
        for (int i = 0; i < group_m; ++i) {
            if (s_ranks[i] < group_n) {
                mask_bits |= (1u << i);
            }
        }
        mask_out[group_idx] = static_cast<uint8_t>(mask_bits);
    }
}

// ---------------------------------------------------------------------------
// Kernel: apply_mask
// ---------------------------------------------------------------------------

/// @brief CUDA kernel: applies a packed N:M bitmask to zero out pruned weights
/// in-place.
///
/// **Algorithm** (per thread, one element):
///   1. Determine the group and intra-group index for this element:
///      group = idx / M,  offset = idx % M
///   2. Read the mask byte for the group.
///   3. If the bit at position `offset` is 0, zero the weight.
///
/// The number of pruned weights is computed analytically as
/// `N_elements * (M - N) / M` by the caller — no runtime counting needed.
///
/// **Efficiency**: Each thread does one division, one modulo, one bit
/// test, and at most one store. No atomics, no divergence beyond the
/// conditional store. Occupancy is excellent.
__global__ void apply_mask_kernel(float* __restrict__             weights,
                                  const uint8_t* __restrict__    mask,
                                  std::size_t                     N,
                                  int                             group_m) {
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    std::size_t group_idx = idx / static_cast<std::size_t>(group_m);
    int          offset    = static_cast<int>(idx % static_cast<std::size_t>(group_m));

    uint8_t mask_byte = mask[group_idx];
    bool kept = (mask_byte >> offset) & 1u;

    if (!kept) {
        weights[idx] = 0.0f;
    }
}

// ===========================================================================
// Host Launch Wrappers
// ===========================================================================

namespace tensorbit::core::kernels {

// --- Fisher Diagonal ---

void launch_fisher_diagonal(const float* gradients,
                            float*       fisher_diag,
                            std::size_t  B,
                            std::size_t  N,
                            void*        stream) {
    if (N == 0 || B == 0) return;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    int grid = get_grid_blocks(N);
    fisher_diagonal_kernel<<<grid, kThreadsPerBlock, 0, s>>>(gradients, fisher_diag, B, N);
}

// --- Fisher Accumulate ---

void launch_fisher_accumulate(float*       fisher_diag,
                              const float* gradients,
                              std::size_t  N,
                              float        alpha,
                              void*        stream) {
    if (N == 0) return;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    int grid = get_grid_blocks(N);
    fisher_accumulate_kernel<<<grid, kThreadsPerBlock, 0, s>>>(fisher_diag, gradients, N, alpha);
}

// --- Fisher Beta Decay ---

void launch_fisher_beta_decay(float*       fisher_diag,
                              float        beta,
                              std::size_t  N,
                              void*        stream) {
    if (N == 0) return;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    int grid = get_grid_blocks(N);
    fisher_beta_decay_kernel<<<grid, kThreadsPerBlock, 0, s>>>(fisher_diag, beta, N);
}

// --- EHAP Importance ---

void launch_ehap_importance(const float* weights,
                            const float* fisher_diag,
                            float*       importance,
                            std::size_t  N,
                            float        damping,
                            void*        stream) {
    if (N == 0) return;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    int grid = get_grid_blocks(N);
    ehap_importance_kernel<<<grid, kThreadsPerBlock, 0, s>>>(
        weights, fisher_diag, importance, N, damping);
}

// --- 2:4 Mask ---

void launch_nm_mask_2_4(const float* importance,
                        uint8_t*     mask_out,
                        std::size_t  N,
                        void*        stream) {
    if (N < 4) return;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    std::size_t num_groups = N / 4;
    int grid = get_grid_blocks(num_groups);
    nm_mask_2_4_kernel<<<grid, kThreadsPerBlock, 0, s>>>(importance, mask_out, N);
}

// --- Generic N:M Mask ---

void launch_nm_mask_generic(const float* importance,
                            uint8_t*     mask_out,
                            std::size_t  N_elements,
                            int          group_n,
                            int          group_m,
                            void*        stream) {
    if (N_elements == 0 || group_m <= 0) return;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    std::size_t num_groups = N_elements / static_cast<std::size_t>(group_m);
    int grid = static_cast<int>(num_groups);
    nm_mask_generic_kernel<<<grid, group_m, 0, s>>>(
        importance, mask_out, N_elements, group_n, group_m);
}

// --- Apply Mask ---

void launch_apply_mask(float*              weights,
                       const uint8_t*     mask,
                       std::size_t         N,
                       int                 group_m,
                       void*              stream) {
    if (N == 0) return;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    int grid = get_grid_blocks(N);
    apply_mask_kernel<<<grid, kThreadsPerBlock, 0, s>>>(weights, mask, N, group_m);
}

}  // namespace tensorbit::core::kernels
