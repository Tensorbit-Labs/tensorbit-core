#pragma once

/// @file kernels.hpp
/// @brief CUDA kernel function declarations for EHAP scoring and N:M mask generation.
/// @ingroup tensorbit-core

#include <cstddef>
#include <cstdint>

namespace tensorbit::core::kernels {

// ---------------------------------------------------------------------------
// EHAP Fisher-Diagonal Kernel
// ---------------------------------------------------------------------------

/// @brief GPU kernel: computes the per-element squared-gradient sum for the
/// diagonal Fisher approximation: F_ii = sum over batch (grad_i)^2.
///
/// @param gradients   Device pointer to gradient tensor (shape: [B, N]).
/// @param fisher_diag Device pointer to output Fisher diagonal (shape: [N]).
/// @param B           Batch dimension.
/// @param N           Model dimension.
/// @param stream      CUDA stream handle (0 = default stream).
void launch_fisher_diagonal(const float* gradients,
                            float*       fisher_diag,
                            std::size_t  B,
                            std::size_t  N,
                            void*        stream);

/// @brief GPU kernel: accumulates per-element gradients into the Fisher diagonal.
/// Computes F[i] += alpha * g[i]^2 element-wise.
///
/// @param fisher_diag  Device pointer to Fisher diagonal (accumulated in-place).
/// @param gradients    Device pointer to gradient tensor (same shape as fisher_diag).
/// @param N            Number of elements.
/// @param alpha        Scaling factor (typically 1.0 / accumulation_steps).
/// @param stream       CUDA stream handle.
void launch_fisher_accumulate(float*       fisher_diag,
                              const float* gradients,
                              std::size_t  N,
                              float        alpha,
                              void*        stream);

/// @brief GPU kernel: computes EHAP importance scores.
/// s[i] = w[i]^2 * (F[i] + damping)
///
/// @param weights       Device pointer to model weights.
/// @param fisher_diag   Device pointer to Fisher diagonal.
/// @param importance    Device pointer to output importance scores.
/// @param N             Number of elements.
/// @param damping       Fisher damping factor λ.
/// @param stream        CUDA stream handle.
void launch_ehap_importance(const float* weights,
                            const float* fisher_diag,
                            float*       importance,
                            std::size_t  N,
                            float        damping,
                            void*        stream);

// ---------------------------------------------------------------------------
// N:M Structured Sparsity Mask Kernels
// ---------------------------------------------------------------------------

/// @brief GPU kernel: generates a 2:4 structured-sparsity bitmask from
/// importance scores. Keeps the top-2 elements in each contiguous group of 4.
///
/// Each CUDA thread processes one group independently (no warp-level sync needed).
///
/// @param importance Device pointer to importance scores (shape: [N]).
/// @param mask_out   Device pointer to output bitmask (1 byte per group, LSB = element 0).
/// @param N          Number of elements (must be divisible by 4).
/// @param stream     CUDA stream handle.
void launch_nm_mask_2_4(const float* importance,
                        uint8_t*     mask_out,
                        std::size_t  N,
                        void*        stream);

/// @brief GPU kernel: generates a generic N:M structured-sparsity bitmask.
///
/// Each thread block of size M processes one group. Uses shared memory for
/// cooperative ranking to find the top-N indices.
///
/// @param importance Device pointer to importance scores.
/// @param mask_out   Device pointer to output bitmask (packed per group).
/// @param N_elements Total number of elements.
/// @param group_n    Number of elements to keep per group (N).
/// @param group_m    Group size (M). Must be <= 32 for the cooperative sort.
/// @param stream     CUDA stream handle.
void launch_nm_mask_generic(const float* importance,
                            uint8_t*     mask_out,
                            std::size_t  N_elements,
                            int          group_n,
                            int          group_m,
                            void*        stream);

// ---------------------------------------------------------------------------
// Utility Kernels
// ---------------------------------------------------------------------------

/// @brief GPU kernel: applies a packed N:M bitmask to zero out pruned weights in-place.
///
/// @param weights Device pointer to weight tensor (shape: [N]).
/// @param mask    Device pointer to packed bitmask (1 byte per group).
/// @param N       Number of elements.
/// @param group_m Group size M (used to decode mask bits). Must match the kernel
///                that originally generated the mask.
/// @param stream  CUDA stream handle.
void launch_apply_mask(float*              weights,
                       const uint8_t*     mask,
                       std::size_t         N,
                       int                 group_m,
                       void*              stream);

}  // namespace tensorbit::core::kernels
