#pragma once

/// @file coring.hpp
/// @brief CORING N:M Structured Sparsity pruner — class skeleton.
/// @ingroup tensorbit-core

#include <cstddef>
#include <cstdint>
#include <expected>
#include <string_view>

#include "tensorbit/core/tensor.hpp"

namespace tensorbit::core {

/// @brief Error codes returned by CORINGPruner operations.
enum class CORINGError : uint8_t {
    kOk              = 0,
    kShapeMismatch   = 1,
    kZeroSizeTensor  = 2,
    kInvalidNMConfig = 3,
    kCudaNotAvailable = 4,
};

/// @brief Configuration for the CORING N:M pruner.
struct CORINGConfig {
    /// Number of elements to keep in each group (N in N:M).
    int N = 2;

    /// Group size (M in N:M). Must be a power of two and M > N.
    int M = 4;

    /// If true, use CUDA-accelerated mask generation.
    bool use_cuda = true;
};

// ---------------------------------------------------------------------------
// CORINGPruner — N:M Structured Sparsity Engine
// ---------------------------------------------------------------------------

/// @class CORINGPruner
/// @brief Enforces N:M structured sparsity patterns on weight tensors.
///
/// ## N:M Structured Sparsity
/// For a given N:M pattern (e.g., 2:4), the tensor is divided into contiguous
/// groups of M elements. Within each group, only the N elements with the
/// highest importance scores are retained; the remaining M - N are zeroed.
///
/// This produces hardware-friendly sparsity that maps directly to the NVIDIA
/// Ampere Sparse Tensor Core instruction set, delivering up to 2× throughput
/// on A100 and H100 GPUs.
///
/// ## Algorithm Outline
/// 1. Receive per-weight importance scores from the upstream EHAP pruner.
/// 2. Partition scores into groups of M consecutive elements.
/// 3. Within each group, select the top-N elements.
/// 4. Emit a binary mask (1 = keep, 0 = prune) — typically packed as bits.
/// 5. Apply the mask to zero out pruned weights.
///
/// @tparam F Floating-point precision (float or double).
template<FloatingPoint F>
class CORINGPruner {
public:
    /// @brief Constructs a CORING pruner with the given N:M configuration.
    /// @param config Pruner configuration (N, M, CUDA toggle).
    explicit CORINGPruner(CORINGConfig config);

    ~CORINGPruner() = default;

    CORINGPruner(const CORINGPruner&)                = delete;
    CORINGPruner& operator=(const CORINGPruner&)     = delete;
    CORINGPruner(CORINGPruner&&) noexcept            = default;
    CORINGPruner& operator=(CORINGPruner&&) noexcept = default;

    /// @name Core Pipeline
    /// @{

    /// @brief Generates an N:M structured-sparsity mask from importance scores.
    ///
    /// Groups elements into blocks of M, keeps the top-N highest scores in
    /// each block, and writes a binary mask.
    ///
    /// @param importance Per-weight importance scores.
    /// @param out_mask   Output mask tensor (uint8, shape == importance shape).
    ///                   On device if CUDA is enabled.
    /// @return std::expected with success or CORINGError.
    auto generate_nm_mask(const TensorDense<F>& importance,
                          TensorDense<uint8_t>& out_mask)
        -> std::expected<void, CORINGError>;

    /// @brief Applies the N:M mask to zero out pruned weights.
    ///
    /// @param weights Weight tensor to prune (modified in-place).
    /// @param mask    Binary mask tensor (1 = keep).
    /// @return std::expected with the number of weights zeroed, or CORINGError.
    auto apply_mask(TensorDense<F>&            weights,
                    const TensorDense<uint8_t>& mask)
        -> std::expected<std::size_t, CORINGError>;

    /// @brief Full N:M pruning pass: generates mask from importance, then applies it.
    ///
    /// This convenience method chains `generate_nm_mask()` and `apply_mask()`.
    ///
    /// @param importance Importance scores.
    /// @param weights    Weight tensor to prune (modified in-place).
    /// @return std::expected with the number of weights pruned, or CORINGError.
    auto prune(const TensorDense<F>& importance,
               TensorDense<F>&       weights)
        -> std::expected<std::size_t, CORINGError>;
    /// @}

    /// @name Accessors
    /// @{

    /// @brief Returns the current configuration.
    [[nodiscard]] const CORINGConfig& config() const noexcept { return config_; }
    /// @}

private:
    CORINGConfig config_;

    /// @brief Validates that N, M, and tensor dimensions are compatible.
    auto validate_config(std::size_t num_elements) -> std::expected<void, CORINGError>;
};

// =========================================================================
// Explicit Instantiation Declarations
// =========================================================================

extern template class CORINGPruner<float>;
extern template class CORINGPruner<double>;

}  // namespace tensorbit::core
