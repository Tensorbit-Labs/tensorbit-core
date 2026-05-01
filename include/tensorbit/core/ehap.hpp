#pragma once

/// @file ehap.hpp
/// @brief Efficient Hessian-Aware Pruning (EHAP) — pruner class skeleton.
/// @ingroup tensorbit-core

#include <cstddef>
#include <expected>
#include <string_view>

#include "tensorbit/core/tensor.hpp"

namespace tensorbit::core {

/// @brief Error codes returned by EHAPPruner operations.
enum class EHAPError : uint8_t {
    kOk                = 0,
    kShapeMismatch     = 1,
    kZeroSizeTensor    = 2,
    kInvalidConfig     = 3,
    kCudaNotAvailable  = 4,
};

/// @brief Configuration for the EHAP pruner.
struct EHAPConfig {
    /// Damping term added to the Fisher diagonal for numerical stability.
    float damping = 0.01f;

    /// If true, uses the diagonal Fisher Information approximation (O(N) memory).
    /// If false, falls back to magnitude-based pruning.
    bool use_diagonal_fisher = true;

    /// Target sparsity ratio (fraction of weights to retain, 0.0–1.0).
    float sparsity_ratio = 0.5f;

    /// Number of gradient accumulation steps before recomputing importance.
    std::size_t accumulation_steps = 100;
};

// ---------------------------------------------------------------------------
// EHAPPruner — Efficient Hessian-Aware Pruning Engine
// ---------------------------------------------------------------------------

/// @class EHAPPruner
/// @brief Computes per-weight importance scores using a diagonal Fisher
/// Information approximation and selects weights for removal.
///
/// ## Mathematical Foundation
/// The EHAP objective approximates the loss landscape curvature around
/// weight @f$ w_i @f$ using the diagonal of the Fisher Information Matrix:
/// @f[
///   F_{ii} = \mathbb{E}_{x \sim D} \left[ \left( \frac{\partial \mathcal{L}}{\partial w_i} \right)^2 \right]
/// @f]
///
/// The sensitivity score for weight @f$ w_i @f$ is then:
/// @f[
///   s_i = w_i^2 \cdot (F_{ii} + \lambda)
/// @f]
/// where @f$ \lambda @f$ is a damping factor.
///
/// Weights with the lowest scores are pruned.
///
/// @tparam F Floating-point precision (float or double).
template<FloatingPoint F>
class EHAPPruner {
public:
    /// @brief Constructs an EHAP pruner with the given configuration.
    /// @param config Pruner configuration (damping, fisher mode, sparsity).
    explicit EHAPPruner(EHAPConfig config);

    ~EHAPPruner() = default;

    EHAPPruner(const EHAPPruner&)                = delete;
    EHAPPruner& operator=(const EHAPPruner&)     = delete;
    EHAPPruner(EHAPPruner&&) noexcept            = default;
    EHAPPruner& operator=(EHAPPruner&&) noexcept = default;

    /// @name Core Pipeline
    /// @{

    /// @brief Accumulates gradients into the Fisher diagonal buffer.
    ///
    /// Call this every backward pass. The Fisher diagonal is updated:
    /// @f$ F_{ii} \leftarrow F_{ii} + \alpha \cdot g_i^2 @f$
    ///
    /// @param gradients Gradient tensor matching the weight shape.
    /// @param alpha     Scaling factor (typically 1.0 / batch_size).
    /// @return std::expected with success or EHAPError.
    auto accumulate_fisher(const TensorDense<F>& gradients, F alpha)
        -> std::expected<void, EHAPError>;

    /// @brief Computes importance scores from the accumulated Fisher diagonal
    /// and current weight magnitudes.
    ///
    /// @param weights Current weight tensor.
    /// @param out_importance Output tensor (same shape as weights) receiving importance scores.
    /// @return std::expected with success or EHAPError.
    auto compute_importance(const TensorDense<F>& weights,
                            TensorDense<F>&       out_importance)
        -> std::expected<void, EHAPError>;

    /// @brief Selects weights to prune based on importance scores and the target sparsity.
    ///
    /// @param importance Importance score tensor.
    /// @param out_mask   Output binary mask tensor (1 = keep, 0 = prune).
    /// @return std::expected with the number of pruned weights on success, or EHAPError.
    auto select_pruning_mask(const TensorDense<F>& importance,
                             TensorDense<uint8_t>& out_mask)
        -> std::expected<std::size_t, EHAPError>;
    /// @}

    /// @name Accessors
    /// @{

    /// @brief Returns the current configuration (read-only).
    [[nodiscard]] const EHAPConfig& config() const noexcept { return config_; }

    /// @brief Returns the accumulated Fisher diagonal buffer.
    /// Empty tensor if accumulation has not started.
    [[nodiscard]] const TensorDense<F>& fisher_diagonal() const noexcept {
        return fisher_diag_;
    }

    /// @brief Resets the internal Fisher accumulator.
    void reset() { fisher_diag_ = TensorDense<F>{}; }
    /// @}

private:
    EHAPConfig    config_;
    TensorDense<F> fisher_diag_;  ///< Accumulated diagonal of Fisher Information Matrix.
};

// =========================================================================
// Explicit Instantiation Declarations
// =========================================================================

extern template class EHAPPruner<float>;
extern template class EHAPPruner<double>;

}  // namespace tensorbit::core
