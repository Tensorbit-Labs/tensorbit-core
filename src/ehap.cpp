/// @file ehap.cpp
/// @brief Efficient Hessian-Aware Pruning (EHAP) — full implementation.
/// @ingroup tensorbit-core

#include "tensorbit/core/ehap.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include "tensorbit/core/common.hpp"
#include "tensorbit/core/kernels.hpp"
#include "tensorbit/core/tensor.hpp"

namespace tensorbit::core {

// ===========================================================================
// EHAPPruner<float>
// ===========================================================================

template<>
EHAPPruner<float>::EHAPPruner(EHAPConfig config)
    : config_(config) {
    TENSORBIT_LOG_DEBUG("EHAPPruner<float> constructed (damping={}, sparsity_ratio={}, "
                        "fisher={}, accumulation_steps={})",
                        config_.damping,
                        config_.sparsity_ratio,
                        config_.use_diagonal_fisher ? "on" : "off",
                        config_.accumulation_steps);
}

// ---------------------------------------------------------------------------
// accumulate_fisher
// ---------------------------------------------------------------------------

template<>
auto EHAPPruner<float>::accumulate_fisher(const TensorDense<float>& gradients,
                                          float alpha)
    -> std::expected<void, EHAPError> {
    if (gradients.empty()) {
        return std::unexpected(EHAPError::kZeroSizeTensor);
    }

    if (fisher_diag_.empty()) {
        auto shape = std::span(gradients.shape().data(), gradients.rank());
        fisher_diag_ = TensorDense<float>(shape, gradients.device());
        TENSORBIT_LOG_DEBUG("EHAPPruner<float>::accumulate_fisher — initialized "
                            "Fisher buffer ({} elements, device={})",
                            fisher_diag_.size(),
                            static_cast<int>(fisher_diag_.device()));
    }

    if (fisher_diag_.size() != gradients.size()) {
        TENSORBIT_LOG_ERROR("accumulate_fisher: shape mismatch (fisher={}, gradients={})",
                            fisher_diag_.size(), gradients.size());
        return std::unexpected(EHAPError::kShapeMismatch);
    }

    if (!config_.use_diagonal_fisher) {
        return {};
    }

    if (gradients.device() == DeviceLocation::kDevice) {
        // --- GPU Path ---
        kernels::launch_fisher_accumulate(
            fisher_diag_.data(), gradients.data(),
            gradients.size(), alpha, nullptr);
        CUDA_SYNC_CHECK();
    } else {
        // --- CPU Path ---
        float* __restrict__       f_out = fisher_diag_.data();
        const float* __restrict__ g_in  = gradients.data();
        std::size_t               N     = gradients.size();

        for (std::size_t i = 0; i < N; ++i) {
            f_out[i] += alpha * g_in[i] * g_in[i];
        }
    }

    TENSORBIT_LOG_TRACE("accumulate_fisher: alpha={}, N={}", alpha, gradients.size());

    return {};
}

// ---------------------------------------------------------------------------
// compute_importance
// ---------------------------------------------------------------------------

template<>
auto EHAPPruner<float>::compute_importance(const TensorDense<float>& weights,
                                           TensorDense<float>&       out_importance)
    -> std::expected<void, EHAPError> {
    if (weights.empty() || out_importance.empty()) {
        return std::unexpected(EHAPError::kZeroSizeTensor);
    }

    if (weights.size() != out_importance.size()) {
        TENSORBIT_LOG_ERROR("compute_importance: shape mismatch (weights={}, importance={})",
                            weights.size(), out_importance.size());
        return std::unexpected(EHAPError::kShapeMismatch);
    }

    std::size_t N = weights.size();
    float damping = config_.damping;

    if (weights.device() == DeviceLocation::kDevice) {
        // --- GPU Path ---
        const float* fisher_ptr = nullptr;
        if (config_.use_diagonal_fisher && !fisher_diag_.empty()) {
            fisher_ptr = fisher_diag_.data();
        }
        kernels::launch_ehap_importance(
            weights.data(), fisher_ptr,
            out_importance.data(), N, damping, nullptr);
        CUDA_SYNC_CHECK();
    } else {
        // --- CPU Path ---
        const float* __restrict__ w_in     = weights.data();
        float* __restrict__       s_out    = out_importance.data();
        const float* __restrict__ f_in     = nullptr;
        bool have_fisher = config_.use_diagonal_fisher && !fisher_diag_.empty();

        if (have_fisher) {
            f_in = fisher_diag_.data();
        }

        if (have_fisher) {
            // s_i = w_i^2 * (F_i + λ)
            for (std::size_t i = 0; i < N; ++i) {
                float w    = w_in[i];
                float fval = f_in[i] + damping;
                s_out[i]   = w * w * fval;
            }
        } else {
            // Magnitude fallback: s_i = w_i^2
            for (std::size_t i = 0; i < N; ++i) {
                float w = w_in[i];
                s_out[i] = w * w;
            }
        }
    }

    TENSORBIT_LOG_TRACE("compute_importance: N={}, damping={}", N, damping);

    return {};
}

// ---------------------------------------------------------------------------
// select_pruning_mask
// ---------------------------------------------------------------------------

template<>
auto EHAPPruner<float>::select_pruning_mask(const TensorDense<float>& importance,
                                            TensorDense<uint8_t>&    out_mask)
    -> std::expected<std::size_t, EHAPError> {
    if (importance.empty() || out_mask.empty()) {
        return std::unexpected(EHAPError::kZeroSizeTensor);
    }

    if (importance.size() != out_mask.size()) {
        return std::unexpected(EHAPError::kShapeMismatch);
    }

    if (importance.device() != DeviceLocation::kHost) {
        TENSORBIT_LOG_ERROR("select_pruning_mask requires host-resident importance tensor");
        return std::unexpected(EHAPError::kCudaNotAvailable);
    }

    float sparsity_ratio = config_.sparsity_ratio;
    if (sparsity_ratio <= 0.0f || sparsity_ratio >= 1.0f) {
        return std::unexpected(EHAPError::kInvalidConfig);
    }

    std::size_t N = importance.size();
    std::size_t num_to_keep = static_cast<std::size_t>(sparsity_ratio * static_cast<float>(N));
    if (num_to_keep == 0) num_to_keep = 1;
    if (num_to_keep >= N) {
        // Keep everything — no pruning needed.
        std::fill_n(out_mask.data(), N, static_cast<uint8_t>(1));
        return std::size_t{0};
    }

    // --- Threshold-based selection via std::nth_element ---
    // We need to find the (N - num_to_keep)-th smallest element and prune
    // everything below that threshold. A copy is required because nth_element
    // reorders the array.

    std::vector<float> importance_copy(importance.data(), importance.data() + N);
    std::size_t threshold_idx = N - num_to_keep;

    std::nth_element(importance_copy.begin(),
                     importance_copy.begin() + static_cast<std::ptrdiff_t>(threshold_idx),
                     importance_copy.end());

    float threshold = importance_copy[threshold_idx];

    // --- Emit mask: 1 = keep (≥ threshold), 0 = prune (< threshold) ---
    std::size_t pruned_count = 0;
    uint8_t* __restrict__ mask_out = out_mask.data();
    const float* __restrict__ imp_in = importance.data();

    for (std::size_t i = 0; i < N; ++i) {
        bool keep = (imp_in[i] >= threshold);
        mask_out[i] = keep ? static_cast<uint8_t>(1) : static_cast<uint8_t>(0);
        if (!keep) ++pruned_count;
    }

    TENSORBIT_LOG_DEBUG("select_pruning_mask: N={}, threshold={:.6e}, pruned={}, kept={}",
                        N, threshold, pruned_count, num_to_keep);

    return pruned_count;
}

template class EHAPPruner<float>;

// ===========================================================================
// EHAPPruner<double>
// ===========================================================================

template<>
EHAPPruner<double>::EHAPPruner(EHAPConfig config)
    : config_(config) {
    TENSORBIT_LOG_DEBUG("EHAPPruner<double> constructed (damping={}, sparsity_ratio={}, "
                        "fisher={}, accumulation_steps={})",
                        config_.damping,
                        config_.sparsity_ratio,
                        config_.use_diagonal_fisher ? "on" : "off",
                        config_.accumulation_steps);
}

// ---------------------------------------------------------------------------
// accumulate_fisher (double)
// ---------------------------------------------------------------------------

template<>
auto EHAPPruner<double>::accumulate_fisher(const TensorDense<double>& gradients,
                                           double alpha)
    -> std::expected<void, EHAPError> {
    if (gradients.empty()) {
        return std::unexpected(EHAPError::kZeroSizeTensor);
    }

    if (fisher_diag_.empty()) {
        auto shape = std::span(gradients.shape().data(), gradients.rank());
        fisher_diag_ = TensorDense<double>(shape, gradients.device());
        TENSORBIT_LOG_DEBUG("EHAPPruner<double>::accumulate_fisher — initialized "
                            "Fisher buffer ({} elements, device={})",
                            fisher_diag_.size(),
                            static_cast<int>(fisher_diag_.device()));
    }

    if (fisher_diag_.size() != gradients.size()) {
        return std::unexpected(EHAPError::kShapeMismatch);
    }

    if (!config_.use_diagonal_fisher) {
        return {};
    }

    // CPU path only for double precision (GPU kernels are float-only).
    // Future: add double-precision CUDA kernels for H100 FP64 Tensor Cores.

    double* __restrict__       f_out = fisher_diag_.data();
    const double* __restrict__ g_in  = gradients.data();
    std::size_t                N     = gradients.size();

    for (std::size_t i = 0; i < N; ++i) {
        f_out[i] += alpha * g_in[i] * g_in[i];
    }

    return {};
}

// ---------------------------------------------------------------------------
// compute_importance (double)
// ---------------------------------------------------------------------------

template<>
auto EHAPPruner<double>::compute_importance(const TensorDense<double>& weights,
                                            TensorDense<double>&       out_importance)
    -> std::expected<void, EHAPError> {
    if (weights.empty() || out_importance.empty()) {
        return std::unexpected(EHAPError::kZeroSizeTensor);
    }

    if (weights.size() != out_importance.size()) {
        return std::unexpected(EHAPError::kShapeMismatch);
    }

    std::size_t N       = weights.size();
    double      damping = static_cast<double>(config_.damping);

    const double* __restrict__ w_in  = weights.data();
    double* __restrict__       s_out = out_importance.data();

    bool have_fisher = config_.use_diagonal_fisher && !fisher_diag_.empty();

    if (have_fisher) {
        const double* __restrict__ f_in = fisher_diag_.data();
        for (std::size_t i = 0; i < N; ++i) {
            double w    = w_in[i];
            double fval = f_in[i] + damping;
            s_out[i]    = w * w * fval;
        }
    } else {
        for (std::size_t i = 0; i < N; ++i) {
            double w = w_in[i];
            s_out[i] = w * w;
        }
    }

    TENSORBIT_LOG_TRACE("EHAPPruner<double>::compute_importance: N={}, damping={}",
                        N, damping);

    return {};
}

// ---------------------------------------------------------------------------
// select_pruning_mask (double)
// ---------------------------------------------------------------------------

template<>
auto EHAPPruner<double>::select_pruning_mask(const TensorDense<double>& importance,
                                             TensorDense<uint8_t>&     out_mask)
    -> std::expected<std::size_t, EHAPError> {
    if (importance.empty() || out_mask.empty()) {
        return std::unexpected(EHAPError::kZeroSizeTensor);
    }

    if (importance.size() != out_mask.size()) {
        return std::unexpected(EHAPError::kShapeMismatch);
    }

    float sparsity_ratio = config_.sparsity_ratio;
    if (sparsity_ratio <= 0.0f || sparsity_ratio >= 1.0f) {
        return std::unexpected(EHAPError::kInvalidConfig);
    }

    std::size_t N = importance.size();
    std::size_t num_to_keep = static_cast<std::size_t>(sparsity_ratio * static_cast<float>(N));
    if (num_to_keep == 0) num_to_keep = 1;
    if (num_to_keep >= N) {
        std::fill_n(out_mask.data(), N, static_cast<uint8_t>(1));
        return std::size_t{0};
    }

    std::vector<double> importance_copy(importance.data(), importance.data() + N);
    std::size_t threshold_idx = N - num_to_keep;

    std::nth_element(importance_copy.begin(),
                     importance_copy.begin() + static_cast<std::ptrdiff_t>(threshold_idx),
                     importance_copy.end());

    double threshold = importance_copy[threshold_idx];

    std::size_t pruned_count = 0;
    uint8_t* __restrict__ mask_out = out_mask.data();
    const double* __restrict__ imp_in = importance.data();

    for (std::size_t i = 0; i < N; ++i) {
        bool keep = (imp_in[i] >= threshold);
        mask_out[i] = keep ? static_cast<uint8_t>(1) : static_cast<uint8_t>(0);
        if (!keep) ++pruned_count;
    }

    TENSORBIT_LOG_DEBUG("EHAPPruner<double>::select_pruning_mask: N={}, threshold={:.6e}, "
                        "pruned={}, kept={}",
                        N, threshold, pruned_count, num_to_keep);

    return pruned_count;
}

template class EHAPPruner<double>;

}  // namespace tensorbit::core
