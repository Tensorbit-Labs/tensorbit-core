#pragma once

/// @file ehap.hpp
/// @brief Efficient Hessian-Aware Pruning (EHAP) — full implementation.
/// @ingroup tensorbit-core

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <type_traits>
#include <vector>

#include "tensorbit/core/kernels.hpp"
#include "tensorbit/core/tensor.hpp"

namespace tensorbit::core {

enum class EHAPError : uint8_t {
    kOk                = 0,
    kShapeMismatch     = 1,
    kZeroSizeTensor    = 2,
    kInvalidConfig     = 3,
    kCudaNotAvailable  = 4,
};

struct EHAPConfig {
    float       damping = 0.01f;
    bool        use_diagonal_fisher = true;
    float       sparsity_ratio = 0.5f;
    std::size_t accumulation_steps = 100;
};

template<FloatingPoint F>
class EHAPPruner {
public:
    explicit EHAPPruner(EHAPConfig config) : config_(config) {}

    ~EHAPPruner() = default;

    EHAPPruner(const EHAPPruner&)                = delete;
    EHAPPruner& operator=(const EHAPPruner&)     = delete;
    EHAPPruner(EHAPPruner&&) noexcept            = default;
    EHAPPruner& operator=(EHAPPruner&&) noexcept = default;

    // -----------------------------------------------------------------------
    // accumulate_fisher
    // -----------------------------------------------------------------------
    auto accumulate_fisher(const TensorDense<F>& gradients, F alpha)
        -> Result<void, EHAPError>
    {
        if (gradients.empty())
            return unexpected(EHAPError::kZeroSizeTensor);

        if (fisher_diag_.empty()) {
            auto shp = std::span(gradients.shape().data(), gradients.rank());
            fisher_diag_ = TensorDense<F>(shp, gradients.device());
        }

        if (fisher_diag_.size() != gradients.size())
            return unexpected(EHAPError::kShapeMismatch);

        if (!config_.use_diagonal_fisher)
            return {};

        if constexpr (std::is_same_v<F, float>) {
            if (gradients.device() == DeviceLocation::kDevice) {
                kernels::launch_fisher_accumulate(
                    fisher_diag_.data(), gradients.data(),
                    gradients.size(), alpha, nullptr);
                CUDA_SYNC_CHECK();
                return {};
            }
        }

        // --- CPU path ---
        F* __restrict__       fo = fisher_diag_.data();
        const F* __restrict__ gi = gradients.data();
        std::size_t           N  = gradients.size();
        for (std::size_t i = 0; i < N; ++i)
            fo[i] += alpha * gi[i] * gi[i];
        return {};
    }

    // -----------------------------------------------------------------------
    // compute_importance
    // -----------------------------------------------------------------------
    auto compute_importance(const TensorDense<F>& weights,
                            TensorDense<F>&       out_importance)
        -> Result<void, EHAPError>
    {
        if (weights.empty() || out_importance.empty())
            return unexpected(EHAPError::kZeroSizeTensor);
        if (weights.size() != out_importance.size())
            return unexpected(EHAPError::kShapeMismatch);

        std::size_t N = weights.size();
        F damp = static_cast<F>(config_.damping);

        if constexpr (std::is_same_v<F, float>) {
            if (weights.device() == DeviceLocation::kDevice) {
                const float* fp = nullptr;
                if (config_.use_diagonal_fisher && !fisher_diag_.empty())
                    fp = fisher_diag_.data();
                kernels::launch_ehap_importance(
                    weights.data(), fp, out_importance.data(),
                    N, config_.damping, nullptr);
                CUDA_SYNC_CHECK();
                return {};
            }
        }

        // --- CPU path ---
        const F* __restrict__ wi = weights.data();
        F* __restrict__       so = out_importance.data();
        bool have_f = config_.use_diagonal_fisher && !fisher_diag_.empty();

        if (have_f) {
            const F* __restrict__ fi = fisher_diag_.data();
            for (std::size_t i = 0; i < N; ++i) {
                F w = wi[i];
                so[i] = w * w * (fi[i] + damp);
            }
        } else {
            for (std::size_t i = 0; i < N; ++i) {
                F w = wi[i];
                so[i] = w * w;
            }
        }
        return {};
    }

    // -----------------------------------------------------------------------
    // select_pruning_mask
    // -----------------------------------------------------------------------
    auto select_pruning_mask(const TensorDense<F>& importance,
                             TensorDense<uint8_t>& out_mask)
        -> Result<std::size_t, EHAPError>
    {
        if (importance.empty() || out_mask.empty())
            return unexpected(EHAPError::kZeroSizeTensor);
        if (importance.size() != out_mask.size())
            return unexpected(EHAPError::kShapeMismatch);
        if (importance.device() != DeviceLocation::kHost)
            return unexpected(EHAPError::kCudaNotAvailable);

        float sp = config_.sparsity_ratio;
        if (sp <= 0.0f || sp >= 1.0f)
            return unexpected(EHAPError::kInvalidConfig);

        std::size_t N = importance.size();
        std::size_t keep = static_cast<std::size_t>(sp * static_cast<float>(N));
        if (keep == 0) keep = 1;
        if (keep >= N) {
            std::fill_n(out_mask.data(), N, static_cast<uint8_t>(1));
            return std::size_t{0};
        }

        std::vector<F> copy(importance.data(), importance.data() + N);
        std::size_t tidx = N - keep;
        std::nth_element(copy.begin(),
                         copy.begin() + static_cast<std::ptrdiff_t>(tidx),
                         copy.end());
        F thr = copy[tidx];

        std::size_t pruned = 0;
        uint8_t* __restrict__ mo = out_mask.data();
        const F* __restrict__  im = importance.data();
        for (std::size_t i = 0; i < N; ++i) {
            bool k = (im[i] >= thr);
            mo[i] = k ? static_cast<uint8_t>(1) : static_cast<uint8_t>(0);
            if (!k) ++pruned;
        }
        return pruned;
    }

    [[nodiscard]] const EHAPConfig& config() const noexcept { return config_; }
    [[nodiscard]] const TensorDense<F>& fisher_diagonal() const noexcept { return fisher_diag_; }
    void reset() { fisher_diag_ = TensorDense<F>{}; }

private:
    EHAPConfig    config_;
    TensorDense<F> fisher_diag_;
};

}  // namespace tensorbit::core
