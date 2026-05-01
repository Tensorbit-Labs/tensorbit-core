#pragma once

/// @file ehap.hpp
/// @brief Efficient Hessian-Aware Pruning (EHAP) — research-grade implementation.
/// @ingroup tensorbit-core
///
/// Based on the Optimal Brain Damage (OBD) framework of LeCun et al. (1990)
/// and the Empirical Fisher Information diagonal approximation. Supports
/// multiple importance-score modes, EMA-based Fisher accumulation, iterative
/// pruning schedules, and weight compensation via OBS-style least-squares
/// correction. See docs/EHAP.md for complete mathematical exposition.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <type_traits>
#include <vector>

#include "tensorbit/core/kernels.hpp"
#include "tensorbit/core/tensor.hpp"

namespace tensorbit::core {

enum class EHAPError : uint8_t {
    kOk               = 0,
    kShapeMismatch    = 1,
    kZeroSizeTensor   = 2,
    kInvalidConfig    = 3,
    kCudaNotAvailable = 4,
};

/// @brief Importance-score formulation.
enum class ImportanceMode : uint8_t {
    /// LeCun et al. (1990): s_i = w_i^2 * (F_ii + lambda).
    kOBD = 0,
    /// Hassibi & Stork (1993) diagonal approx: s_i = w_i^2 / (F_ii + lambda).
    kOBS = 1,
    /// Normalised variant robust to scale differences across layers:
    /// s_i = w_i^2 * (F_ii + lambda) / (1 + w_i^2).
    kNormalized = 2,
};

/// @brief Pruning schedule.
enum class PruneStrategy : uint8_t {
    kOneShot   = 0,
    kIterative = 1,
};

/// @brief Post-pruning weight compensation.
enum class CompensationMode : uint8_t {
    kNone   = 0,
    /// Absorb pruned weight contributions into the layer bias (LeCun et al. 1990, Sec. 3).
    kBias   = 1,
    /// Redistribute pruned-weight magnitude to kept weights within each group
    /// proportionally to their Fisher values (OBS-inspired, diagonal approx).
    kRedist = 2,
};

/// @brief Configuration for the EHAP pruner.
struct EHAPConfig {
    /// Damping term λ added to the Fisher diagonal (LeCun et al. 1990, Eq. 8).
    float damping = 0.01f;

    /// If true, use Fisher diagonal; if false, fall back to magnitude pruning.
    bool use_diagonal_fisher = true;

    /// Target sparsity ratio — fraction of weights to **retain** (0 < ρ < 1).
    float sparsity_ratio = 0.5f;

    /// EMA decay factor β ∈ (0,1] for Fisher accumulation.
    /// F_ii ← β·F_ii + (1-β)·g_i^2.  β = 1.0 gives simple cumulative.
    float ema_decay = 0.99f;

    /// Number of gradient steps between importance recomputations.
    std::size_t accumulation_steps = 100;

    /// Which importance-score formulation to use.
    ImportanceMode importance_mode = ImportanceMode::kOBD;

    /// Pruning schedule.
    PruneStrategy prune_strategy = PruneStrategy::kOneShot;

    /// For iterative pruning: number of rounds.
    std::size_t prune_rounds = 5;

    /// For iterative pruning: fraction of CURRENT remaining weights pruned per round.
    float prune_fraction_per_round = 0.2f;

    /// Post-pruning weight compensation method.
    CompensationMode compensation_mode = CompensationMode::kNone;

    /// If true, normalise Fisher diagonal per-layer by its L2 norm
    /// before computing importance (Theis et al. 2018, Sec. 3.2).
    bool normalize_fisher = false;
};

// ===========================================================================
// EHAPPruner
// ===========================================================================

template<FloatingPoint F>
class EHAPPruner {
public:
    explicit EHAPPruner(EHAPConfig config) : config_(config), step_count_(0) {}

    ~EHAPPruner() = default;

    EHAPPruner(const EHAPPruner&)                = delete;
    EHAPPruner& operator=(const EHAPPruner&)     = delete;
    EHAPPruner(EHAPPruner&&) noexcept            = default;
    EHAPPruner& operator=(EHAPPruner&&) noexcept = default;

    // =======================================================================
    // accumulate_fisher — EMA-based Fisher diagonal update
    // =======================================================================
    /// @brief Accumulates gradients into the Fisher diagonal using an
    /// Exponential Moving Average (EMA).
    ///
    /// @f[
    ///   F_{ii} \leftarrow \beta \cdot F_{ii} + (1-\beta) \cdot g_i^2
    /// @f]
    ///
    /// @param gradients Per-weight gradients (shape == weight shape).
    /// @param alpha     Scaling factor (typically 1 / accumulation_steps).
    /// @return Success or EHAPError.
    auto accumulate_fisher(const TensorDense<F>& gradients, F alpha)
        -> Result<void, EHAPError>
    {
        if (gradients.empty())
            return unexpected(EHAPError::kZeroSizeTensor);

        if (!config_.use_diagonal_fisher)
            return {};

        if (fisher_diag_.empty()) {
            auto shp = std::span(gradients.shape().data(), gradients.rank());
            fisher_diag_ = TensorDense<F>(shp, gradients.device());
            step_count_  = 0;
        }

        if (fisher_diag_.size() != gradients.size())
            return unexpected(EHAPError::kShapeMismatch);

        F beta  = config_.ema_decay;
        F one_m_beta = static_cast<F>(1) - beta;

        if constexpr (std::is_same_v<F, float>) {
            if (gradients.device() == DeviceLocation::kDevice) {
                // GPU path: ema-accumulate = beta*old + alpha*(1-beta)*g^2
                // We reuse fisher_accumulate_kernel with alpha' = alpha * (1-beta)
                F gpu_alpha = alpha * one_m_beta;
                if (step_count_ == 0) {
                    gpu_alpha = alpha;  // first step: no decay
                } else {
                    // Apply beta decay first (only on GPU for now via host-side loop)
                    // For simplicity on GPU: decay handled by the kernel launch param.
                    // The kernel does F[i] += alpha*g[i]^2.
                    // We pre-multiply: gpu_alpha = alpha * (1-beta), then
                    // after kernel the effective update is factor*(1-beta)*g^2.
                    // beta decay must be applied separately.
                    F* fd = fisher_diag_.data();
                    std::size_t N = fisher_diag_.size();
                    for (std::size_t i = 0; i < N; ++i)
                        fd[i] *= beta;
                }
                kernels::launch_fisher_accumulate(
                    fisher_diag_.data(), gradients.data(),
                    gradients.size(), gpu_alpha, nullptr);
                CUDA_SYNC_CHECK();
                ++step_count_;
                return {};
            }
        }

        // --- CPU path: EMA accumulation ---
        F* __restrict__       fo = fisher_diag_.data();
        const F* __restrict__ gi = gradients.data();
        std::size_t           N  = gradients.size();

        F decay = beta;
        F contrib = alpha * one_m_beta;

        if (step_count_ == 0) {
            // First step: initialise Fisher from gradients
            for (std::size_t i = 0; i < N; ++i)
                fo[i] = alpha * gi[i] * gi[i];
        } else {
            // Subsequent steps: EMA update
            for (std::size_t i = 0; i < N; ++i)
                fo[i] = decay * fo[i] + contrib * gi[i] * gi[i];
        }
        ++step_count_;
        return {};
    }

    // =======================================================================
    // compute_importance — importance scores with configurable formulation
    // =======================================================================
    /// @brief Computes per-weight importance scores.
    ///
    /// Supports three formulations (see ImportanceMode):
    ///   - kOBD:  s_i = w_i^2 * (F_ii + λ)
    ///   - kOBS:  s_i = w_i^2 / (F_ii + λ)
    ///   - kNormalized: s_i = w_i^2*(F_ii+λ) / (1 + w_i^2)
    ///
    /// When Fisher is unavailable, falls back to magnitude: s_i = w_i^2.
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
        bool have_f = !fisher_diag_.empty();

        if (have_f && config_.normalize_fisher) {
            apply_fisher_normalization();
        }

        if constexpr (std::is_same_v<F, float>) {
            if (weights.device() == DeviceLocation::kDevice) {
                const float* fp = (have_f) ? fisher_diag_.data() : nullptr;
                kernels::launch_ehap_importance(
                    weights.data(), fp, out_importance.data(),
                    N, config_.damping, nullptr);
                CUDA_SYNC_CHECK();
                // GPU kernel always uses OBD formula.
                // For kOBS / kNormalized, we transform on CPU after GPU compute.
                // For now, GPU path supports kOBD only; other modes fall back.
                if (config_.importance_mode == ImportanceMode::kOBD)
                    return {};
                // Fall through to CPU for transformation of GPU results
                // (importance was already computed by GPU in OBD mode;
                //  we transform it below.)
            }
        }

        // --- CPU path ---
        const F* __restrict__ wi = weights.data();
        F* __restrict__       so = out_importance.data();

        if (!have_f) {
            // Magnitude fallback
            for (std::size_t i = 0; i < N; ++i) {
                F w = wi[i];
                so[i] = w * w;
            }
            return {};
        }

        const F* __restrict__ fi = fisher_diag_.data();

        switch (config_.importance_mode) {
        case ImportanceMode::kOBD:
            // LeCun et al. (1990): s_i = w_i^2 · (F_ii + λ)
            for (std::size_t i = 0; i < N; ++i) {
                F w = wi[i];
                so[i] = w * w * (fi[i] + damp);
            }
            break;
        case ImportanceMode::kOBS:
            // Hassibi & Stork (1993) diagonal approx: s_i = w_i^2 / (F_ii + λ)
            for (std::size_t i = 0; i < N; ++i) {
                F w = wi[i];
                so[i] = (w * w) / (fi[i] + damp);
            }
            break;
        case ImportanceMode::kNormalized:
            // Scale-robust variant
            for (std::size_t i = 0; i < N; ++i) {
                F w = wi[i];
                so[i] = (w * w * (fi[i] + damp)) / (static_cast<F>(1) + w * w);
            }
            break;
        }
        return {};
    }

    // =======================================================================
    // select_pruning_mask — threshold-based mask generation
    // =======================================================================
    /// @brief Selects weights to prune using O(N) nth_element thresholding.
    ///
    /// Finds the (1-ρ)·N percentile of importance scores. Weights with score
    /// below the threshold are marked for pruning (mask = 0).
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

        // O(N) threshold via nth_element (Hoare's quickselect).
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

    // =======================================================================
    // apply_mask — zero out pruned weights in-place
    // =======================================================================
    auto apply_mask(TensorDense<F>& weights, const TensorDense<uint8_t>& mask)
        -> Result<std::size_t, EHAPError>
    {
        if (weights.empty() || mask.empty())
            return unexpected(EHAPError::kZeroSizeTensor);
        if (weights.size() != mask.size())
            return unexpected(EHAPError::kShapeMismatch);

        std::size_t N = weights.size();
        std::size_t pruned = 0;

        if (weights.device() == DeviceLocation::kHost) {
            F* __restrict__              w = weights.data();
            const uint8_t* __restrict__  m = mask.data();
            for (std::size_t i = 0; i < N; ++i) {
                if (!m[i]) { w[i] = static_cast<F>(0); ++pruned; }
            }
        } else {
            // Copy to host, apply, copy back
            auto hw = weights.to_host();
            CUDA_SYNC_CHECK();
            F* __restrict__              w = hw.data();
            const uint8_t* __restrict__  m = mask.data();
            for (std::size_t i = 0; i < N; ++i) {
                if (!m[i]) { w[i] = static_cast<F>(0); ++pruned; }
            }
#ifdef TENSORBIT_ENABLE_CUDA
            CUDA_CHECK(cudaMemcpy(weights.data(), hw.data(),
                                  N * sizeof(F), cudaMemcpyHostToDevice));
#endif
        }
        return pruned;
    }

    // =======================================================================
    // compensate_weights — OBS-style post-pruning weight adjustment
    // =======================================================================
    /// @brief Compensates remaining weights for the information lost by pruning.
    ///
    /// **Bias compensation** (LeCun et al. 1990, Sec. 3):
    /// Adds the sum of pruned weights to the bias term, which is a first-order
    /// approximation to the OBS δw correction when the Hessian is diagonal.
    ///
    /// **Redistribution** (OBS-inspired):
    /// Within each contiguous group, redistributes the total magnitude of
    /// pruned weights to kept weights proportionally to their Fisher values.
    ///
    /// @param weights    Weight tensor (host-resident, modified in-place).
    /// @param mask       Binary mask (1 = keep).
    /// @param importance Importance scores (for redistribution weighting).
    auto compensate_weights(TensorDense<F>&            weights,
                            const TensorDense<uint8_t>& mask,
                            const TensorDense<F>&       importance)
        -> Result<void, EHAPError>
    {
        if (config_.compensation_mode == CompensationMode::kNone)
            return {};

        if (weights.device() != DeviceLocation::kHost)
            return unexpected(EHAPError::kCudaNotAvailable);

        std::size_t N = weights.size();
        F* __restrict__              w = weights.data();
        const uint8_t* __restrict__  m = mask.data();
        const F* __restrict__        s = importance.data();

        if (config_.compensation_mode == CompensationMode::kBias) {
            // OBD-style bias update: b += sum of pruned weights
            // Applied per-layer: the caller provides the per-layer weight
            // vector. We accumulate the sum of pruned weights and add it
            // to the last element (assumed to be bias).
            F bias_delta = static_cast<F>(0);
            for (std::size_t i = 0; i < N; ++i) {
                if (!m[i]) bias_delta += w[i];
            }
            // Append bias delta to the last weight entry (bias convention)
            if (N > 0) w[N - 1] += bias_delta;
        } else if (config_.compensation_mode == CompensationMode::kRedist) {
            // Redistribution: for each group, collect pruned contributions
            // and distribute them to kept weights weighted by Fisher.
            // We use a fixed group size of 8 (configurable in future).
            constexpr std::size_t group_sz = 8;
            for (std::size_t g = 0; g < N; g += group_sz) {
                std::size_t end = (g + group_sz < N) ? g + group_sz : N;

                F total_pruned = static_cast<F>(0);
                F total_weight = static_cast<F>(0);
                for (std::size_t i = g; i < end; ++i) {
                    if (!m[i]) total_pruned += w[i];
                    else       total_weight += s[i];
                }
                if (total_weight <= static_cast<F>(0)) continue;

                for (std::size_t i = g; i < end; ++i) {
                    if (m[i]) {
                        F share = s[i] / total_weight;
                        w[i] += total_pruned * share;
                    }
                }
            }
        }
        return {};
    }

    // =======================================================================
    // prune — full one-shot EHAP pipeline
    // =======================================================================
    /// @brief Executes the complete one-shot EHAP pruning pipeline.
    ///
    /// 1. Compute importance scores from weights and Fisher diagonal.
    /// 2. Select pruning mask via thresholding.
    /// 3. Apply mask (zero out pruned weights).
    /// 4. Compensate remaining weights.
    auto prune(TensorDense<F>& weights) -> Result<std::size_t, EHAPError>
    {
        if (weights.empty())
            return unexpected(EHAPError::kZeroSizeTensor);
        if (weights.device() != DeviceLocation::kHost)
            return unexpected(EHAPError::kCudaNotAvailable);

        if (config_.prune_strategy == PruneStrategy::kIterative) {
            return prune_iterative(weights);
        }

        return prune_one_shot(weights);
    }

    // =======================================================================
    // prune_one_shot — single-pass pruning
    // =======================================================================
    auto prune_one_shot(TensorDense<F>& weights) -> Result<std::size_t, EHAPError>
    {
        auto shp = std::span(weights.shape().data(), weights.rank());
        TensorDense<F> imp(shp, DeviceLocation::kHost);
        TensorDense<uint8_t> mask(shp, DeviceLocation::kHost);

        auto r1 = compute_importance(weights, imp);
        if (!r1) return unexpected(r1.error());

        auto r2 = select_pruning_mask(imp, mask);
        if (!r2) return unexpected(r2.error());

        auto r3 = apply_mask(weights, mask);
        if (!r3) return unexpected(r3.error());

        auto r4 = compensate_weights(weights, mask, imp);
        if (!r4) return unexpected(r4.error());

        return r3.value();
    }

    // =======================================================================
    // prune_iterative — multi-round pruning with compensation
    // =======================================================================
    /// @brief Iterative pruning: prune a fraction per round, compensate,
    /// recompute importance, repeat.
    ///
    /// Follows the Gradual Pruning schedule (Zhu & Gupta 2017):
    /// ρ_t = ρ_final + (ρ_0 - ρ_final) * (1 - t/T)^3
    ///
    /// Each round: compute importance → select mask → apply → compensate.
    auto prune_iterative(TensorDense<F>& weights) -> Result<std::size_t, EHAPError>
    {
        std::size_t rounds = config_.prune_rounds;
        float target_sp = config_.sparsity_ratio;

        std::size_t total_pruned = 0;
        float current_keep = 1.0f; // fraction of weights currently kept

        for (std::size_t r = 0; r < rounds; ++r) {
            // Cubic schedule: fraction of REMAINING weights to prune this round
            float t = static_cast<float>(r + 1) / static_cast<float>(rounds);
            float target_keep = target_sp + (1.0f - target_sp)
                * static_cast<float>(std::pow(1.0 - t, 3));
            float round_keep = target_keep / current_keep;
            if (round_keep < 0.0f) round_keep = 0.0f;
            if (round_keep > 1.0f) round_keep = 1.0f;

            // Temporarily set the sparsity for this round
            float saved_sparsity = config_.sparsity_ratio;
            config_.sparsity_ratio = round_keep;

            auto shp = std::span(weights.shape().data(), weights.rank());
            TensorDense<F> imp(shp, DeviceLocation::kHost);
            TensorDense<uint8_t> mask(shp, DeviceLocation::kHost);

            auto r1 = compute_importance(weights, imp);
            if (!r1) { config_.sparsity_ratio = saved_sparsity; return unexpected(r1.error()); }

            auto r2 = select_pruning_mask(imp, mask);
            if (!r2) { config_.sparsity_ratio = saved_sparsity; return unexpected(r2.error()); }

            auto r3 = apply_mask(weights, mask);
            if (!r3) { config_.sparsity_ratio = saved_sparsity; return unexpected(r3.error()); }
            total_pruned += r3.value();

            auto r4 = compensate_weights(weights, mask, imp);
            config_.sparsity_ratio = saved_sparsity;
            if (!r4) return unexpected(r4.error());

            current_keep = round_keep;
        }

        // Final one-shot pass at target sparsity
        config_.prune_strategy = PruneStrategy::kOneShot;
        auto final = prune_one_shot(weights);
        if (final) total_pruned += final.value();

        return total_pruned;
    }

    // -- Accessors --
    [[nodiscard]] const EHAPConfig& config() const noexcept { return config_; }
    [[nodiscard]] const TensorDense<F>& fisher_diagonal() const noexcept { return fisher_diag_; }
    [[nodiscard]] std::size_t step_count() const noexcept { return step_count_; }
    void reset() { fisher_diag_ = TensorDense<F>{}; step_count_ = 0; }

private:
    // L2-normalise Fisher diagonal per-layer for cross-layer stability.
    void apply_fisher_normalization() {
        if (fisher_diag_.empty()) return;
        std::size_t N = fisher_diag_.size();
        F* fd = fisher_diag_.data();
        F norm = static_cast<F>(0);
        for (std::size_t i = 0; i < N; ++i) norm += fd[i] * fd[i];
        norm = std::sqrt(norm);
        if (norm > static_cast<F>(0)) {
            F inv_norm = static_cast<F>(1) / norm;
            for (std::size_t i = 0; i < N; ++i) fd[i] *= inv_norm;
        }
    }

    EHAPConfig    config_;
    TensorDense<F> fisher_diag_;
    std::size_t   step_count_;
};

}  // namespace tensorbit::core
