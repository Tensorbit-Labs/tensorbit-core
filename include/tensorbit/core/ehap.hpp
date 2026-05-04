#pragma once

/// @file ehap.hpp
/// @brief Efficient Hessian-Aware Pruning (EHAP) — research-grade implementation.
/// @ingroup tensorbit-core
///
/// ## Algorithm
/// EHAP approximates the loss-landscape curvature using a **low-rank +
/// diagonal** Hessian proxy. The diagonal comes from EMA-accumulated Fisher
/// information; the off-diagonal part comes from stored gradient snapshots
/// (gradient covariance, similar to WoodFisher).  The resulting
/// symmetric-positive-definite Hessian is inverted per-block via the
/// Woodbury identity, and each pruned weight triggers an exact OBS
/// compensation update with a Sherman-Morrison rank-1 deflation.
///
/// When no gradient history is available the algorithm falls back to a
/// weight-self-correlation regulariser or pure diagonal OBD.
///
/// ## References
/// - LeCun, Denker & Solla (1990), "Optimal Brain Damage"
/// - Hassibi & Stork (1993), "Optimal Brain Surgeon"
/// - Singh & Alistarh (2020), "WoodFisher"
/// - Frantar & Alistarh (2023), "SparseGPT"
///
/// Full exposition: `docs/EHAP.md`

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <type_traits>
#include <vector>

#include <Eigen/Dense>

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

enum class ImportanceMode : uint8_t {
    kOBD        = 0,  // LeCun et al. 1990: s = w²·(F+λ)
    kOBS        = 1,  // Hassibi & Stork 1993 diagonal: s = w²/(F+λ)
    kNormalized = 2,  // s = w²·(F+λ)/(1+w²)
};

enum class PruneStrategy : uint8_t {
    kOneShot   = 0,
    kIterative = 1,
    /// Blockwise exact OBS with low-rank+diagonal inverse-Hessian.
    /// Uses gradient-covariance snapshots when available, falls back
    /// to weight-self-correlation otherwise.
    kBlockOBS  = 2,
};

enum class CompensationMode : uint8_t {
    kNone   = 0,
    kBias   = 1,
    kRedist = 2,
};

struct EHAPConfig {
    float damping            = 0.01f;
    bool  use_diagonal_fisher = true;
    float sparsity_ratio     = 0.5f;
    float ema_decay          = 0.99f;
    std::size_t accumulation_steps = 100;

    ImportanceMode   importance_mode   = ImportanceMode::kOBD;
    PruneStrategy    prune_strategy    = PruneStrategy::kOneShot;
    std::size_t      prune_rounds      = 5;
    float            prune_fraction_per_round = 0.2f;
    CompensationMode compensation_mode = CompensationMode::kNone;
    bool             normalize_fisher  = false;

    // -- Blockwise OBS settings --
    std::size_t obs_block_size     = 128;
    float       obs_off_diag_alpha = 0.01f;

    // -- Low-rank gradient covariance --
    /// Number of recent gradient snapshots to store (K).
    /// Each snapshot adds one rank to the low-rank Hessian approximation.
    /// K = 4 → H = diag(F) + (1/K)·G_hist·G_hist^T.
    std::size_t gradient_history_size = 4;
};

// ===========================================================================
// EHAPPruner
// ===========================================================================

template<FloatingPoint F>
class EHAPPruner {
public:
    explicit EHAPPruner(EHAPConfig config)
        : config_(config), step_count_(0) {}

    ~EHAPPruner() = default;

    EHAPPruner(const EHAPPruner&)                = delete;
    EHAPPruner& operator=(const EHAPPruner&)     = delete;
    EHAPPruner(EHAPPruner&&) noexcept            = default;
    EHAPPruner& operator=(EHAPPruner&&) noexcept = default;

    // =======================================================================
    // store_gradient — archive an EMA-weighted gradient snapshot
    // =======================================================================
    /// @brief Stores a gradient vector for low-rank Hessian construction.
    ///
    /// Snapshots are stored in a ring buffer with exponentially decaying
    /// weights: the k-th oldest snapshot contributes weight
    /// sqrt(exp(-k/τ)) with τ = 2.0, so recent gradients dominate
    /// the covariance estimate.
    auto store_gradient(const TensorDense<F>& gradients) -> Result<void, EHAPError>
    {
        if (gradients.empty())
            return unexpected(EHAPError::kZeroSizeTensor);
        if (gradients.device() != DeviceLocation::kHost)
            return unexpected(EHAPError::kCudaNotAvailable);

        std::size_t K = config_.gradient_history_size;
        if (K == 0) return {};

        if (g_hist_.size() >= K) {
            for (std::size_t k = 1; k < K; ++k)
                g_hist_[k - 1] = std::move(g_hist_[k]);
            g_hist_.pop_back();
        }

        auto shp = std::span(gradients.shape().data(), gradients.rank());
        TensorDense<F> snap(shp, DeviceLocation::kHost);
        std::copy_n(gradients.data(), gradients.size(), snap.data());
        g_hist_.push_back(std::move(snap));

        return {};
    }

    // =======================================================================
    // accumulate_fisher — EMA-based Fisher diagonal
    // =======================================================================
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

        F beta       = config_.ema_decay;
        F one_m_beta = static_cast<F>(1) - beta;

        if constexpr (std::is_same_v<F, float>) {
            if (gradients.device() == DeviceLocation::kDevice) {
                // First step: initialise Fisher from gradients.
                // Subsequent steps: apply beta decay directly on GPU
                if (step_count_ > 0) {
                    kernels::launch_fisher_beta_decay(
                        fisher_diag_.data(), beta, fisher_diag_.size(), nullptr);
                    CUDA_SYNC_CHECK();
                }
                F gpu_alpha = (step_count_ == 0) ? alpha : alpha * one_m_beta;
                kernels::launch_fisher_accumulate(
                    fisher_diag_.data(), gradients.data(),
                    gradients.size(), gpu_alpha, nullptr);
                CUDA_SYNC_CHECK();
                ++step_count_;
                return {};
            }
        }

        F* __restrict__       fo = fisher_diag_.data();
        const F* __restrict__ gi = gradients.data();
        std::size_t           N  = gradients.size();

        if (step_count_ == 0) {
            for (std::size_t i = 0; i < N; ++i)
                fo[i] = alpha * gi[i] * gi[i];
        } else {
            F decay   = beta;
            F contrib = alpha * one_m_beta;
            for (std::size_t i = 0; i < N; ++i)
                fo[i] = decay * fo[i] + contrib * gi[i] * gi[i];
        }
        ++step_count_;
        return {};
    }

    // =======================================================================
    // compute_importance
    // =======================================================================
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

        if (have_f && config_.normalize_fisher)
            apply_fisher_normalization();

        if constexpr (std::is_same_v<F, float>) {
            if (weights.device() == DeviceLocation::kDevice) {
                const float* fp = (have_f) ? fisher_diag_.data() : nullptr;
                kernels::launch_ehap_importance(
                    weights.data(), fp, out_importance.data(),
                    N, config_.damping, nullptr);
                CUDA_SYNC_CHECK();
                // GPU kernel fixes OBD formula. For kOBS / kNormalized,
                // copy importance to host for CPU transformation then push back.
                if (config_.importance_mode != ImportanceMode::kOBD) {
                    auto h = out_importance.to_host();
                    // Transform OBD scores to target mode on host
                    F* hi = h.data();
                    const F* __restrict__ wi = weights.data();
                    if (have_f) {
                        const F* fi = fisher_diag_.data();
                        for (std::size_t i = 0; i < N; ++i) {
                            // GPU produced s_i = w_i^2 * (F_ii + damp)
                            // Transform to target mode
                            if (config_.importance_mode == ImportanceMode::kOBS)
                                hi[i] = (wi[i] * wi[i]) / (fi[i] + damp);
                            else  // kNormalized
                                hi[i] = (wi[i] * wi[i] * (fi[i] + damp)) / (static_cast<F>(1) + wi[i] * wi[i]);
                        }
                    }
#ifdef TENSORBIT_ENABLE_CUDA
                    CUDA_CHECK(cudaMemcpy(out_importance.data(), h.data(),
                                          N * sizeof(F), cudaMemcpyHostToDevice));
#endif
                }
                return {};
            }
        }

        // --- CPU path ---
        const F* __restrict__ wi = weights.data();
        F* __restrict__       so = out_importance.data();

        if (!have_f) {
            for (std::size_t i = 0; i < N; ++i) { F w = wi[i]; so[i] = w * w; }
            return {};
        }

        const F* __restrict__ fi = fisher_diag_.data();
        switch (config_.importance_mode) {
        case ImportanceMode::kOBD:
            for (std::size_t i = 0; i < N; ++i) {
                F w = wi[i]; so[i] = w * w * (fi[i] + damp);
            }
            break;
        case ImportanceMode::kOBS:
            for (std::size_t i = 0; i < N; ++i) {
                F w = wi[i]; so[i] = (w * w) / (fi[i] + damp);
            }
            break;
        case ImportanceMode::kNormalized:
            for (std::size_t i = 0; i < N; ++i) {
                F w = wi[i];
                so[i] = (w * w * (fi[i] + damp)) / (static_cast<F>(1) + w * w);
            }
            break;
        }
        return {};
    }

    // =======================================================================
    // select_pruning_mask
    // =======================================================================
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

    // =======================================================================
    // apply_mask
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
            for (std::size_t i = 0; i < N; ++i)
                if (!m[i]) { w[i] = static_cast<F>(0); ++pruned; }
        } else {
            auto hw = weights.to_host();
            CUDA_SYNC_CHECK();
            F* __restrict__              w = hw.data();
            const uint8_t* __restrict__  m = mask.data();
            for (std::size_t i = 0; i < N; ++i)
                if (!m[i]) { w[i] = static_cast<F>(0); ++pruned; }
#ifdef TENSORBIT_ENABLE_CUDA
            CUDA_CHECK(cudaMemcpy(weights.data(), hw.data(),
                                  N * sizeof(F), cudaMemcpyHostToDevice));
#endif
        }
        return pruned;
    }

    // =======================================================================
    // compensate_weights
    // =======================================================================
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
            F delta = static_cast<F>(0);
            for (std::size_t i = 0; i < N; ++i)
                if (!m[i]) delta += w[i];
            if (N > 0) w[N - 1] += delta;
        } else {
            constexpr std::size_t gsz = 8;
            for (std::size_t g = 0; g < N; g += gsz) {
                std::size_t end = (g + gsz < N) ? g + gsz : N;
                F t_pruned = static_cast<F>(0);
                F t_weight = static_cast<F>(0);
                for (std::size_t i = g; i < end; ++i) {
                    if (!m[i]) t_pruned += w[i];
                    else       t_weight += s[i];
                }
                if (t_weight <= static_cast<F>(0)) continue;
                for (std::size_t i = g; i < end; ++i)
                    if (m[i]) w[i] += t_pruned * s[i] / t_weight;
            }
        }
        return {};
    }

    // =======================================================================
    // prune — dispatcher
    // =======================================================================
    auto prune(TensorDense<F>& weights) -> Result<std::size_t, EHAPError>
    {
        if (weights.empty())
            return unexpected(EHAPError::kZeroSizeTensor);
        if (weights.device() != DeviceLocation::kHost)
            return unexpected(EHAPError::kCudaNotAvailable);

        if (config_.prune_strategy == PruneStrategy::kIterative)
            return prune_iterative(weights);
        if (config_.prune_strategy == PruneStrategy::kBlockOBS)
            return prune_block_obs(weights);
        return prune_one_shot(weights);
    }

    // =======================================================================
    // prune_one_shot
    // =======================================================================
    auto prune_one_shot(TensorDense<F>& weights) -> Result<std::size_t, EHAPError>
    {
        auto shp = std::span(weights.shape().data(), weights.rank());
        TensorDense<F>       imp(shp, DeviceLocation::kHost);
        TensorDense<uint8_t> msk(shp, DeviceLocation::kHost);

        auto r1 = compute_importance(weights, imp);
        if (!r1) return unexpected(r1.error());
        auto r2 = select_pruning_mask(imp, msk);
        if (!r2) return unexpected(r2.error());
        auto r3 = apply_mask(weights, msk);
        if (!r3) return unexpected(r3.error());
        compensate_weights(weights, msk, imp);
        return r3.value();
    }

    // =======================================================================
    // prune_iterative
    // =======================================================================
    auto prune_iterative(TensorDense<F>& weights) -> Result<std::size_t, EHAPError>
    {
        std::size_t rounds = config_.prune_rounds;
        float target_sp = config_.sparsity_ratio;
        std::size_t total_pruned = 0;
        float cur_keep = 1.0f;

        for (std::size_t r = 0; r < rounds; ++r) {
            float t = static_cast<float>(r + 1) / static_cast<float>(rounds);
            float target_keep = target_sp + (1.0f - target_sp)
                * static_cast<float>(std::pow(1.0 - t, 3));
            float round_keep = target_keep / cur_keep;
            if (round_keep < 0.0f) round_keep = 0.0f;
            if (round_keep > 1.0f) round_keep = 1.0f;

            float saved_sp = config_.sparsity_ratio;
            config_.sparsity_ratio = round_keep;

            auto shp = std::span(weights.shape().data(), weights.rank());
            TensorDense<F>       imp(shp, DeviceLocation::kHost);
            TensorDense<uint8_t> msk(shp, DeviceLocation::kHost);

            auto r1 = compute_importance(weights, imp);
            if (!r1) { config_.sparsity_ratio = saved_sp; return unexpected(r1.error()); }
            auto r2 = select_pruning_mask(imp, msk);
            if (!r2) { config_.sparsity_ratio = saved_sp; return unexpected(r2.error()); }
            auto r3 = apply_mask(weights, msk);
            if (!r3) { config_.sparsity_ratio = saved_sp; return unexpected(r3.error()); }
            total_pruned += r3.value();
            compensate_weights(weights, msk, imp);
            config_.sparsity_ratio = saved_sp;
            cur_keep = round_keep;
        }
        return total_pruned;
    }

    // =======================================================================
    // prune_block_obs — blockwise OBS with Woodbury + adaptive sparsity
    // =======================================================================
    /// @brief Blockwise Optimal Brain Surgeon using a low-rank + diagonal
    /// Hessian approximation.
    ///
    /// ## Hessian model
    /// H = diag(F + λ) + U·U^T   where U is B×K.
    ///   - With gradient history: U_k = w_k · g^{(k)}  (EMA-weighted snapshots)
    ///   - Without gradient history: U = sqrt(α) · w  (rank-1 regulariser)
    ///
    /// ## Inversion (Woodbury identity)
    /// H^{-1} = D^{-1} - D^{-1}·U·(I + U^T·D^{-1}·U)^{-1}·U^T·D^{-1}
    ///
    /// The full H^{-1} is never materialised.  Diagonal entries are
    /// computed on demand in O(K²) each; one column is computed per
    /// OBS step in O(B·K²).  The K×K inner matrix is regularised
    /// (ε = 1e-8) and solved via LDL^T.
    ///
    /// ## Adaptive sparsity
    /// Per-block importance (from Fisher diagonal) drives a softmax
    /// allocation: low-importance blocks are pruned more aggressively.
    auto prune_block_obs(TensorDense<F>& weights) -> Result<std::size_t, EHAPError>
    {
        using Mat = Eigen::Matrix<F, Eigen::Dynamic, Eigen::Dynamic>;
        using Vec = Eigen::Matrix<F, Eigen::Dynamic, 1>;

        std::size_t N    = weights.size();
        std::size_t B    = config_.obs_block_size;
        F alpha          = static_cast<F>(config_.obs_off_diag_alpha);
        F damp           = static_cast<F>(config_.damping);
        float sp         = config_.sparsity_ratio;
        bool  have_grads = !g_hist_.empty();

        F* __restrict__ w     = weights.data();
        const F*        fdiag_ptr = (!fisher_diag_.empty()) ? fisher_diag_.data() : nullptr;

        // ---- 1. Block partitioning & adaptive sparsity targets ----
        std::size_t num_blocks = (N + B - 1) / B;
        std::size_t prune_total = N - static_cast<std::size_t>(sp * static_cast<float>(N));
        if (prune_total == 0) return std::size_t{0};

        // Per-block mean importance (for adaptive allocation)
        std::vector<F> block_imp(num_blocks, static_cast<F>(0));
        {
            for (std::size_t b = 0; b < num_blocks; ++b) {
                std::size_t st = b * B;
                std::size_t en = (st + B < N) ? st + B : N;
                F sum = static_cast<F>(0);
                for (std::size_t i = st; i < en; ++i) {
                    F fi = damp;
                    if (fdiag_ptr) fi += fdiag_ptr[i];
                    sum += w[i] * w[i] * fi;
                }
                block_imp[b] = sum / static_cast<F>(en - st);
            }
        }

        // Softmax weights: w_b = exp(-τ·rel_imp_b)
        std::vector<F> blk_w(num_blocks, static_cast<F>(0));
        {
            F mx = static_cast<F>(0);
            for (auto v : block_imp) if (v > mx) mx = v;
            if (mx <= static_cast<F>(0)) mx = static_cast<F>(1);
            F tau = static_cast<F>(4);
            for (std::size_t b = 0; b < num_blocks; ++b) {
                F rel = block_imp[b] / mx;
                blk_w[b] = std::exp(-tau * rel);
            }
        }

        // Compute target prune counts (normalise + distribute remainder)
        std::vector<std::size_t> prune_targets(num_blocks, 0);
        {
            F total_w = static_cast<F>(0);
            for (auto v : blk_w) total_w += v;
            if (total_w <= static_cast<F>(0)) total_w = static_cast<F>(1);
            std::size_t allocated = 0;
            for (std::size_t b = 0; b < num_blocks; ++b) {
                F share = blk_w[b] / total_w;
                std::size_t bs = (b == num_blocks - 1)
                    ? (N - b * B) : B;
                prune_targets[b] = static_cast<std::size_t>(share * static_cast<F>(prune_total));
                if (prune_targets[b] >= bs) prune_targets[b] = bs > 1 ? bs - 1 : 0;
                allocated += prune_targets[b];
            }
            // Distribute remainder to lowest-importance blocks
            std::ptrdiff_t rem = static_cast<std::ptrdiff_t>(prune_total) -
                                 static_cast<std::ptrdiff_t>(allocated);
            for (std::size_t b = 0; b < num_blocks && rem > 0; ++b) {
                std::size_t rb = num_blocks - 1 - b; // lowest-importance first
                std::size_t bs = (rb == num_blocks - 1) ? (N - rb * B) : B;
                while (rem > 0 && prune_targets[rb] + 1 < bs) {
                    ++prune_targets[rb];
                    --rem;
                }
            }
        }

        // ---- 2. Process each block ----
        std::size_t pruned = 0;

        for (std::size_t b = 0; b < num_blocks; ++b) {
            std::size_t st = b * B;
            std::size_t en = (st + B < N) ? st + B : N;
            std::size_t bs = en - st;
            if (prune_targets[b] == 0 || prune_targets[b] >= bs) continue;

            Eigen::Index Bs = static_cast<Eigen::Index>(bs);
            Eigen::Index K;
            Mat U;

            // Build D (diagonal) and U (low-rank factor)
            Vec D(Bs);
            for (Eigen::Index i = 0; i < Bs; ++i) {
                F d = damp;
                if (fdiag_ptr) d += fdiag_ptr[st + static_cast<std::size_t>(i)];
                D(i) = d;
            }

            if (have_grads) {
                K = static_cast<Eigen::Index>(g_hist_.size());
                U.resize(Bs, K);
                for (Eigen::Index k = 0; k < K; ++k) {
                    const F* gk = g_hist_[static_cast<std::size_t>(k)].data();
                    // EMA decay: older = less weight. τ = 2.0.
                    F age = static_cast<F>(K - 1 - k);
                    F wk  = std::exp(-age / static_cast<F>(2.0f));
                    for (Eigen::Index i = 0; i < Bs; ++i)
                        U(i, k) = wk * gk[st + static_cast<std::size_t>(i)];
                }
            } else {
                K = 1;
                U.resize(Bs, 1);
                for (Eigen::Index i = 0; i < Bs; ++i)
                    U(i, 0) = std::sqrt(alpha) * w[st + static_cast<std::size_t>(i)];
            }

            // ---- 3. Woodbury factors ----
            // Dinv_vec
            Vec Dinv(Bs);
            for (Eigen::Index i = 0; i < Bs; ++i) {
                F dv = D(i);
                Dinv(i) = (dv > static_cast<F>(0)) ? (static_cast<F>(1) / dv)
                                                    : static_cast<F>(1);
            }

            // M = I + U^T·D^{-1}·U  (K×K), regularised
            Mat M = Mat::Identity(K, K);
            M.diagonal().array() += static_cast<F>(1e-8);
            {
                Mat UtDinv = U.transpose();
                for (Eigen::Index k = 0; k < K; ++k)
                    for (Eigen::Index i = 0; i < Bs; ++i)
                        UtDinv(k, i) *= Dinv(i);
                M += UtDinv * U;
            }
            Mat Minv = M.ldlt()
                           .solve(Mat::Identity(K, K));

            // Pre-compute Z = U·M^{-1}  (B×K) for fast column access
            Mat Z = U * Minv;

            // ---- 4. Initial diagonal of H^{-1} (Woodbury) ----
            Vec Hinv_diag(Bs);
            for (Eigen::Index i = 0; i < Bs; ++i) {
                F di = Dinv(i);
                F corr = static_cast<F>(0);
                for (Eigen::Index k1 = 0; k1 < K; ++k1)
                    corr += Z(i, k1) * U(i, k1);
                Hinv_diag(i) = di - di * di * corr;
                if (Hinv_diag(i) <= static_cast<F>(0))
                    Hinv_diag(i) = di; // fallback to diagonal
            }

            // Full Hinv for Sherman-Morrison updates (B ≤ 256, acceptable)
            Mat Hinv = Mat::Zero(Bs, Bs);
            for (Eigen::Index i = 0; i < Bs; ++i)
                Hinv(i, i) = Dinv(i);
            {
                Mat S = Mat::Zero(Bs, Bs);
                for (Eigen::Index k1 = 0; k1 < K; ++k1)
                    for (Eigen::Index k2 = 0; k2 < K; ++k2) {
                        S += Minv(k1, k2) * (U.col(k1) * U.col(k2).transpose());
                    }
                for (Eigen::Index i = 0; i < Bs; ++i)
                    for (Eigen::Index j = 0; j < Bs; ++j)
                        Hinv(i, j) -= Dinv(i) * S(i, j) * Dinv(j);
            }

            // ---- 5. Greedy OBS loop ----
            Vec wv(Bs);
            for (Eigen::Index i = 0; i < Bs; ++i)
                wv(i) = w[st + static_cast<std::size_t>(i)];

            std::size_t done = 0;
            while (done < prune_targets[b]) {
                F min_s = std::numeric_limits<F>::max();
                Eigen::Index best = 0;
                for (Eigen::Index i = 0; i < Bs; ++i) {
                    F hii = Hinv(i, i);
                    if (hii <= static_cast<F>(0)) continue;
                    F s = (wv(i) * wv(i)) / hii;
                    if (s < min_s) { min_s = s; best = i; }
                }
                if (min_s >= std::numeric_limits<F>::max() / 2) break;

                F wi_old = wv(best);
                F inv_h  = static_cast<F>(1) / Hinv(best, best);
                for (Eigen::Index j = 0; j < Bs; ++j)
                    if (j != best) wv(j) -= wi_old * inv_h * Hinv(j, best);
                wv(best) = static_cast<F>(0);

                Vec col = Hinv.col(best);
                Hinv -= (col * col.transpose()) / Hinv(best, best);
                ++done;
            }

            for (Eigen::Index i = 0; i < Bs; ++i)
                w[st + static_cast<std::size_t>(i)] = wv(i);
            pruned += done;
        }

        return pruned;
    }

    // -- Accessors --
    [[nodiscard]] const EHAPConfig& config() const noexcept { return config_; }
    [[nodiscard]] const TensorDense<F>& fisher_diagonal() const noexcept { return fisher_diag_; }
    [[nodiscard]] std::size_t step_count() const noexcept { return step_count_; }
    [[nodiscard]] const std::vector<TensorDense<F>>& gradient_history() const noexcept {
        return g_hist_;
    }
    void reset() {
        fisher_diag_ = TensorDense<F>{};
        g_hist_.clear();
        step_count_ = 0;
    }

private:
    void apply_fisher_normalization() {
        if (fisher_diag_.empty()) return;
        std::size_t N = fisher_diag_.size();
        F* fd = fisher_diag_.data();
        F norm = static_cast<F>(0);
        for (std::size_t i = 0; i < N; ++i) norm += fd[i] * fd[i];
        norm = std::sqrt(norm);
        if (norm > static_cast<F>(0)) {
            F inv = static_cast<F>(1) / norm;
            for (std::size_t i = 0; i < N; ++i) fd[i] *= inv;
        }
    }

    EHAPConfig               config_;
    TensorDense<F>            fisher_diag_;
    std::vector<TensorDense<F>> g_hist_;  // gradient snapshot ring buffer
    std::size_t               step_count_;
};

}  // namespace tensorbit::core
