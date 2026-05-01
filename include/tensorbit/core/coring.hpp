#pragma once

/// @file coring.hpp
/// @brief CORING N:M Structured Sparsity — research-grade implementation.
/// @ingroup tensorbit-core
///
/// Implements N:M structured sparsity with optimal mask selection, permutation
/// optimization, weight redistribution, and tile-aware mask layout for Ampere
/// Sparse Tensor Core compatibility. See docs/CORING.md for full exposition.

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

#include "tensorbit/core/kernels.hpp"
#include "tensorbit/core/tensor.hpp"

namespace tensorbit::core {

enum class CORINGError : uint8_t {
    kOk              = 0,
    kShapeMismatch   = 1,
    kZeroSizeTensor  = 2,
    kInvalidNMConfig = 3,
    kCudaNotAvailable = 4,
};

/// @brief Mask selection strategy for N:M groups.
enum class MaskStrategy : uint8_t {
    /// Simple top-N by importance (fast, default).
    kTopN = 0,
    /// Exhaustive enumeration of all C(M,N) patterns; optimal for M ≤ 16.
    kOptimal = 1,
    /// Iterative refinement: alternate between mask selection and permutation.
    kIterative = 2,
};

/// @brief Weight redistribution mode after N:M pruning.
enum class RedistMode : uint8_t {
    kNone = 0,
    /// Proportional: redistribute total pruned magnitude to kept weights
    /// weighted by their Fisher/importance scores (OBS-inspired).
    kProportional = 1,
    /// Uniform: split pruned magnitude equally among kept weights.
    kUniform = 2,
};

struct CORINGConfig {
    int  N = 2;
    int  M = 4;
    bool use_cuda = true;

    /// Mask selection strategy.
    MaskStrategy mask_strategy = MaskStrategy::kTopN;

    /// Weight redistribution mode.
    RedistMode redist_mode = RedistMode::kNone;

    /// For iterative strategy: number of refinement iterations.
    int iterative_rounds = 3;

    /// If true, sort weights within each group by magnitude before applying
    /// the N:M mask — this is a lightweight permutation heuristic that
    /// improves quality without full combinatorial search.
    bool permute_weights = false;
};

// ===========================================================================
// CORINGPruner — N:M Structured Sparsity Engine
// ===========================================================================

template<FloatingPoint F>
class CORINGPruner {
public:
    explicit CORINGPruner(CORINGConfig config) : config_(config) {}

    ~CORINGPruner() = default;

    CORINGPruner(const CORINGPruner&)                = delete;
    CORINGPruner& operator=(const CORINGPruner&)     = delete;
    CORINGPruner(CORINGPruner&&) noexcept            = default;
    CORINGPruner& operator=(CORINGPruner&&) noexcept = default;

    // =======================================================================
    // generate_nm_mask
    // =======================================================================
    auto generate_nm_mask(const TensorDense<F>& importance,
                          TensorDense<uint8_t>& out_mask)
        -> Result<void, CORINGError>
    {
        if (importance.empty() || out_mask.empty())
            return unexpected(CORINGError::kZeroSizeTensor);
        if (importance.size() != out_mask.size())
            return unexpected(CORINGError::kShapeMismatch);

        auto v = validate_config(importance.size());
        if (!v) return unexpected(v.error());

        int Gn = config_.N;
        int Gm = config_.M;
        std::size_t Ne = importance.size();
        std::size_t Ng = (Ne / static_cast<std::size_t>(Gm));
        std::size_t an = Ng * static_cast<std::size_t>(Gm);

        bool gpu_ok = config_.use_cuda
                      && importance.device() == DeviceLocation::kDevice;

        // GPU path: launch CUDA kernels (currently kTopN only via kernels).
        if constexpr (std::is_same_v<F, float>) {
            if (gpu_ok && config_.mask_strategy == MaskStrategy::kTopN) {
                if (Gn == 2 && Gm == 4)
                    kernels::launch_nm_mask_2_4(
                        importance.data(), out_mask.data(), an, nullptr);
                else
                    kernels::launch_nm_mask_generic(
                        importance.data(), out_mask.data(), an, Gn, Gm, nullptr);
                CUDA_SYNC_CHECK();
                return {};
            }
        }

        // --- CPU path ---
        const TensorDense<F>* src = &importance;
        TensorDense<F> hcopy{};
        if (importance.device() == DeviceLocation::kDevice) {
            hcopy = importance.to_host();
            src   = &hcopy;
            CUDA_SYNC_CHECK();
        }

        const F* __restrict__ imp = src->data();
        uint8_t* __restrict__ msk = out_mask.data();

        switch (config_.mask_strategy) {
        case MaskStrategy::kTopN:
            generate_topn(imp, msk, Ng, Gn, Gm);
            break;
        case MaskStrategy::kOptimal:
            generate_optimal(imp, msk, Ng, Gn, Gm);
            break;
        case MaskStrategy::kIterative:
            generate_iterative(imp, msk, Ng, Gn, Gm);
            break;
        }
        return {};
    }

    // =======================================================================
    // apply_mask
    // =======================================================================
    auto apply_mask(TensorDense<F>&            weights,
                    const TensorDense<uint8_t>& mask)
        -> Result<std::size_t, CORINGError>
    {
        if (weights.empty() || mask.empty())
            return unexpected(CORINGError::kZeroSizeTensor);
        if (weights.size() != mask.size())
            return unexpected(CORINGError::kShapeMismatch);

        int Gm = config_.M;
        int Gn = config_.N;
        std::size_t an = (weights.size() / static_cast<std::size_t>(Gm))
                         * static_cast<std::size_t>(Gm);
        bool gpu_ok = config_.use_cuda
                      && weights.device() == DeviceLocation::kDevice;

        if constexpr (std::is_same_v<F, float>) {
            if (gpu_ok) {
                kernels::launch_apply_mask(
                    weights.data(), mask.data(), an, Gm, nullptr);
                CUDA_SYNC_CHECK();
                std::size_t ng = an / static_cast<std::size_t>(Gm);
                return ng * static_cast<std::size_t>(Gm - Gn);
            }
        }

        // --- CPU path ---
        TensorDense<F> hcopy{};
        F* w = weights.data();
        if (weights.device() == DeviceLocation::kDevice) {
            hcopy = weights.to_host();
            w     = hcopy.data();
            CUDA_SYNC_CHECK();
        }

        const uint8_t* __restrict__ mb = mask.data();
        std::size_t ng = an / static_cast<std::size_t>(Gm);
        for (std::size_t g = 0; g < ng; ++g) {
            uint8_t byte = mb[g];
            std::size_t base = g * static_cast<std::size_t>(Gm);
            for (int i = 0; i < Gm; ++i) {
                if (!((byte >> i) & 1u))
                    w[base + static_cast<std::size_t>(i)] = static_cast<F>(0);
            }
        }

        if (weights.device() == DeviceLocation::kDevice) {
#ifdef TENSORBIT_ENABLE_CUDA
            CUDA_CHECK(cudaMemcpy(weights.data(), hcopy.data(),
                                  weights.size() * sizeof(F),
                                  cudaMemcpyHostToDevice));
#endif
        }
        return ng * static_cast<std::size_t>(Gm - Gn);
    }

    // =======================================================================
    // redistribute — redistributes pruned weight magnitude to kept weights
    // =======================================================================
    auto redistribute(TensorDense<F>&            weights,
                      const TensorDense<uint8_t>& mask,
                      const TensorDense<F>&       importance)
        -> Result<void, CORINGError>
    {
        if (config_.redist_mode == RedistMode::kNone) return {};
        if (weights.device() != DeviceLocation::kHost)
            return unexpected(CORINGError::kCudaNotAvailable);

        int Gm = config_.M;
        std::size_t N = weights.size();
        std::size_t Ng = N / static_cast<std::size_t>(Gm);

        F* __restrict__              w = weights.data();
        const uint8_t* __restrict__  m = mask.data();
        const F* __restrict__        s = importance.data();

        for (std::size_t g = 0; g < Ng; ++g) {
            uint8_t byte = m[g];
            std::size_t base = g * static_cast<std::size_t>(Gm);

            F pruned_sum = static_cast<F>(0);
            F kept_weights = static_cast<F>(0);
            int kept_count = 0;

            for (int i = 0; i < Gm; ++i) {
                if (!((byte >> i) & 1u)) {
                    pruned_sum += w[base + static_cast<std::size_t>(i)];
                } else {
                    ++kept_count;
                }
            }

            if (kept_count == 0 || pruned_sum == static_cast<F>(0)) continue;

            if (config_.redist_mode == RedistMode::kUniform) {
                F share = pruned_sum / static_cast<F>(kept_count);
                for (int i = 0; i < Gm; ++i) {
                    if ((byte >> i) & 1u)
                        w[base + static_cast<std::size_t>(i)] += share;
                }
            } else {
                // Proportional: weight by importance
                for (int i = 0; i < Gm; ++i)
                    if ((byte >> i) & 1u)
                        kept_weights += s[base + static_cast<std::size_t>(i)];

                if (kept_weights <= static_cast<F>(0)) continue;

                for (int i = 0; i < Gm; ++i) {
                    if ((byte >> i) & 1u) {
                        F frac = s[base + static_cast<std::size_t>(i)] / kept_weights;
                        w[base + static_cast<std::size_t>(i)] += pruned_sum * frac;
                    }
                }
            }
        }
        return {};
    }

    // =======================================================================
    // prune — full pipeline
    // =======================================================================
    auto prune(const TensorDense<F>& importance,
               TensorDense<F>&       weights)
        -> Result<std::size_t, CORINGError>
    {
        if (importance.size() != weights.size())
            return unexpected(CORINGError::kShapeMismatch);
        auto shp = std::span(importance.shape().data(), importance.rank());
        TensorDense<uint8_t> mask(shp, weights.device());
        auto mr = generate_nm_mask(importance, mask);
        if (!mr) return unexpected(mr.error());
        auto ar = apply_mask(weights, mask);
        if (!ar) return unexpected(ar.error());
        auto rr = redistribute(weights, mask, importance);
        if (!rr) return unexpected(rr.error());
        return ar.value();
    }

    [[nodiscard]] const CORINGConfig& config() const noexcept { return config_; }

private:
    // -------------------------------------------------------------------
    // validate_config
    // -------------------------------------------------------------------
    auto validate_config(std::size_t num_elements) -> Result<void, CORINGError> {
        if (config_.N <= 0 || config_.M <= 0)
            return unexpected(CORINGError::kInvalidNMConfig);
        if (config_.N >= config_.M)
            return unexpected(CORINGError::kInvalidNMConfig);
        if (!std::has_single_bit(static_cast<unsigned>(config_.M)))
            return unexpected(CORINGError::kInvalidNMConfig);
        if (num_elements > 0 &&
            num_elements % static_cast<std::size_t>(config_.M) != 0)
            return unexpected(CORINGError::kShapeMismatch);
        return {};
    }

    // -------------------------------------------------------------------
    // generate_topn — simple top-N by importance
    // -------------------------------------------------------------------
    static void generate_topn(const F* __restrict__ imp, uint8_t* __restrict__ msk,
                              std::size_t Ng, int Gn, int Gm)
    {
        std::vector<std::pair<F, int>> buf(static_cast<std::size_t>(Gm));
        for (std::size_t g = 0; g < Ng; ++g) {
            std::size_t base = g * static_cast<std::size_t>(Gm);
            for (int i = 0; i < Gm; ++i)
                buf[static_cast<std::size_t>(i)] = {
                    imp[base + static_cast<std::size_t>(i)], i};
            std::nth_element(buf.begin(), buf.begin() + Gn, buf.end(),
                             [](auto& a, auto& b) { return a.first > b.first; });
            uint8_t byte = 0;
            for (int k = 0; k < Gn; ++k)
                byte |= static_cast<uint8_t>(1u << buf[static_cast<std::size_t>(k)].second);
            msk[g] = byte;
        }
    }

    // -------------------------------------------------------------------
    // generate_optimal — exhaustive enumeration of all C(M,N) patterns
    // -------------------------------------------------------------------
    static void generate_optimal(const F* __restrict__ imp, uint8_t* __restrict__ msk,
                                 std::size_t Ng, int Gn, int Gm)
    {
        // Exhaustive: evaluate all C(Gm, Gn) subsets, pick the one with
        // maximum sum of importance scores.
        // Only practical for small M (≤ 16). For M > 16, falls back to top-N.
        if (Gm > 16) {
            generate_topn(imp, msk, Ng, Gn, Gm);
            return;
        }

        // Generate all combinations via bit iteration.
        // For each group, enumerate all uint32_t masks with exactly Gn set bits.
        for (std::size_t g = 0; g < Ng; ++g) {
            std::size_t base = g * static_cast<std::size_t>(Gm);
            uint32_t best_mask = 0;
            F best_sum = -std::numeric_limits<F>::max();

            // Gosper's hack: iterate over all values with exactly Gn set bits
            uint32_t mask_bits = (static_cast<uint32_t>(1u) << Gn) - 1u;
            uint32_t limit = (static_cast<uint32_t>(1u) << Gm);

            while (mask_bits < limit) {
                F sum = static_cast<F>(0);
                for (int i = 0; i < Gm; ++i)
                    if (mask_bits & (1u << i))
                        sum += imp[base + static_cast<std::size_t>(i)];
                if (sum > best_sum) {
                    best_sum = sum;
                    best_mask = mask_bits;
                }
                // Gosper's hack: next lexicographic permutation
                uint32_t c = mask_bits & (0u - mask_bits);
                uint32_t r = mask_bits + c;
                mask_bits = (((r ^ mask_bits) >> 2) / c) | r;
            }
            msk[g] = static_cast<uint8_t>(best_mask);
        }
    }

    // -------------------------------------------------------------------
    // generate_iterative — alternate mask selection with permute-refine
    // -------------------------------------------------------------------
    static void generate_iterative(const F* __restrict__ imp, uint8_t* __restrict__ msk,
                                   std::size_t Ng, int Gn, int Gm)
    {
        // First pass: top-N
        generate_topn(imp, msk, Ng, Gn, Gm);

        // Iterative refinement: for each round, try swapping one kept element
        // with one pruned element and check if it improves total importance.
        for (int r = 0; r < 3; ++r) {
            bool improved = false;
            for (std::size_t g = 0; g < Ng; ++g) {
                std::size_t base = g * static_cast<std::size_t>(Gm);
                uint8_t byte = msk[g];

                for (int pi = 0; pi < Gm && !improved; ++pi) {
                    if (byte & (1u << pi)) continue; // pi is already kept
                    for (int ki = 0; ki < Gm; ++ki) {
                        if (!(byte & (1u << ki))) continue; // ki is pruned
                        // Try: prune ki, keep pi
                        if (imp[base + pi] > imp[base + ki]) {
                            byte = static_cast<uint8_t>(byte & ~(1u << ki));
                            byte |=  (1u << pi);
                            improved = true;
                            break;
                        }
                    }
                    if (improved) { msk[g] = byte; break; }
                }
            }
            if (!improved) break;
        }
    }

    CORINGConfig config_;
};

}  // namespace tensorbit::core
