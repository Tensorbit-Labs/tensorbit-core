/// @file coring.cpp
/// @brief CORING N:M Structured Sparsity — full implementation.
/// @ingroup tensorbit-core

#include "tensorbit/core/coring.hpp"

#include <algorithm>
#include <bit>
#include <vector>

#include "tensorbit/core/common.hpp"
#include "tensorbit/core/kernels.hpp"

namespace tensorbit::core {

// ===========================================================================
// Helper: Analytical Pruned Count
// ===========================================================================

/// @brief Computes the number of weights pruned by an N:M mask.
/// @f[
///   N_{\text{pruned}} = N_{\text{elements}} \cdot \frac{M - N}{M}
/// @f]
///
/// This is exact for tensors whose size is divisible by M (enforced by
/// validate_config). Each of the `N_elements / M` groups has exactly
/// `M - N` elements whose mask bits are 0.
static std::size_t compute_pruned_count(std::size_t N_elements, int group_n, int group_m) {
    std::size_t num_groups = N_elements / static_cast<std::size_t>(group_m);
    return num_groups * static_cast<std::size_t>(group_m - group_n);
}

// ===========================================================================
// CORINGPruner<float>
// ===========================================================================

template<>
CORINGPruner<float>::CORINGPruner(CORINGConfig config)
    : config_(config) {
    TENSORBIT_LOG_DEBUG("CORINGPruner<float> constructed (N={}, M={}, CUDA={})",
                        config_.N, config_.M, config_.use_cuda ? "on" : "off");
}

// ---------------------------------------------------------------------------
// validate_config
// ---------------------------------------------------------------------------

template<>
auto CORINGPruner<float>::validate_config(std::size_t num_elements)
    -> std::expected<void, CORINGError> {
    if (config_.N <= 0 || config_.M <= 0) {
        TENSORBIT_LOG_ERROR("CORINGPruner: invalid N:M pair (N={}, M={})",
                            config_.N, config_.M);
        return std::unexpected(CORINGError::kInvalidNMConfig);
    }

    if (config_.N >= config_.M) {
        TENSORBIT_LOG_ERROR("CORINGPruner: N ({}) must be strictly less than M ({})",
                            config_.N, config_.M);
        return std::unexpected(CORINGError::kInvalidNMConfig);
    }

    // M must be a power of two for efficient GPU bit-packing.
    // cuSPARSELt and Sparse Tensor Cores require M ∈ {4, 8, 16}.
    // For general use we enforce power-of-two, warning if > 32 (shared-mem limit).
    if (!std::has_single_bit(static_cast<unsigned>(config_.M))) {
        TENSORBIT_LOG_ERROR("CORINGPruner: M ({}) must be a power of two for structured "
                            "sparsity (required by Ampere Sparse Tensor Cores)",
                            config_.M);
        return std::unexpected(CORINGError::kInvalidNMConfig);
    }

    if (config_.M > 32) {
        TENSORBIT_LOG_WARN("CORINGPruner: M ({}) > 32 — generic N:M kernel only supports "
                           "M ≤ 32. Falling back to CPU path.", config_.M);
    }

    if (num_elements % static_cast<std::size_t>(config_.M) != 0) {
        TENSORBIT_LOG_WARN("CORINGPruner: tensor size ({}) is not divisible by M ({}) — "
                           "last incomplete group will be truncated ({} elements ignored)",
                           num_elements, config_.M,
                           num_elements % static_cast<std::size_t>(config_.M));
    }

    return {};
}

// ---------------------------------------------------------------------------
// generate_nm_mask
// ---------------------------------------------------------------------------

template<>
auto CORINGPruner<float>::generate_nm_mask(const TensorDense<float>& importance,
                                           TensorDense<uint8_t>&     out_mask)
    -> std::expected<void, CORINGError> {
    if (importance.empty() || out_mask.empty()) {
        return std::unexpected(CORINGError::kZeroSizeTensor);
    }

    if (importance.size() != out_mask.size()) {
        TENSORBIT_LOG_ERROR("generate_nm_mask: size mismatch (importance={}, mask={})",
                            importance.size(), out_mask.size());
        return std::unexpected(CORINGError::kShapeMismatch);
    }

    auto validation = validate_config(importance.size());
    if (!validation) {
        return std::unexpected(validation.error());
    }

    std::size_t N_elements = importance.size();
    // Truncate to an even multiple of M.
    std::size_t aligned_N = (N_elements / static_cast<std::size_t>(config_.M))
                            * static_cast<std::size_t>(config_.M);

    bool use_cuda = config_.use_cuda && importance.device() == DeviceLocation::kDevice;

    if (use_cuda) {
        // --- GPU Path ---
        // For 2:4 specifically, use the optimized warp-local kernel.
        if (config_.N == 2 && config_.M == 4) {
            kernels::launch_nm_mask_2_4(importance.data(), out_mask.data(),
                                        aligned_N, nullptr);
        } else {
            kernels::launch_nm_mask_generic(importance.data(), out_mask.data(),
                                            aligned_N, config_.N, config_.M, nullptr);
        }
        CUDA_SYNC_CHECK();
    } else {
        // --- CPU Path ---
        // Requires host-resident importance for direct access.
        const TensorDense<float>* imp_ptr = &importance;
        TensorDense<float> host_copy{};

        if (importance.device() == DeviceLocation::kDevice) {
            host_copy = importance.to_host();
            imp_ptr   = &host_copy;
            CUDA_SYNC_CHECK();
        }

        const float* __restrict__ imp_in     = imp_ptr->data();
        uint8_t* __restrict__     mask_out   = out_mask.data();
        int                       group_n    = config_.N;
        int                       group_m    = config_.M;
        std::size_t               num_groups = aligned_N / static_cast<std::size_t>(group_m);

        // Per-group: find top-N indices via partial sort.
        // Use a small array on the heap for M elements at a time.
        std::vector<std::pair<float, int>> group_vals(static_cast<std::size_t>(group_m));

        for (std::size_t g = 0; g < num_groups; ++g) {
            std::size_t base = g * static_cast<std::size_t>(group_m);

            for (int i = 0; i < group_m; ++i) {
                group_vals[static_cast<std::size_t>(i)] = {imp_in[base + static_cast<std::size_t>(i)], i};
            }

            // Partial sort: put the top-N at the front (descending).
            std::nth_element(
                group_vals.begin(),
                group_vals.begin() + group_n,
                group_vals.end(),
                [](const auto& a, const auto& b) { return a.first > b.first; });

            // Emit mask byte: bit i = 1 for top-N indices.
            uint8_t mask_byte = 0;
            for (int k = 0; k < group_n; ++k) {
                int idx = group_vals[static_cast<std::size_t>(k)].second;
                mask_byte |= static_cast<uint8_t>(1u << idx);
            }
            mask_out[g] = mask_byte;
        }
    }

    TENSORBIT_LOG_DEBUG("generate_nm_mask: N_elements={}, aligned={}, N:M={}:{}, CUDA={}",
                        N_elements, aligned_N, config_.N, config_.M,
                        use_cuda ? "on" : "off");

    return {};
}

// ---------------------------------------------------------------------------
// apply_mask
// ---------------------------------------------------------------------------

template<>
auto CORINGPruner<float>::apply_mask(TensorDense<float>&            weights,
                                     const TensorDense<uint8_t>&    mask)
    -> std::expected<std::size_t, CORINGError> {
    if (weights.empty() || mask.empty()) {
        return std::unexpected(CORINGError::kZeroSizeTensor);
    }

    if (weights.size() != mask.size()) {
        return std::unexpected(CORINGError::kShapeMismatch);
    }

    std::size_t N_elements = weights.size();
    std::size_t aligned_N = (N_elements / static_cast<std::size_t>(config_.M))
                            * static_cast<std::size_t>(config_.M);

    bool use_cuda = config_.use_cuda && weights.device() == DeviceLocation::kDevice;

    if (use_cuda) {
        // --- GPU Path ---
        kernels::launch_apply_mask(weights.data(), mask.data(),
                                   aligned_N, config_.M, nullptr);
        CUDA_SYNC_CHECK();
    } else {
        // --- CPU Path ---
        // Mask is packed: 1 byte per group, bit i = 1 means keep weight i.
        // Must work on host-resident data.
        TensorDense<float> host_weights{};
        float* w_ptr = weights.data();

        if (weights.device() == DeviceLocation::kDevice) {
            host_weights = weights.to_host();
            w_ptr        = host_weights.data();
            CUDA_SYNC_CHECK();
        }

        const uint8_t* __restrict__ m_in = mask.data();
        int group_m = config_.M;
        std::size_t num_groups = aligned_N / static_cast<std::size_t>(group_m);

        for (std::size_t g = 0; g < num_groups; ++g) {
            uint8_t mask_byte = m_in[g];
            std::size_t base = g * static_cast<std::size_t>(group_m);

            for (int i = 0; i < group_m; ++i) {
                if (!((mask_byte >> i) & 1u)) {
                    w_ptr[base + static_cast<std::size_t>(i)] = 0.0f;
                }
            }
        }

        // If we had to copy from device, write back.
        if (weights.device() == DeviceLocation::kDevice) {
#ifdef __CUDACC__
            CUDA_CHECK(cudaMemcpy(weights.data(), host_weights.data(),
                                  weights.size() * sizeof(float),
                                  cudaMemcpyHostToDevice));
#endif
        }
    }

    std::size_t pruned = compute_pruned_count(aligned_N, config_.N, config_.M);

    TENSORBIT_LOG_DEBUG("apply_mask: N_elements={}, pruned={}, N:M={}:{}",
                        N_elements, pruned, config_.N, config_.M);

    return pruned;
}

// ---------------------------------------------------------------------------
// prune
// ---------------------------------------------------------------------------

template<>
auto CORINGPruner<float>::prune(const TensorDense<float>& importance,
                                TensorDense<float>&       weights)
    -> std::expected<std::size_t, CORINGError> {
    if (importance.size() != weights.size()) {
        return std::unexpected(CORINGError::kShapeMismatch);
    }

    // Allocate mask on the same device as weights for zero-copy GPU dispatch.
    auto shape = std::span(importance.shape().data(), importance.rank());
    TensorDense<uint8_t> mask(shape, weights.device());

    auto mask_result = generate_nm_mask(importance, mask);
    if (!mask_result) {
        return std::unexpected(mask_result.error());
    }

    return apply_mask(weights, mask);
}

template class CORINGPruner<float>;

// ===========================================================================
// CORINGPruner<double>
// ===========================================================================

template<>
CORINGPruner<double>::CORINGPruner(CORINGConfig config)
    : config_(config) {
    TENSORBIT_LOG_DEBUG("CORINGPruner<double> constructed (N={}, M={}, CUDA={})",
                        config_.N, config_.M, config_.use_cuda ? "on" : "off");
}

// ---------------------------------------------------------------------------
// validate_config (double)
// ---------------------------------------------------------------------------

template<>
auto CORINGPruner<double>::validate_config(std::size_t num_elements)
    -> std::expected<void, CORINGError> {
    if (config_.N <= 0 || config_.M <= 0) {
        return std::unexpected(CORINGError::kInvalidNMConfig);
    }

    if (config_.N >= config_.M) {
        return std::unexpected(CORINGError::kInvalidNMConfig);
    }

    if (!std::has_single_bit(static_cast<unsigned>(config_.M))) {
        TENSORBIT_LOG_ERROR("CORINGPruner: M ({}) must be a power of two", config_.M);
        return std::unexpected(CORINGError::kInvalidNMConfig);
    }

    return {};
}

// ---------------------------------------------------------------------------
// generate_nm_mask (double)
// ---------------------------------------------------------------------------

template<>
auto CORINGPruner<double>::generate_nm_mask(const TensorDense<double>& importance,
                                            TensorDense<uint8_t>&      out_mask)
    -> std::expected<void, CORINGError> {
    if (importance.empty() || out_mask.empty()) {
        return std::unexpected(CORINGError::kZeroSizeTensor);
    }

    if (importance.size() != out_mask.size()) {
        return std::unexpected(CORINGError::kShapeMismatch);
    }

    auto validation = validate_config(importance.size());
    if (!validation) {
        return std::unexpected(validation.error());
    }

    std::size_t N_elements = importance.size();
    std::size_t aligned_N = (N_elements / static_cast<std::size_t>(config_.M))
                            * static_cast<std::size_t>(config_.M);

    // GPU kernels are float-only. Use CPU path for double precision.
    const TensorDense<double>* imp_ptr = &importance;
    TensorDense<double> host_copy{};

    if (importance.device() == DeviceLocation::kDevice) {
        host_copy = importance.to_host();
        imp_ptr   = &host_copy;
        CUDA_SYNC_CHECK();
    }

    const double* __restrict__ imp_in     = imp_ptr->data();
    uint8_t* __restrict__      mask_out   = out_mask.data();
    int                        group_n    = config_.N;
    int                        group_m    = config_.M;
    std::size_t                num_groups = aligned_N / static_cast<std::size_t>(group_m);

    std::vector<std::pair<double, int>> group_vals(static_cast<std::size_t>(group_m));

    for (std::size_t g = 0; g < num_groups; ++g) {
        std::size_t base = g * static_cast<std::size_t>(group_m);

        for (int i = 0; i < group_m; ++i) {
            group_vals[static_cast<std::size_t>(i)] = {
                imp_in[base + static_cast<std::size_t>(i)], i};
        }

        std::nth_element(
            group_vals.begin(),
            group_vals.begin() + group_n,
            group_vals.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; });

        uint8_t mask_byte = 0;
        for (int k = 0; k < group_n; ++k) {
            int idx = group_vals[static_cast<std::size_t>(k)].second;
            mask_byte |= static_cast<uint8_t>(1u << idx);
        }
        mask_out[g] = mask_byte;
    }

    TENSORBIT_LOG_DEBUG("CORINGPruner<double>::generate_nm_mask: N_elements={}, N:M={}:{}",
                        N_elements, config_.N, config_.M);

    return {};
}

// ---------------------------------------------------------------------------
// apply_mask (double)
// ---------------------------------------------------------------------------

template<>
auto CORINGPruner<double>::apply_mask(TensorDense<double>&            weights,
                                      const TensorDense<uint8_t>&    mask)
    -> std::expected<std::size_t, CORINGError> {
    if (weights.empty() || mask.empty()) {
        return std::unexpected(CORINGError::kZeroSizeTensor);
    }

    if (weights.size() != mask.size()) {
        return std::unexpected(CORINGError::kShapeMismatch);
    }

    std::size_t N_elements = weights.size();
    std::size_t aligned_N = (N_elements / static_cast<std::size_t>(config_.M))
                            * static_cast<std::size_t>(config_.M);

    // CPU path for double precision (GPU kernels are float-only).
    TensorDense<double> host_weights{};
    double* w_ptr = weights.data();

    if (weights.device() == DeviceLocation::kDevice) {
        host_weights = weights.to_host();
        w_ptr        = host_weights.data();
        CUDA_SYNC_CHECK();
    }

    const uint8_t* __restrict__ m_in      = mask.data();
    int                         group_m   = config_.M;
    std::size_t                 num_groups = aligned_N / static_cast<std::size_t>(group_m);

    for (std::size_t g = 0; g < num_groups; ++g) {
        uint8_t mask_byte = m_in[g];
        std::size_t base = g * static_cast<std::size_t>(group_m);

        for (int i = 0; i < group_m; ++i) {
            if (!((mask_byte >> i) & 1u)) {
                w_ptr[base + static_cast<std::size_t>(i)] = 0.0;
            }
        }
    }

    if (weights.device() == DeviceLocation::kDevice) {
#ifdef __CUDACC__
        CUDA_CHECK(cudaMemcpy(weights.data(), host_weights.data(),
                              weights.size() * sizeof(double),
                              cudaMemcpyHostToDevice));
#endif
    }

    std::size_t pruned = compute_pruned_count(aligned_N, config_.N, config_.M);

    TENSORBIT_LOG_DEBUG("CORINGPruner<double>::apply_mask: N_elements={}, pruned={}, N:M={}:{}",
                        N_elements, pruned, config_.N, config_.M);

    return pruned;
}

// ---------------------------------------------------------------------------
// prune (double)
// ---------------------------------------------------------------------------

template<>
auto CORINGPruner<double>::prune(const TensorDense<double>& importance,
                                 TensorDense<double>&       weights)
    -> std::expected<std::size_t, CORINGError> {
    if (importance.size() != weights.size()) {
        return std::unexpected(CORINGError::kShapeMismatch);
    }

    auto shape = std::span(importance.shape().data(), importance.rank());
    TensorDense<uint8_t> mask(shape, weights.device());

    auto mask_result = generate_nm_mask(importance, mask);
    if (!mask_result) {
        return std::unexpected(mask_result.error());
    }

    return apply_mask(weights, mask);
}

template class CORINGPruner<double>;

}  // namespace tensorbit::core
