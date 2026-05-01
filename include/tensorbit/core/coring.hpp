#pragma once

/// @file coring.hpp
/// @brief CORING N:M Structured Sparsity pruner — full implementation.
/// @ingroup tensorbit-core

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
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

struct CORINGConfig {
    int  N = 2;
    int  M = 4;
    bool use_cuda = true;
};

template<FloatingPoint F>
class CORINGPruner {
public:
    explicit CORINGPruner(CORINGConfig config) : config_(config) {}

    ~CORINGPruner() = default;

    CORINGPruner(const CORINGPruner&)                = delete;
    CORINGPruner& operator=(const CORINGPruner&)     = delete;
    CORINGPruner(CORINGPruner&&) noexcept            = default;
    CORINGPruner& operator=(CORINGPruner&&) noexcept = default;

    // -----------------------------------------------------------------------
    // generate_nm_mask
    // -----------------------------------------------------------------------
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

        std::size_t Ne = importance.size();
        int Gn = config_.N;
        int Gm = config_.M;
        std::size_t Ng = (Ne / static_cast<std::size_t>(Gm));
        std::size_t an = Ng * static_cast<std::size_t>(Gm);

        bool gpu_ok = config_.use_cuda
                      && importance.device() == DeviceLocation::kDevice;

        if constexpr (std::is_same_v<F, float>) {
            if (gpu_ok) {
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

        const F* __restrict__ imp  = src->data();
        uint8_t* __restrict__  msk  = out_mask.data();
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
        return {};
    }

    // -----------------------------------------------------------------------
    // apply_mask
    // -----------------------------------------------------------------------
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

    // -----------------------------------------------------------------------
    // prune
    // -----------------------------------------------------------------------
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
        return apply_mask(weights, mask);
    }

    [[nodiscard]] const CORINGConfig& config() const noexcept { return config_; }

private:
    auto validate_config(std::size_t num_elements) -> Result<void, CORINGError> {
        if (config_.N <= 0 || config_.M <= 0)
            return unexpected(CORINGError::kInvalidNMConfig);
        if (config_.N >= config_.M)
            return unexpected(CORINGError::kInvalidNMConfig);
        if (!std::has_single_bit(static_cast<unsigned>(config_.M)))
            return unexpected(CORINGError::kInvalidNMConfig);
        if (num_elements > 0 &&
            num_elements % static_cast<std::size_t>(config_.M) != 0) {
            return unexpected(CORINGError::kShapeMismatch);
        }
        return {};
    }

    CORINGConfig config_;
};

}  // namespace tensorbit::core
