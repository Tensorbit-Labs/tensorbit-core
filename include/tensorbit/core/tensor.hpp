#pragma once

/// @file tensor.hpp
/// @brief Tensor abstraction with C++20 concepts for type-safe EHAP/CORING operations.
/// @ingroup tensorbit-core

#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <span>
#include <type_traits>
#include <utility>

#include "tensorbit/core/common.hpp"

// ---------------------------------------------------------------------------
// Conditional CUDA Support
// ---------------------------------------------------------------------------

#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

// ---------------------------------------------------------------------------
// Core Concepts
// ---------------------------------------------------------------------------

namespace tensorbit::core {

/// @brief Constrains template parameters to IEEE-754 floating-point types.
template<typename T>
concept FloatingPoint = std::is_floating_point_v<T>;

/// @brief An owning dense multi-dimensional array with a contiguous memory layout.
///
/// Satisfies this concept if it exposes:
/// - `value_type`
/// - `.data()` returning a raw pointer
/// - `.size()` returning total element count
/// - `.shape()` returning a span of per-dimension sizes
/// - `.rank()` returning the number of dimensions
template<typename T>
concept TensorType = requires(T t) {
    typename T::value_type;
    { t.data() }  -> std::same_as<typename T::value_type*>;
    { t.size() }  -> std::convertible_to<std::size_t>;
    { t.shape() } -> std::convertible_to<std::span<const std::size_t>>;
    { t.rank() }  -> std::convertible_to<std::size_t>;
};

// ---------------------------------------------------------------------------
// Device Location
// ---------------------------------------------------------------------------

/// @brief Target device location for a tensor.
enum class DeviceLocation : uint8_t {
    kHost   = 0,
    kDevice = 1,
};

// ---------------------------------------------------------------------------
// Tensor Dense
// ---------------------------------------------------------------------------

/// @brief A dense, row-major tensor owned on a single device (CPU or CUDA GPU).
///
/// @tparam F A FloatingPoint type (float, double).
///
/// This class serves as the primary data container for all pruning operations.
/// GPU-resident tensors manage their own CUDA allocation via a custom deleter.
/// The class satisfies `TensorType<F>` and can be used directly with EHAPPruner
/// and CORINGPruner templates.
template<FloatingPoint F>
class TensorDense {
public:
    using value_type = F;

    /// @name Construction and Destruction
    /// @{

    /// @brief Creates a tensor of the given shape, allocated on the specified device.
    /// @param dims Dimension sizes ordered outermost to innermost (row-major).
    /// @param dev  Target device location.
    explicit TensorDense(std::span<const std::size_t> dims,
                         DeviceLocation dev = DeviceLocation::kHost)
        : device_(dev) {
        rank_ = dims.size();
        TENSORBIT_CHECK(rank_ <= kMaxRank, "Tensor rank exceeds maximum supported dimensions");
        size_ = 1;
        for (std::size_t d = 0; d < rank_; ++d) {
            shape_[d] = dims[d];
            size_ *= dims[d];
        }
        if (size_ == 0) return;

        if (dev == DeviceLocation::kDevice) {
#ifdef __CUDACC__
            F* raw_ptr = nullptr;
            CUDA_CHECK(cudaMalloc(&raw_ptr, size_ * sizeof(F)));
            data_ = std::unique_ptr<F[], Deleter>(raw_ptr, default_device_deleter);
#else
            TENSORBIT_CHECK(false, "CUDA not available — cannot allocate device tensor");
#endif
        } else {
            F* raw_ptr = new F[size_]();
            data_ = std::unique_ptr<F[], Deleter>(raw_ptr, default_host_deleter);
        }
    }

    /// @brief Creates a tensor from an existing host buffer (takes ownership).
    /// @param data Raw pointer to contiguous F data. Ownership is transferred; the
    ///             pointer must have been allocated with `new F[]`.
    /// @param dims Dimension sizes.
    TensorDense(F* data, std::span<const std::size_t> dims)
        : device_(DeviceLocation::kHost) {
        rank_ = dims.size();
        TENSORBIT_CHECK(rank_ <= kMaxRank, "Tensor rank exceeds maximum supported dimensions");
        size_ = 1;
        for (std::size_t d = 0; d < rank_; ++d) {
            shape_[d] = dims[d];
            size_ *= dims[d];
        }
        data_ = std::unique_ptr<F[], Deleter>(data, default_host_deleter);
    }

    TensorDense()                                                 = default;
    ~TensorDense()                                                = default;
    TensorDense(const TensorDense&)                               = delete;
    TensorDense& operator=(const TensorDense&)                    = delete;
    TensorDense(TensorDense&& other) noexcept
        : data_(std::move(other.data_))
        , shape_(other.shape_)
        , size_(other.size_)
        , rank_(other.rank_)
        , device_(other.device_) {
        other.size_   = 0;
        other.rank_   = 0;
        other.device_ = DeviceLocation::kHost;
    }
    TensorDense& operator=(TensorDense&& other) noexcept {
        if (this != &other) {
            data_   = std::move(other.data_);
            shape_  = other.shape_;
            size_   = other.size_;
            rank_   = other.rank_;
            device_ = other.device_;
            other.size_   = 0;
            other.rank_   = 0;
            other.device_ = DeviceLocation::kHost;
        }
        return *this;
    }
    /// @}

    /// @name Data Access
    /// @{

    /// @brief Returns a pointer to the raw data buffer (host or device).
    [[nodiscard]] F* data() noexcept { return data_.get(); }

    /// @brief Returns a const pointer to the raw data buffer.
    [[nodiscard]] const F* data() const noexcept { return data_.get(); }

    /// @brief Returns the total number of elements in the tensor.
    [[nodiscard]] std::size_t size() const noexcept { return size_; }

    /// @brief Returns the shape (per-dimension sizes) as a span.
    [[nodiscard]] std::span<const std::size_t> shape() const noexcept {
        return std::span<const std::size_t>(shape_.data(), rank_);
    }

    /// @brief Returns the number of dimensions (rank).
    [[nodiscard]] std::size_t rank() const noexcept { return rank_; }

    /// @brief Returns the device location.
    [[nodiscard]] DeviceLocation device() const noexcept { return device_; }

    /// @brief Returns the size in bytes of the raw buffer.
    [[nodiscard]] std::size_t bytes() const noexcept { return size_ * sizeof(F); }
    /// @}

    /// @name Element Access (Host-Only)
    /// @{

    /// @brief Element access (no bounds check). Valid only on host-resident tensors.
    [[nodiscard]] F& operator[](std::size_t idx) { return data_.get()[idx]; }

    /// @brief Const element access.
    [[nodiscard]] const F& operator[](std::size_t idx) const { return data_.get()[idx]; }

    /// @brief Returns true if the tensor owns no data.
    [[nodiscard]] bool empty() const noexcept { return size_ == 0; }
    /// @}

    /// @name Device Transfer
    /// @{

    /// @brief Allocates and copies this (host) tensor to the GPU.
    /// @return A new TensorDense resident on the device.
    /// @note Only valid when this tensor is on the host.
    [[nodiscard]] TensorDense to_device() const {
#ifdef __CUDACC__
        TENSORBIT_CHECK(device_ == DeviceLocation::kHost,
                         "to_device() requires a host-resident source tensor");
        TensorDense result(shape(), DeviceLocation::kDevice);
        if (size_ > 0) {
            CUDA_CHECK(cudaMemcpy(result.data(), data_.get(),
                                  size_ * sizeof(F), cudaMemcpyHostToDevice));
        }
        return result;
#else
        TENSORBIT_CHECK(false, "CUDA not available — cannot transfer to device");
        return {};
#endif
    }

    /// @brief Allocates and copies this (device) tensor to the host.
    /// @return A new TensorDense resident on the host.
    /// @note Only valid when this tensor is on the device.
    [[nodiscard]] TensorDense to_host() const {
#ifdef __CUDACC__
        TENSORBIT_CHECK(device_ == DeviceLocation::kDevice,
                         "to_host() requires a device-resident source tensor");
        TensorDense result(shape(), DeviceLocation::kHost);
        if (size_ > 0) {
            CUDA_CHECK(cudaMemcpy(result.data(), data_.get(),
                                  size_ * sizeof(F), cudaMemcpyDeviceToHost));
        }
        return result;
#else
        TENSORBIT_CHECK(false, "CUDA not available — cannot transfer from device");
        return {};
#endif
    }
    /// @}

private:
    static constexpr std::size_t kMaxRank = 8;
    using Deleter = void (*)(F*);

    /// @brief Deallocates host memory allocated with `new F[]`.
    static void default_host_deleter(F* ptr) {
        delete[] ptr;
    }

    /// @brief Deallocates device memory allocated with `cudaMalloc`.
    static void default_device_deleter(F* ptr) {
#ifdef __CUDACC__
        cudaFree(ptr);
#else
        (void)ptr;
#endif
    }

    std::unique_ptr<F[], Deleter> data_{nullptr, nullptr};
    std::array<std::size_t, 8>    shape_{};
    std::size_t                    size_{0};
    std::size_t                    rank_{0};
    DeviceLocation                 device_{DeviceLocation::kHost};
};

// =========================================================================
// Deduction Guide
// =========================================================================

template<FloatingPoint F>
TensorDense(F*, std::span<const std::size_t>) -> TensorDense<F>;

}  // namespace tensorbit::core
