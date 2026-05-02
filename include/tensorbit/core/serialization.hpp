#pragma once

/// @file serialization.hpp
/// @brief Tensorbit Binary (.tb) serialization — full implementation.
/// @ingroup tensorbit-core
///
/// The .tb format stores pruned weights and N:M masks for fast inference.
/// Layout:
///   [0..4095]     TBHeader (4096 bytes, packed)
///   [4096..W+4095]  Weight data (FP32/FP16/BF16, little-endian)
///   [W+4096..]      Mask data (1 byte per group, bits 0..M-1)

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <span>
#include <string_view>
#include <vector>

#include "tensorbit/core/common.hpp"
#include "tensorbit/core/tensor.hpp"

namespace tensorbit::core {

// ---------------------------------------------------------------------------
// .tb Format Constants
// ---------------------------------------------------------------------------

inline constexpr uint32_t   kTBMagic          = 0x31304254;
inline constexpr uint32_t   kTBVersion        = 1;
inline constexpr std::size_t kTBHeaderSize    = 4096;

// ---------------------------------------------------------------------------
// File Header (on-disk layout)
// ---------------------------------------------------------------------------

#pragma pack(push, 1)
struct TBHeader {
    uint32_t magic            = kTBMagic;
    uint32_t version          = kTBVersion;
    uint32_t nm_n             = 0;
    uint32_t nm_m             = 0;
    uint64_t num_weights      = 0;
    uint64_t num_mask_bytes   = 0;
    uint64_t weights_offset   = kTBHeaderSize;
    uint64_t masks_offset     = kTBHeaderSize;  // set at write-time
    uint8_t  precision        = 0;  // 0=FP32, 1=FP16, 2=BF16
    uint8_t  reserved[4047];        ///< Padding — total struct = 4096 bytes
};
#pragma pack(pop)

static_assert(sizeof(TBHeader) == kTBHeaderSize,
              "TBHeader must be exactly 4096 bytes");

// ---------------------------------------------------------------------------
// Serialization Error Codes
// ---------------------------------------------------------------------------

enum class TBError : uint8_t {
    kOk             = 0,
    kFileOpen       = 1,
    kFileWrite      = 2,
    kFileRead       = 3,
    kBadMagic       = 4,
    kVersionMismatch = 5,
    kTruncated       = 6,
    kSeekFailed     = 7,
};

// ===========================================================================
// TBWriter
// ===========================================================================

/// @brief Serializes pruned weights and N:M masks into the `.tb` format.
///
/// Usage:
/// @code
///   TBWriter writer;
///   auto r = writer.write("output.tb", weight_span, mask_span, 2, 4);
/// @endcode
class TBWriter {
public:
    TBWriter()  = default;
    ~TBWriter() = default;

    TBWriter(const TBWriter&)            = delete;
    TBWriter& operator=(const TBWriter&) = delete;
    TBWriter(TBWriter&&) noexcept        = default;
    TBWriter& operator=(TBWriter&&) noexcept = default;

    /// @brief Writes a .tb file containing pruned weights and N:M masks.
    ///
    /// @tparam F Floating-point type of the weights (float/double).
    /// @param path     Output file path.
    /// @param weights  Pruned weight data (host-resident span).
    /// @param masks    Packed N:M bitmask (1 byte per group, host-resident).
    /// @param nm_n     N in the N:M pattern.
    /// @param nm_m     M in the N:M pattern.
    /// @return Success or TBError.
    template<FloatingPoint F>
    auto write(std::string_view         path,
               std::span<const F>       weights,
               std::span<const uint8_t> masks,
               uint32_t                 nm_n,
               uint32_t                 nm_m) -> Result<void, TBError>
    {
        std::ofstream file(path.data(), std::ios::binary | std::ios::trunc);
        if (!file.is_open())
            return unexpected(TBError::kFileOpen);

        std::size_t num_mask_bytes = masks.size();
        std::size_t weights_byte_size = weights.size() * sizeof(F);

        // Build header
        TBHeader hdr{};
        hdr.magic          = kTBMagic;
        hdr.version        = kTBVersion;
        hdr.nm_n           = nm_n;
        hdr.nm_m           = nm_m;
        hdr.num_weights    = weights.size();
        hdr.num_mask_bytes  = num_mask_bytes;
        hdr.weights_offset  = kTBHeaderSize;
        hdr.masks_offset    = kTBHeaderSize + weights_byte_size;

        // Precision byte
        if constexpr (std::is_same_v<F, float>)
            hdr.precision = 0;
        else if constexpr (std::is_same_v<F, double>)
            hdr.precision = 3;  // FP64 (custom encoding)
        else
            hdr.precision = 0;

        std::memset(hdr.reserved, 0, sizeof(hdr.reserved));

        // Write header (4096 bytes)
        file.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
        if (!file.good()) return unexpected(TBError::kFileWrite);

        // Write weight blob
        file.write(reinterpret_cast<const char*>(weights.data()),
                   static_cast<std::streamsize>(weights_byte_size));
        if (!file.good()) return unexpected(TBError::kFileWrite);

        // Write mask blob
        file.write(reinterpret_cast<const char*>(masks.data()),
                   static_cast<std::streamsize>(num_mask_bytes));
        if (!file.good()) return unexpected(TBError::kFileWrite);

        return {};
    }

private:
    std::string last_error_;
};

// ===========================================================================
// TBReader
// ===========================================================================

/// @brief Deserializes a `.tb` file, extracting weights and masks.
///
/// Provides a `read_mmap()` stub for future zero-copy memory-mapped loading.
class TBReader {
public:
    TBReader()  = default;
    ~TBReader() = default;

    TBReader(const TBReader&)            = delete;
    TBReader& operator=(const TBReader&) = delete;
    TBReader(TBReader&&) noexcept        = default;
    TBReader& operator=(TBReader&&) noexcept = default;

    /// @brief Opens and validates a .tb file, returning the parsed header.
    ///
    /// Performs magic-number and version checks.  Weight/mask data is not
    /// loaded until `read_weights()` or `read_masks()` is called.
    auto open(std::string_view path) -> Result<TBHeader, TBError>
    {
        file_.open(path.data(), std::ios::binary);
        if (!file_.is_open())
            return unexpected(TBError::kFileOpen);

        TBHeader hdr{};
        file_.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));
        if (!file_.good() || file_.gcount() != static_cast<std::streamsize>(sizeof(hdr)))
            return unexpected(TBError::kTruncated);

        if (hdr.magic != kTBMagic)
            return unexpected(TBError::kBadMagic);
        if (hdr.version != kTBVersion)
            return unexpected(TBError::kVersionMismatch);

        header_ = hdr;
        return hdr;
    }

    /// @brief Reads weight data from the currently open .tb file.
    ///
    /// @tparam F Expected floating-point type.
    /// @param out_weights Destination buffer (must be pre-sized to header_.num_weights).
    template<FloatingPoint F>
    auto read_weights(std::span<F> out_weights) -> Result<void, TBError>
    {
        if (!file_.is_open()) return unexpected(TBError::kFileOpen);
        if (out_weights.size() < header_.num_weights)
            return unexpected(TBError::kTruncated);

        file_.seekg(static_cast<std::streamoff>(header_.weights_offset));
        if (!file_.good()) return unexpected(TBError::kSeekFailed);

        file_.read(reinterpret_cast<char*>(out_weights.data()),
                   static_cast<std::streamsize>(header_.num_weights * sizeof(F)));
        if (!file_.good()) return unexpected(TBError::kFileRead);

        return {};
    }

    /// @brief Reads mask data from the currently open .tb file.
    auto read_masks(std::span<uint8_t> out_masks) -> Result<void, TBError>
    {
        if (!file_.is_open()) return unexpected(TBError::kFileOpen);
        if (out_masks.size() < header_.num_mask_bytes)
            return unexpected(TBError::kTruncated);

        file_.seekg(static_cast<std::streamoff>(header_.masks_offset));
        if (!file_.good()) return unexpected(TBError::kSeekFailed);

        file_.read(reinterpret_cast<char*>(out_masks.data()),
                   static_cast<std::streamsize>(header_.num_mask_bytes));
        if (!file_.good()) return unexpected(TBError::kFileRead);

        return {};
    }

    /// @brief Closes the currently open file.
    void close() { file_.close(); }

    /// @brief Returns the parsed header (valid after successful open()).
    [[nodiscard]] const TBHeader& header() const noexcept { return header_; }

private:
    std::ifstream file_;
    TBHeader      header_{};
};

}  // namespace tensorbit::core
