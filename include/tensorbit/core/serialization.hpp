#pragma once

/// @file serialization.hpp
/// @brief Tensorbit Binary (.tb) serialization layer skeleton.
///
/// Provides hooks for reading and writing the proprietary `.tb` file format,
/// which packages pruned weights and N:M masks for fast inference.
///
/// @ingroup tensorbit-core

#include <cstddef>
#include <cstdint>
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

/// Magic bytes at the start of every .tb file: "TB01" in ASCII.
inline constexpr uint32_t kTBMagic = 0x31304254;  // "TB01" big-endian

/// Current .tb format version.
inline constexpr uint32_t kTBVersion = 1;

/// Maximum header size (reserved for metadata).
inline constexpr std::size_t kTBHeaderReserved = 4096;

// ---------------------------------------------------------------------------
// File Header (on-disk layout)
// ---------------------------------------------------------------------------

#pragma pack(push, 1)
/// @brief On-disk header for the .tb binary format.
///
/// Immediately follows the magic bytes. All fields are little-endian.
struct TBHeader {
    uint32_t magic    = kTBMagic;   ///< Magic identifier.
    uint32_t version  = kTBVersion; ///< Format version.
    uint32_t nm_n     = 0;          ///< N in N:M sparsity pattern.
    uint32_t nm_m     = 0;          ///< M in N:M sparsity pattern.
    uint64_t num_weights = 0;       ///< Total number of weight elements.
    uint64_t num_masks   = 0;       ///< Total number of mask bytes (packed bits).
    uint64_t weights_offset = 0;    ///< Byte offset to the start of weight data.
    uint64_t masks_offset   = 0;    ///< Byte offset to the start of mask data.
    uint8_t  precision   = 0;       ///< 0=FP32, 1=FP16, 2=BF16.
    uint8_t  reserved[2047];        ///< Padding for future extensions.
};
#pragma pack(pop)

static_assert(sizeof(TBHeader) == 4096, "TBHeader must be exactly 4096 bytes");

// ---------------------------------------------------------------------------
// Serialization Error Codes
// ---------------------------------------------------------------------------

/// @brief Error codes for .tb serialization operations.
enum class TBError : uint8_t {
    kOk           = 0,
    kFileOpen     = 1,
    kFileWrite    = 2,
    kFileRead     = 3,
    kBadMagic     = 4,
    kVersionMismatch = 5,
    kTruncated    = 6,
};

// ---------------------------------------------------------------------------
// TBWriter — writes .tb files
// ---------------------------------------------------------------------------

/// @class TBWriter
/// @brief Serializes pruned weights and N:M masks into the `.tb` format.
///
/// Usage:
/// @code
///   TBWriter writer;
///   auto result = writer.write("output.tb", pruned_weights, masks, 2, 4);
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
    /// @tparam F Floating-point type of the weights.
    /// @param path        Output file path (e.g., "resnet50_pruned.tb").
    /// @param weights     Pruned weight tensor (host-resident).
    /// @param masks       Packed N:M bitmask (host-resident, uint8).
    /// @param nm_n        N in the N:M pattern.
    /// @param nm_m        M in the N:M pattern.
    /// @return std::expected with success or TBError.
    template<FloatingPoint F>
    auto write(std::string_view              path,
               std::span<const F>            weights,
               std::span<const uint8_t>      masks,
               uint32_t                      nm_n,
               uint32_t                      nm_m)
        -> Result<void, TBError>;

    /// @brief Returns the human-readable error message for the last operation.
    [[nodiscard]] std::string_view last_error() const noexcept { return last_error_; }

private:
    std::string last_error_;
};

// ---------------------------------------------------------------------------
// TBReader — reads .tb files
// ---------------------------------------------------------------------------

/// @class TBReader
/// @brief Deserializes a `.tb` file, extracting weights and masks.
///
/// Intended for the Tensorbit inference runtime. Provides hooks for
/// zero-copy memory mapping in future releases.
class TBReader {
public:
    TBReader()  = default;
    ~TBReader() = default;

    TBReader(const TBReader&)            = delete;
    TBReader& operator=(const TBReader&) = delete;
    TBReader(TBReader&&) noexcept        = default;
    TBReader& operator=(TBReader&&) noexcept = default;

    /// @brief Opens and validates a .tb file, returning metadata.
    ///
    /// This performs header validation (magic & version) and populates
    /// internal metadata. Actual weight/mask data is not loaded until
    /// `read_weights()` or `read_masks()` is called.
    ///
    /// @param path  Path to the .tb file.
    /// @return std::expected with the parsed TBHeader on success, or TBError.
    auto open(std::string_view path) -> Result<TBHeader, TBError>;

    /// @brief Reads weight data from the currently open .tb file.
    ///
    /// @tparam F Expected floating-point type.
    /// @param out_weights Destination buffer (must be pre-sized).
    /// @return std::expected with success or TBError.
    template<FloatingPoint F>
    auto read_weights(std::span<F> out_weights) -> Result<void, TBError>;

    /// @brief Reads mask data from the currently open .tb file.
    ///
    /// @param out_masks Destination buffer (must be pre-sized).
    /// @return std::expected with success or TBError.
    auto read_masks(std::span<uint8_t> out_masks) -> Result<void, TBError>;

    /// @brief Closes the currently open file.
    void close();

private:
    std::ifstream file_;
    TBHeader      header_{};
};

}  // namespace tensorbit::core
