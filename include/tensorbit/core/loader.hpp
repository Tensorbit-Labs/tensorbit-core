#pragma once

/// @file loader.hpp
/// @brief Lightweight .safetensors metadata parser.
/// @ingroup tensorbit-core
///
/// Parses the JSON header of a .safetensors file to extract tensor names,
/// shapes, data types, and byte offsets.  Does not depend on any external
/// JSON library — uses a minimal hand-rolled parser for the subset of
/// JSON produced by HuggingFace safetensors exporters.
///
/// ## .safetensors Format
///   [0..7]      uint64_t N = length of JSON header (little-endian)
///   [8..8+N-1]  UTF-8 JSON metadata
///   [8+N..]     raw tensor data (contiguous, dtype'd)
///
/// Example JSON (simplified):
/// @code
/// {
///   "model.layers.0.weight": {
///     "dtype": "F32",
///     "shape": [4096, 4096],
///     "data_offsets": [0, 67108864]
///   },
///   ...
/// }
/// @endcode

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <string_view>
#include <vector>

#include "tensorbit/core/common.hpp"
#include "tensorbit/core/tensor.hpp"

namespace tensorbit::core::loader {

/// @brief Error codes for safetensors loading.
enum class LoaderError : uint8_t {
    kOk             = 0,
    kFileOpen       = 1,
    kBadHeader      = 2,
    kBadJSON        = 3,
    kUnsupportedDtype = 4,
    kNotFound       = 5,
};

/// @brief Supported data types.
enum class STDtype : uint8_t {
    kF32 = 0,
    kF16 = 1,
    kBF16 = 2,
    kI64 = 3,
    kUnknown = 255,
};

/// @brief Returns element size in bytes for a safetensors dtype.
inline std::size_t dtype_size(STDtype dt) {
    switch (dt) {
        case STDtype::kF32:  return 4;
        case STDtype::kF16:  return 2;
        case STDtype::kBF16: return 2;
        case STDtype::kI64:  return 8;
        default:             return 0;
    }
}

/// @brief Convert safetensors dtype string to enum.
inline STDtype parse_dtype(std::string_view s) {
    if (s == "F32" || s == "float32" || s == "\"F32\"")  return STDtype::kF32;
    if (s == "F16" || s == "float16" || s == "\"F16\"")  return STDtype::kF16;
    if (s == "BF16" || s == "bfloat16" || s == "\"BF16\"") return STDtype::kBF16;
    if (s == "I64" || s == "int64" || s == "\"I64\"")    return STDtype::kI64;
    return STDtype::kUnknown;
}

/// @brief Metadata for a single tensor in a .safetensors file.
struct TensorMeta {
    std::string name;
    STDtype      dtype = STDtype::kUnknown;
    std::vector<std::size_t> shape;
    std::size_t  offset = 0;   // byte offset from start of data section
    std::size_t  length = 0;   // byte length
    std::size_t  numel = 0;    // total elements
};

/// @brief Lightweight .safetensors metadata reader.
///
/// Opens a .safetensors file, parses the JSON header, and provides
/// access to tensor metadata (names, shapes, dtypes, offsets).
/// Actual tensor data can be read via `read_tensor_data()`.
class SafeTensorsFile {
public:
    SafeTensorsFile() = default;
    ~SafeTensorsFile() { close(); }

    SafeTensorsFile(const SafeTensorsFile&) = delete;
    SafeTensorsFile& operator=(const SafeTensorsFile&) = delete;

    /// @brief Opens and parses the metadata of a .safetensors file.
    auto open(std::string_view path) -> Result<void, LoaderError>
    {
        file_.open(path.data(), std::ios::binary);
        if (!file_.is_open())
            return unexpected(LoaderError::kFileOpen);

        // Read 8-byte header length (little-endian uint64)
        uint64_t header_len = 0;
        file_.read(reinterpret_cast<char*>(&header_len), 8);
        if (!file_.good() || header_len == 0 || header_len > 100'000'000ULL)
            return unexpected(LoaderError::kBadHeader);

        // Read JSON header
        std::string json_str(header_len, '\0');
        file_.read(json_str.data(), static_cast<std::streamsize>(header_len));
        if (!file_.good())
            return unexpected(LoaderError::kBadHeader);

        data_offset_ = 8 + header_len;
        auto r = parse_json(json_str);
        if (!r) return unexpected(r.error());
        return {};
    }

    /// @brief Returns metadata for all tensors in the file.
    [[nodiscard]] const std::vector<TensorMeta>& tensors() const noexcept {
        return tensors_;
    }

    /// @brief Finds a tensor by name.
    [[nodiscard]] const TensorMeta* find(std::string_view name) const noexcept {
        for (auto& t : tensors_)
            if (t.name == name) return &t;
        return nullptr;
    }

    /// @brief Reads raw tensor data into a pre-allocated buffer.
    auto read_tensor_data(const TensorMeta& meta, std::span<std::byte> buffer)
        -> Result<void, LoaderError>
    {
        if (!file_.is_open()) return unexpected(LoaderError::kFileOpen);
        if (buffer.size() < meta.length)
            return unexpected(LoaderError::kBadHeader);

        file_.seekg(static_cast<std::streamoff>(data_offset_ + meta.offset));
        if (!file_.good())
            return unexpected(LoaderError::kBadHeader);

        file_.read(reinterpret_cast<char*>(buffer.data()),
                   static_cast<std::streamsize>(meta.length));
        if (!file_.good())
            return unexpected(LoaderError::kBadHeader);

        return {};
    }

    void close() { file_.close(); }

    [[nodiscard]] std::size_t data_offset() const noexcept { return data_offset_; }

private:
    /// Minimal JSON parser — extracts only the fields we need from
    /// the flat HuggingFace safetensors JSON format.
    auto parse_json(const std::string& json)
        -> Result<void, LoaderError>
    {
        tensors_.clear();
        const char* p = json.c_str();
        const char* end = p + json.size();

        auto skip_ws = [&]() {
            while (p < end && (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t'))
                ++p;
        };

        auto read_string = [&]() -> std::string {
            skip_ws();
            if (p >= end || *p != '"') return {};
            ++p;
            std::string s;
            while (p < end && *p != '"') {
                if (*p == '\\' && p + 1 < end) ++p;
                s += *p++;
            }
            if (p < end) ++p; // closing quote
            return s;
        };

        auto read_integer = [&]() -> uint64_t {
            skip_ws();
            uint64_t val = 0;
            while (p < end && *p >= '0' && *p <= '9')
                val = val * 10 + static_cast<uint64_t>(*p++ - '0');
            return val;
        };

        // Skip opening brace
        skip_ws();
        if (p >= end || *p != '{') return unexpected(LoaderError::kBadJSON);
        ++p;

        // Parse each key-value pair
        while (p < end) {
            skip_ws();
            if (p >= end) break;
            if (*p == '}') break;
            if (*p == ',') { ++p; continue; }

            std::string tensor_name = read_string();
            if (tensor_name.empty()) break;

            skip_ws();
            if (p >= end || *p != ':') return unexpected(LoaderError::kBadJSON);
            ++p; // skip ':'

            // Parse tensor metadata object: {"dtype":"F32","shape":[X,Y],"data_offsets":[S,E]}
            skip_ws();
            if (p >= end || *p != '{') return unexpected(LoaderError::kBadJSON);
            ++p;

            TensorMeta meta;
            meta.name = tensor_name;

            while (p < end) {
                skip_ws();
                if (p >= end) break;
                if (*p == '}') { ++p; break; }
                if (*p == ',') { ++p; continue; }

                std::string key = read_string();
                skip_ws();
                if (p >= end || *p != ':') return unexpected(LoaderError::kBadJSON);
                ++p;

                if (key == "dtype") {
                    std::string dt = read_string();
                    meta.dtype = parse_dtype(dt);
                } else if (key == "shape") {
                    skip_ws();
                    if (p >= end || *p != '[')
                        return unexpected(LoaderError::kBadJSON);
                    ++p;
                    while (p < end) {
                        skip_ws();
                        if (*p == ']') { ++p; break; }
                        if (*p == ',') { ++p; continue; }
                        uint64_t dim = read_integer();
                        meta.shape.push_back(static_cast<std::size_t>(dim));
                    }
                } else if (key == "data_offsets") {
                    skip_ws();
                    if (p >= end || *p != '[')
                        return unexpected(LoaderError::kBadJSON);
                    ++p;
                    uint64_t start = read_integer();
                    skip_ws(); if (p < end && *p == ',') ++p;
                    uint64_t end_off = read_integer();
                    skip_ws(); if (p < end && *p == ']') ++p;
                    meta.offset = static_cast<std::size_t>(start);
                    meta.length = static_cast<std::size_t>(end_off - start);
                } else {
                    // Skip unknown keys by advancing past their value
                    skip_ws();
                    if (p < end && *p == '"') { read_string(); }
                    else if (p < end && *p == '[') {
                        ++p; int depth = 1;
                        while (p < end && depth > 0) {
                            if (*p == '[') ++depth;
                            if (*p == ']') --depth;
                            ++p;
                        }
                    } else if (p < end && *p == '{') {
                        ++p; int depth = 1;
                        while (p < end && depth > 0) {
                            if (*p == '{') ++depth;
                            if (*p == '}') --depth;
                            ++p;
                        }
                    } else {
                        while (p < end && *p != ',' && *p != '}' && *p != '\n') ++p;
                    }
                }
            }

            if (meta.shape.empty()) meta.numel = 0;
            else {
                meta.numel = 1;
                for (auto d : meta.shape) meta.numel *= d;
            }
            if (meta.length == 0) meta.length = meta.numel * dtype_size(meta.dtype);
            tensors_.push_back(std::move(meta));
        }

        return {};
    }

    std::ifstream file_;
    std::size_t   data_offset_ = 0;
    std::vector<TensorMeta> tensors_;
};

}  // namespace tensorbit::core::loader
