/// @file serialization.cpp
/// @brief Explicit template instantiations for TBWriter / TBReader.
/// @ingroup tensorbit-core

#include "tensorbit/core/serialization.hpp"

namespace tensorbit::core {

// Explicit instantiations for TBWriter::write<F>
template auto TBWriter::write<float>(
    std::string_view, std::span<const float>, std::span<const uint8_t>,
    uint32_t, uint32_t) -> Result<void, TBError>;

template auto TBWriter::write<double>(
    std::string_view, std::span<const double>, std::span<const uint8_t>,
    uint32_t, uint32_t) -> Result<void, TBError>;

// Explicit instantiations for TBReader::read_weights<F>
template auto TBReader::read_weights<float>(std::span<float>)
    -> Result<void, TBError>;

template auto TBReader::read_weights<double>(std::span<double>)
    -> Result<void, TBError>;

}  // namespace tensorbit::core
