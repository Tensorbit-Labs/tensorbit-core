/// @file coring.cpp
/// @brief Explicit template instantiations for CORINGPruner.
/// @ingroup tensorbit-core

#include "tensorbit/core/coring.hpp"

namespace tensorbit::core {

template class CORINGPruner<float>;
template class CORINGPruner<double>;

}  // namespace tensorbit::core
