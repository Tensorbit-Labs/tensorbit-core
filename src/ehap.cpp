/// @file ehap.cpp
/// @brief Explicit template instantiations for EHAPPruner.
/// @ingroup tensorbit-core

#include "tensorbit/core/ehap.hpp"

namespace tensorbit::core {

template class EHAPPruner<float>;
template class EHAPPruner<double>;

}  // namespace tensorbit::core
