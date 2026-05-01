/// @file kernels_stubs.cpp
/// @brief No-op stubs for CUDA kernel launch functions.
///
/// Compiled only when TENSORBIT_ENABLE_CUDA is OFF. Provides empty
/// implementations so the linker can resolve symbols from coring.cpp
/// and ehap.cpp without CUDA being present. The runtime paths that
/// call these functions are guarded by `tensor.device() == DeviceLocation::kDevice`,
/// which can never be true without CUDA, so these stubs are dead code
/// at runtime.
///
/// @ingroup tensorbit-core

#ifndef TENSORBIT_ENABLE_CUDA

#include "tensorbit/core/kernels.hpp"

namespace tensorbit::core::kernels {

void launch_fisher_diagonal(const float* /*gradients*/,
                            float*       /*fisher_diag*/,
                            std::size_t  /*B*/,
                            std::size_t  /*N*/,
                            void*        /*stream*/) {}

void launch_fisher_accumulate(float*       /*fisher_diag*/,
                              const float* /*gradients*/,
                              std::size_t  /*N*/,
                              float        /*alpha*/,
                              void*        /*stream*/) {}

void launch_ehap_importance(const float* /*weights*/,
                            const float* /*fisher_diag*/,
                            float*       /*importance*/,
                            std::size_t  /*N*/,
                            float        /*damping*/,
                            void*        /*stream*/) {}

void launch_nm_mask_2_4(const float* /*importance*/,
                        uint8_t*     /*mask_out*/,
                        std::size_t  /*N*/,
                        void*        /*stream*/) {}

void launch_nm_mask_generic(const float* /*importance*/,
                            uint8_t*     /*mask_out*/,
                            std::size_t  /*N_elements*/,
                            int          /*group_n*/,
                            int          /*group_m*/,
                            void*        /*stream*/) {}

void launch_apply_mask(float*              /*weights*/,
                       const uint8_t*     /*mask*/,
                       std::size_t         /*N*/,
                       int                 /*group_m*/,
                       void*              /*stream*/) {}

}  // namespace tensorbit::core::kernels

#endif  // TENSORBIT_ENABLE_CUDA
