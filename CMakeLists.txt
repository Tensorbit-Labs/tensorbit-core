cmake_minimum_required(VERSION 3.22)
project(tensorbit-core
    VERSION 0.1.0
    DESCRIPTION "High-performance C++20/CUDA backbone for Hessian-aware structured pruning"
    LANGUAGES CXX CUDA
)

# -----------------------------------------------------------------------------
# C++20 Standard Configuration
# -----------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# -----------------------------------------------------------------------------
# CUDA 12 Configuration
# -----------------------------------------------------------------------------
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# Target NVIDIA A100 (SM80) and H100 (SM90)
set(CMAKE_CUDA_ARCHITECTURES "80;90")

# Required: CUDA Toolkit >= 12.0
find_package(CUDAToolkit 12.0 REQUIRED)

# -----------------------------------------------------------------------------
# External Dependencies
# -----------------------------------------------------------------------------
# Eigen3 — header-only linear algebra
# Provide -DEIGEN3_ROOT=D:/eigen3 if not on a standard path
find_package(Eigen3 REQUIRED NO_MODULE)
if(NOT Eigen3_FOUND)
    message(FATAL_ERROR "Eigen3 not found. Install it with: vcpkg install eigen3")
endif()

# -----------------------------------------------------------------------------
# Compiler Flags
# -----------------------------------------------------------------------------
if(MSVC)
    add_compile_options(/W4 /utf-8 /permissive- /Zc:__cplusplus)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
else()
    add_compile_options(-Wall -Wextra -Wpedantic -Wconversion -Wshadow
                        -Wnon-virtual-dtor -Wold-style-cast -Woverloaded-virtual)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 12)
        add_compile_options(-Wno-psabi)
    endif()
endif()

# -----------------------------------------------------------------------------
# Source Groups
# -----------------------------------------------------------------------------
set(CORE_HEADERS
    include/tensorbit/core/common.hpp
    include/tensorbit/core/tensor.hpp
    include/tensorbit/core/ehap.hpp
    include/tensorbit/core/coring.hpp
    include/tensorbit/core/kernels.hpp
    include/tensorbit/core/serialization.hpp
)

set(CORE_SOURCES
    src/ehap.cpp
    src/coring.cpp
)

set(CORE_CUDA_SOURCES
    src/kernels.cu
)

# -----------------------------------------------------------------------------
# Library Target: tensorbit-core (shared C++ logic)
# -----------------------------------------------------------------------------
add_library(tensorbit-core STATIC
    ${CORE_HEADERS}
    ${CORE_SOURCES}
)

target_include_directories(tensorbit-core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(tensorbit-core
    PUBLIC
        Eigen3::Eigen
)

# -----------------------------------------------------------------------------
# Library Target: tensorbit-core-cuda (CUDA kernels)
# -----------------------------------------------------------------------------
add_library(tensorbit-core-cuda STATIC
    ${CORE_CUDA_SOURCES}
)

target_include_directories(tensorbit-core-cuda
    PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_link_libraries(tensorbit-core-cuda
    PUBLIC
        CUDA::cudart
        CUDA::cublas
        tensorbit-core
)

set_target_properties(tensorbit-core-cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# -----------------------------------------------------------------------------
# Executable Target: tb-prune (CLI tool)
# -----------------------------------------------------------------------------
add_executable(tb-prune
    src/main.cpp
)

target_include_directories(tb-prune
    PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_link_libraries(tb-prune
    PRIVATE
        tensorbit-core
        tensorbit-core-cuda
)

set_target_properties(tb-prune PROPERTIES
    OUTPUT_NAME "tb-prune"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
option(TENSORBIT_BUILD_TESTS "Build unit tests" ON)

if(TENSORBIT_BUILD_TESTS)
    enable_testing()

    # test_ehap
    add_executable(test_ehap
        tests/test_ehap.cpp
        src/ehap.cpp
    )
    target_include_directories(test_ehap
        PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    )
    target_link_libraries(test_ehap
        PRIVATE tensorbit-core
    )
    add_test(NAME test_ehap COMMAND test_ehap)
    set_target_properties(test_ehap PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests"
    )

    # test_coring
    add_executable(test_coring
        tests/test_coring.cpp
        src/coring.cpp
    )
    target_include_directories(test_coring
        PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    )
    target_link_libraries(test_coring
        PRIVATE tensorbit-core
    )
    add_test(NAME test_coring COMMAND test_coring)
    set_target_properties(test_coring PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests"
    )
endif()

# -----------------------------------------------------------------------------
# Install Rules
# -----------------------------------------------------------------------------
include(GNUInstallDirs)

install(TARGETS tensorbit-core tensorbit-core-cuda
    EXPORT tensorbit-core-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(TARGETS tb-prune
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.hpp"
)
