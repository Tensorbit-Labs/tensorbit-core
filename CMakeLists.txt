cmake_minimum_required(VERSION 3.22)

# Suppress CMP0144 warning (EIGEN3_ROOT vs Eigen3_ROOT)
# Guarded: CMP0144 only exists in CMake >= 3.27
if(POLICY CMP0144)
    cmake_policy(SET CMP0144 OLD)
endif()

# -----------------------------------------------------------------------------
# Project Identity
# -----------------------------------------------------------------------------
# CUDA is opt-in: set TENSORBIT_ENABLE_CUDA=OFF for CPU-only host builds
# (e.g., laptops without NVIDIA GPUs, CI runners, WSL without GPU passthrough).
option(TENSORBIT_ENABLE_CUDA "Build with CUDA 12 GPU acceleration" ON)

if(TENSORBIT_ENABLE_CUDA)
    project(tensorbit-core
        VERSION 0.1.0
        DESCRIPTION "High-performance C++20/CUDA backbone for Hessian-aware structured pruning"
        LANGUAGES CXX CUDA
    )
else()
    project(tensorbit-core
        VERSION 0.1.0
        DESCRIPTION "High-performance C++20/CUDA backbone for Hessian-aware structured pruning"
        LANGUAGES CXX
    )
endif()

# -----------------------------------------------------------------------------
# C++20 Standard Configuration
# -----------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# -----------------------------------------------------------------------------
# CUDA Configuration (only when enabled)
# -----------------------------------------------------------------------------
if(TENSORBIT_ENABLE_CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

    # Target NVIDIA A100 (SM80) and H100 (SM90)
    set(CMAKE_CUDA_ARCHITECTURES "80;90")

    # Required: CUDA Toolkit >= 12.0
    find_package(CUDAToolkit 12.0 REQUIRED)
endif()

# -----------------------------------------------------------------------------
# External Dependencies
# -----------------------------------------------------------------------------
# Eigen3 — header-only linear algebra.
# On Ubuntu/WSL:  sudo apt install libeigen3-dev
# On Windows:     git clone https://gitlab.com/libeigen/eigen.git D:/eigen3
# Custom path:    cmake .. -DEIGEN3_ROOT=/path/to/eigen3
#
# Eigen3 ships Eigen3Config.cmake only on recent versions.
# For older installs or custom paths, we fall back to find_path.
find_package(Eigen3 QUIET)
if(NOT Eigen3_FOUND)
    find_path(EIGEN3_INCLUDE_DIR Eigen/Core
        HINTS ${EIGEN3_ROOT}
        PATHS
            /usr/include/eigen3
            /usr/local/include/eigen3
            /usr/include
            /usr/local/include
            D:/eigen3
    )
    if(EIGEN3_INCLUDE_DIR)
        add_library(Eigen3::Eigen INTERFACE IMPORTED)
        set_target_properties(Eigen3::Eigen PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${EIGEN3_INCLUDE_DIR}")
        message(STATUS "Eigen3 found (fallback): ${EIGEN3_INCLUDE_DIR}")
    else()
        message(FATAL_ERROR
            "Eigen3 not found.\n"
            "  Ubuntu/WSL:  sudo apt install libeigen3-dev\n"
            "  Windows:     git clone https://gitlab.com/libeigen/eigen.git D:/eigen3\n"
            "  Custom path: cmake .. -DEIGEN3_ROOT=/path/to/eigen3")
    endif()
else()
    message(STATUS "Eigen3 found: ${EIGEN3_INCLUDE_DIR}")
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
    src/serialization.cpp
)

set(CORE_CUDA_SOURCES
    src/kernels.cu
)

# -----------------------------------------------------------------------------
# Library Target: tensorbit-core (host C++ logic)
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

# When CUDA is enabled, define TENSORBIT_ENABLE_CUDA so that tensor.hpp
# conditionally includes <cuda_runtime.h> and activates CUDA_CHECK macros.
# When CUDA is disabled, kernel launch stubs are compiled instead.
if(TENSORBIT_ENABLE_CUDA)
    target_compile_definitions(tensorbit-core PUBLIC TENSORBIT_ENABLE_CUDA)
    target_link_libraries(tensorbit-core PUBLIC CUDA::cudart)
else()
    target_sources(tensorbit-core PRIVATE src/kernels_stubs.cpp)
endif()

# -----------------------------------------------------------------------------
# Library Target: tensorbit-core-cuda (CUDA kernels) — only when CUDA enabled
# -----------------------------------------------------------------------------
if(TENSORBIT_ENABLE_CUDA)
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
endif()

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

if(TENSORBIT_ENABLE_CUDA)
    target_link_libraries(tb-prune
        PRIVATE
            tensorbit-core
            tensorbit-core-cuda
    )
else()
    target_link_libraries(tb-prune
        PRIVATE
            tensorbit-core
    )
endif()

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

if(TENSORBIT_ENABLE_CUDA)
    set(TB_INSTALL_TARGETS tensorbit-core tensorbit-core-cuda)
else()
    set(TB_INSTALL_TARGETS tensorbit-core)
endif()

install(TARGETS ${TB_INSTALL_TARGETS}
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
