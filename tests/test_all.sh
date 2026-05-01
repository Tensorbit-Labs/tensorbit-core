#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# test_all.sh — Tensorbit Core test runner
#
# Builds and runs all unit tests. Requires CMake 3.22+ and a C++20 compiler.
#
# Usage:
#   ./tests/test_all.sh [--clean] [--build-dir <path>]
#
# Options:
#   --clean        Remove build directory before building.
#   --build-dir    Path to CMake build directory (default: build/).
#   --eigen3-root  Path to Eigen3 installation (e.g., D:/eigen3).
# ---------------------------------------------------------------------------

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
CLEAN=0
EIGEN3_ROOT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --clean)
            CLEAN=1
            shift
            ;;
        --build-dir)
            BUILD_DIR="$2"
            shift 2
            ;;
        --eigen3-root)
            EIGEN3_ROOT="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: test_all.sh [--clean] [--build-dir <path>] [--eigen3-root <path>]"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=== Tensorbit Core Test Runner ==="
echo "Root:     ${ROOT_DIR}"
echo "Build:    ${BUILD_DIR}"
echo "Eigen3:   ${EIGEN3_ROOT:-auto}"

# --- Configure ---
if [[ "$CLEAN" -eq 1 && -d "$BUILD_DIR" ]]; then
    echo "--- Cleaning build directory ---"
    rm -rf "$BUILD_DIR"
fi

CMAKE_ARGS=(
    -DCMAKE_BUILD_TYPE=Debug
    -DTENSORBIT_BUILD_TESTS=ON
)

if [[ -n "$EIGEN3_ROOT" ]]; then
    CMAKE_ARGS+=(-DEIGEN3_ROOT="$EIGEN3_ROOT")
fi

echo "--- Configuring CMake ---"
cmake -S "$ROOT_DIR" -B "$BUILD_DIR" "${CMAKE_ARGS[@]}"

# --- Build ---
echo "--- Building ---"
cmake --build "$BUILD_DIR" --parallel

# --- Run Tests ---
echo "--- Running Tests ---"
cd "$BUILD_DIR"
ctest --output-on-failure --test-dir "$BUILD_DIR"

echo "=== All tests passed ==="
