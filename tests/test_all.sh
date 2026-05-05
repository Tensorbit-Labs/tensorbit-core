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
#   --eigen3-root  Path to Eigen3 installation (e.g., D:/eigen3 or /usr/include/eigen3).
#   --skip-gpu     Skip CUDA checks (useful for CPU-only WSL / CI).
# ---------------------------------------------------------------------------

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
CLEAN=0
EIGEN3_ROOT=""
SKIP_GPU=0

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
        --skip-gpu)
            SKIP_GPU=1
            shift
            ;;
        --help|-h)
            echo "Usage: test_all.sh [--clean] [--build-dir <path>] [--eigen3-root <path>] [--skip-gpu]"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ===========================================================================
# Prerequisite Checks
# ===========================================================================

echo "=== Tensorbit Core Test Runner ==="
echo "Root:     ${ROOT_DIR}"
echo "Build:    ${BUILD_DIR}"
echo "Eigen3:   ${EIGEN3_ROOT:-auto}"
echo ""

MISSING=0

check_cmd() {
    local name="$1"
    local install_hint="$2"
    if ! command -v "$name" &>/dev/null; then
        log_error "'$name' not found. Install it: $install_hint"
        MISSING=1
    else
        log_info "'$name' found: $(command -v "$name")"
    fi
}

check_file() {
    local name="$1"
    local path="$2"
    local install_hint="$3"
    if [[ ! -f "$path" ]]; then
        log_error "'$name' not found at '$path'. $install_hint"
        MISSING=1
    else
        log_info "'$name' found: $path"
    fi
}

# --- Required: cmake ---
check_cmd cmake \
    "sudo apt install cmake   (Ubuntu)   OR   https://cmake.org/download/"

# --- Required: C++ compiler ---
if command -v g++ &>/dev/null; then
    log_info "g++ found: $(g++ --version | head -1)"
elif command -v clang++ &>/dev/null; then
    log_info "clang++ found: $(clang++ --version | head -1)"
else
    log_error "No C++ compiler found. Install: sudo apt install build-essential"
    MISSING=1
fi

# --- Optional but expected: nvcc (CUDA) ---
if [[ "$SKIP_GPU" -eq 1 ]]; then
    log_warn "Skipping CUDA check (--skip-gpu). Only CPU host code will compile."
else
    if ! command -v nvcc &>/dev/null; then
        log_warn "'nvcc' not found. CUDA kernels will NOT compile."
        log_warn "  Install CUDA 12: https://developer.nvidia.com/cuda-downloads"
        log_warn "  Or run with --skip-gpu if you only need CPU host code."
    else
        log_info "nvcc found: $(nvcc --version | grep -oP 'release \K[\d.]+')"
    fi
fi

# --- Check Eigen3 ---
EIGEN_FOUND=0
if [[ -n "$EIGEN3_ROOT" ]]; then
    check_file "Eigen/Core" "$EIGEN3_ROOT/Eigen/Core" \
        "Install Eigen3: git clone https://gitlab.com/libeigen/eigen.git $EIGEN3_ROOT"
    EIGEN_FOUND=1
else
    for candidate in \
        /usr/include/eigen3/Eigen/Core \
        /usr/local/include/eigen3/Eigen/Core \
        /opt/homebrew/include/eigen3/Eigen/Core \
        "$ROOT_DIR/third_party/eigen/Eigen/Core"; do
        if [[ -f "$candidate" ]]; then
            log_info "Eigen3 auto-detected at: $(dirname "$(dirname "$candidate")")"
            EIGEN3_ROOT="$(dirname "$(dirname "$candidate")")"
            EIGEN_FOUND=1
            break
        fi
    done
    if [[ "$EIGEN_FOUND" -eq 0 ]]; then
        log_error "Eigen3 not found. Provide it with --eigen3-root <path>."
        log_error "  Install: sudo apt install libeigen3-dev (Ubuntu/Debian)"
        log_error "       or: brew install eigen (macOS)"
        log_error "       or: git clone https://gitlab.com/libeigen/eigen.git <path>"
        log_error "       or: git clone https://gitlab.com/libeigen/eigen.git /usr/local/include/eigen3"
        MISSING=1
    fi
fi

if [[ "$MISSING" -eq 1 ]]; then
    log_error "One or more prerequisites are missing. See above for install instructions."
    exit 1
fi

log_info "All prerequisites satisfied."
echo ""

# ===========================================================================
# Configure
# ===========================================================================

if [[ "$CLEAN" -eq 1 && -d "$BUILD_DIR" ]]; then
    echo "--- Cleaning build directory ---"
    rm -rf "$BUILD_DIR"
fi

CMAKE_ARGS=(
    -DCMAKE_BUILD_TYPE=Debug
    -DTENSORBIT_BUILD_TESTS=ON
)

if [[ "$SKIP_GPU" -eq 1 ]]; then
    CMAKE_ARGS+=(-DTENSORBIT_ENABLE_CUDA=OFF)
    log_info "Passing -DTENSORBIT_ENABLE_CUDA=OFF to CMake."
fi

if [[ -n "$EIGEN3_ROOT" ]]; then
    CMAKE_ARGS+=(-DEIGEN3_ROOT="$EIGEN3_ROOT")
fi

echo "--- Configuring CMake ---"
cmake -S "$ROOT_DIR" -B "$BUILD_DIR" "${CMAKE_ARGS[@]}"
echo ""

# ===========================================================================
# Build
# ===========================================================================

echo "--- Building ---"
cmake --build "$BUILD_DIR" --parallel
echo ""

# ===========================================================================
# Run Tests
# ===========================================================================

echo "--- Running Tests ---"
cd "$BUILD_DIR"
ctest --output-on-failure --test-dir "$BUILD_DIR"
echo ""

echo "=== All tests passed ==="
