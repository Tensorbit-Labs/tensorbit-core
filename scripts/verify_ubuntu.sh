#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# verify_ubuntu.sh — Check your WSL / Ubuntu environment for Tensorbit Core.
#
# Run this from anywhere. It reports what you have and what's missing.
#
# Usage:
#   bash scripts/verify_ubuntu.sh
# ---------------------------------------------------------------------------

set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

PASS=0
WARN=0
FAIL=0

ok()   { echo -e "${GREEN}[OK]${NC}    $1"; PASS=$((PASS + 1)); }
warn() { echo -e "${YELLOW}[WARN]${NC}  $1"; WARN=$((WARN + 1)); }
fail() { echo -e "${RED}[MISS]${NC}  $1"; FAIL=$((FAIL + 1)); }
info() { echo -e "${CYAN}---${NC} $1"; }

echo "======================================================================"
echo " Tensorbit Core — Environment Verification"
echo "======================================================================"
echo ""

# ---------------------------------------------------------------------------
# 1. Operating System
# ---------------------------------------------------------------------------
info "1. Operating System"
if [[ -f /etc/os-release ]]; then
    # shellcheck disable=SC1091
    source /etc/os-release
    echo "   OS:      ${NAME:-unknown} ${VERSION_ID:-unknown}"
    echo "   Kernel:  $(uname -r)"
    echo "   Arch:    $(uname -m)"
    if [[ "$ID" == "ubuntu" ]] || [[ "$ID" == "debian" ]]; then
        ok "Debian-based distro detected (compatible with setup_cloud.sh)"
    else
        warn "Non-Debian distro — setup_cloud.sh may not work directly"
    fi
else
    fail "/etc/os-release not found — cannot detect OS"
fi
echo ""

# ---------------------------------------------------------------------------
# 2. WSL Check
# ---------------------------------------------------------------------------
info "2. WSL Detection"
if grep -qi microsoft /proc/version 2>/dev/null; then
    ok "Running inside WSL"
    WSL_VERSION=$(wsl.exe --version 2>/dev/null | head -1 || echo "unknown")
    echo "   ${WSL_VERSION}"
    dpath=$(wslpath -w "$(pwd)" 2>/dev/null || echo "n/a")
    echo "   Current dir maps to: ${dpath}"
else
    info "Not WSL (bare Linux or cloud VM)"
fi
echo ""

# ---------------------------------------------------------------------------
# 3. C++ Compiler
# ---------------------------------------------------------------------------
info "3. C++ Compiler"
if command -v g++ &>/dev/null; then
    GXX_VERSION=$(g++ --version | head -1)
    GXX_MAJOR=$(g++ -dumpversion | cut -d. -f1)
    echo "   ${GXX_VERSION}"
    if [[ "$GXX_MAJOR" -ge 12 ]]; then
        ok "GCC ${GXX_MAJOR} supports C++20"
    else
        fail "GCC ${GXX_MAJOR} too old — need 12+ for C++20"
    fi
else
    fail "g++ not found"
fi

if command -v clang++ &>/dev/null; then
    CLANG_VERSION=$(clang++ --version | head -1)
    echo "   (clang available: ${CLANG_VERSION})"
fi
echo ""

# ---------------------------------------------------------------------------
# 4. C++20 Feature Test
# ---------------------------------------------------------------------------
info "4. C++20 Standard Library Check"
TMPDIR=$(mktemp -d)
cat > "$TMPDIR/test_cxx20.cpp" <<'EOF'
#include <concepts>
#include <format>
#include <span>
#include <source_location>
#include <bit>
#include <expected>
#if __cplusplus < 202002L
#error "Not C++20"
#endif
int main() {
    // Test <format> — vformat and make_format_args (require lvalues)
    int val42 = 42;
    auto s = std::vformat("{}", std::make_format_args(val42));
    // Test <span>
    int arr[3];
    std::span<int> sp(arr, 3);
    // Test <source_location>
    auto loc = std::source_location::current();
    // Test <bit> — has_single_bit
    bool b = std::has_single_bit(4u);
    (void)s; (void)sp; (void)loc; (void)b;
    return 0;
}
EOF
if g++ -std=c++20 -c "$TMPDIR/test_cxx20.cpp" -o /dev/null 2>/dev/null; then
    ok "C++20: <format>, <span>, <source_location>, <bit> all compile"
else
    fail "C++20 features failed to compile — check g++ installation"
    echo "   Error details:"
    g++ -std=c++20 -c "$TMPDIR/test_cxx20.cpp" -o /dev/null 2>&1 | head -5
fi
rm -rf "$TMPDIR"
echo ""

# ---------------------------------------------------------------------------
# 5. CMake
# ---------------------------------------------------------------------------
info "5. CMake"
if command -v cmake &>/dev/null; then
    CMAKE_VERSION=$(cmake --version | head -1 | awk '{print $3}')
    CMAKE_MAJOR=$(echo "$CMAKE_VERSION" | cut -d. -f1)
    CMAKE_MINOR=$(echo "$CMAKE_VERSION" | cut -d. -f2)
    echo "   ${CMAKE_VERSION}"
    if [[ "$CMAKE_MAJOR" -gt 3 ]] || \
       { [[ "$CMAKE_MAJOR" -eq 3 ]] && [[ "$CMAKE_MINOR" -ge 22 ]]; }; then
        ok "CMake 3.22+ found"
    else
        fail "CMake too old — need 3.22+ for CUDA 12 support"
    fi
else
    fail "cmake not found"
fi
echo ""

# ---------------------------------------------------------------------------
# 6. Eigen3
# ---------------------------------------------------------------------------
info "6. Eigen3"
EIGEN_FOUND=0
for candidate in \
    /usr/include/eigen3/Eigen/Core \
    /usr/local/include/eigen3/Eigen/Core \
    /usr/include/Eigen/Core \
    /usr/local/include/Eigen/Core; do
    if [[ -f "$candidate" ]]; then
        ok "Eigen3 found: $(dirname "$(dirname "$candidate")")"
        EIGEN_FOUND=1
        break
    fi
done
if [[ "$EIGEN_FOUND" -eq 0 ]]; then
    fail "Eigen3 not found. Install: sudo apt install libeigen3-dev"
fi
echo ""

# ---------------------------------------------------------------------------
# 7. CUDA (optional — not needed for CPU-only builds)
# ---------------------------------------------------------------------------
info "7. CUDA Toolkit (optional)"
if command -v nvcc &>/dev/null; then
    NVCC_VERSION=$(nvcc --version | grep -oP 'release \K[\d.]+')
    ok "nvcc found (CUDA ${NVCC_VERSION})"
    if command -v nvidia-smi &>/dev/null; then
        GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "unknown")
        echo "   GPU: ${GPU}"
    fi
else
    warn "nvcc not found — CPU-only builds via --skip-gpu will work"
fi
echo ""

# ---------------------------------------------------------------------------
# 8. Disk Space
# ---------------------------------------------------------------------------
info "8. Disk Space"
if command -v df &>/dev/null; then
    echo "   $(df -h /mnt/d 2>/dev/null || df -h / 2>/dev/null || echo 'n/a')"
else
    warn "df not available"
fi
echo ""

# ---------------------------------------------------------------------------
# 9. Memory
# ---------------------------------------------------------------------------
info "9. Memory"
if command -v free &>/dev/null; then
    echo "   $(free -h | grep -E '^Mem:' || true)"
else
    warn "free not available"
fi
echo ""

# ---------------------------------------------------------------------------
# 10. Build Tools
# ---------------------------------------------------------------------------
info "10. Build Tools (optional)"
for tool in ninja ccache make; do
    if command -v "$tool" &>/dev/null; then
        ok "$tool available"
    else
        warn "$tool not found (optional)"
    fi
done
echo ""

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo "======================================================================"
echo " SUMMARY:  ${GREEN}${PASS} OK${NC}  |  ${YELLOW}${WARN} WARN${NC}  |  ${RED}${FAIL} MISS${NC}"
echo "======================================================================"

if [[ "$FAIL" -gt 0 ]]; then
    echo ""
    echo "${RED}Some required tools are missing.${NC} Install them before running:"
    echo ""
    echo "  sudo apt update"
    echo "  sudo apt install -y build-essential cmake libeigen3-dev"
    echo ""
    exit 1
fi

echo ""
echo "${GREEN}Environment looks good for CPU-only builds.${NC}"
echo ""
echo "Run the tests with:"
echo "  cd /mnt/d/Dev/tensorbit_labs/tensorbit-core"
echo "  bash tests/test_all.sh --skip-gpu --clean"
echo ""
exit 0
