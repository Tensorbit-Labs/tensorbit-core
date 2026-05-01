#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# setup_cloud.sh — Provision a cloud VM for Tensorbit Core development.
#
# Targets Ubuntu 22.04+ with NVIDIA H100/A100 GPUs.
# Run as root or with sudo.
#
# Usage:
#   sudo ./scripts/setup_cloud.sh
#
# Installed components:
#   - GCC 13 / Clang 17 (C++20 compliant)
#   - CMake 3.28+ (CUDA 12 language support)
#   - CUDA Toolkit 12.6 (includes cuBLAS, cuSPARSE)
#   - Eigen3 (header-only, v3.4+)
#   - Python 3.11 + PyTorch (for gradient extraction from HuggingFace models)
#   - Development tools: ccache, ninja, clang-format, valgrind
# ---------------------------------------------------------------------------

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

log() { echo -e "${GREEN}[setup]${NC} $*"; }
err() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# --- Platform Detection ---
if [[ ! -f /etc/os-release ]]; then
    err "Unsupported OS. Expected Ubuntu 22.04+."
    exit 1
fi

source /etc/os-release
log "Detected: ${NAME} ${VERSION_ID}"

if [[ "$ID" != "ubuntu" ]]; then
    err "This script is designed for Ubuntu. Other distros may require manual steps."
    exit 1
fi

# --- System Dependencies ---
log "Updating package lists..."
apt-get update -qq

log "Installing build essentials..."
apt-get install -y -qq \
    build-essential \
    cmake \
    ninja-build \
    ccache \
    clang-format \
    gdb \
    valgrind \
    git \
    curl \
    wget \
    ca-certificates \
    libssl-dev

# --- Install latest CMake if needed ---
CMAKE_VERSION=$(cmake --version | head -1 | awk '{print $3}')
CMAKE_MAJOR=$(echo "$CMAKE_VERSION" | cut -d. -f1)
CMAKE_MINOR=$(echo "$CMAKE_VERSION" | cut -d. -f2)

if [[ "$CMAKE_MAJOR" -lt 3 || ("$CMAKE_MAJOR" -eq 3 && "$CMAKE_MINOR" -lt 22) ]]; then
    log "CMake $CMAKE_VERSION is too old. Installing 3.28+ via Kitware repo..."
    wget -qO - https://apt.kitware.com/keys/kitware-archive-latest.asc | gpg --dearmor -o /usr/share/keyrings/kitware-archive-keyring.gpg
    echo "deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ ${UBUNTU_CODENAME} main" > /etc/apt/sources.list.d/kitware.list
    apt-get update -qq
    apt-get install -y -qq cmake
fi

# --- CUDA Toolkit 12 ---
if ! command -v nvcc &>/dev/null; then
    log "Installing CUDA Toolkit 12.6..."
    wget -q "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb" -O /tmp/cuda-keyring.deb
    dpkg -i /tmp/cuda-keyring.deb
    apt-get update -qq
    apt-get install -y -qq cuda-toolkit-12-6
    rm -f /tmp/cuda-keyring.deb

    # Add CUDA to PATH
    cat >> /etc/profile.d/cuda.sh <<'EOF'
export PATH=/usr/local/cuda-12/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12/lib64:$LD_LIBRARY_PATH
EOF
    log "CUDA 12 installed. Reboot or source /etc/profile.d/cuda.sh to activate."
else
    NVIDIA_SMI=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null || echo "unknown")
    log "CUDA already installed (driver: ${NVIDIA_SMI})."
fi

# --- Eigen3 ---
EIGEN_INSTALL_DIR="/usr/local/include/eigen3"
if [[ ! -f "$EIGEN_INSTALL_DIR/Eigen/Core" ]]; then
    log "Installing Eigen3 v3.4.0..."
    EIGEN_URL="https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz"
    curl -sL "$EIGEN_URL" | tar xz -C /tmp/
    mkdir -p "$EIGEN_INSTALL_DIR"
    cp -r /tmp/eigen-3.4.0/Eigen "$EIGEN_INSTALL_DIR/"
    cp -r /tmp/eigen-3.4.0/unsupported "$EIGEN_INSTALL_DIR/" 2>/dev/null || true
    cp /tmp/eigen-3.4.0/signature_of_eigen3_matrix_library "$EIGEN_INSTALL_DIR/" 2>/dev/null || true
    rm -rf /tmp/eigen-3.4.0
    log "Eigen3 installed to $EIGEN_INSTALL_DIR"
else
    log "Eigen3 already installed at $EIGEN_INSTALL_DIR"
fi

# --- Python for Model Utilities ---
log "Setting up Python 3.11 environment..."
apt-get install -y -qq python3.11 python3.11-venv python3.11-dev

if [[ ! -d /opt/tensorbit-venv ]]; then
    python3.11 -m venv /opt/tensorbit-venv
fi

# shellcheck disable=SC1091
source /opt/tensorbit-venv/bin/activate

pip install --quiet --upgrade pip setuptools wheel
pip install --quiet \
    torch \
    torchvision \
    safetensors \
    numpy \
    huggingface_hub \
    click

log "Python environment ready at /opt/tensorbit-venv"

# --- ccache config ---
ccache --max-size=20G || true
log "ccache configured (20 GB cache)."

log ""
log "=== Setup Complete ==="
log "  CUDA:    /usr/local/cuda-12"
log "  Eigen3:  $EIGEN_INSTALL_DIR"
log "  Python:  /opt/tensorbit-venv (activate with: source /opt/tensorbit-venv/bin/activate)"
log ""
log "Next steps:"
log "  git clone https://github.com/your-org/tensorbit-core.git"
log "  cd tensorbit-core"
log "  mkdir build && cd build"
log "  cmake .. -DEIGEN3_ROOT=${EIGEN_INSTALL_DIR} -GNinja"
log "  ninja"
