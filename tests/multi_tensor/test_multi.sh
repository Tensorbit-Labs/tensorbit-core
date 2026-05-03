#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# test_multi.sh — Multi-tensor pruning test
#
# Generates a small .safetensors file with 5 tensors, runs tb-prune on it,
# and verifies that the output directory contains one .tb file per tensor.
#
# Prerequisites: Python 3.10+ with torch and safetensors
#   On WSL/Ubuntu: pip3 install torch safetensors --user
#
# Usage:
#   bash tests/multi_tensor/test_multi.sh
# ---------------------------------------------------------------------------

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BUILD_DIR="${ROOT}/build"
TEST_DIR="${ROOT}/tests/multi_tensor"
MODEL_FILE="${TEST_DIR}/multi_model.safetensors"
OUTPUT_DIR="${TEST_DIR}/output"

echo "=== Multi-Tensor Pruning Test ==="
echo "Root:  ${ROOT}"
echo ""

# --- Check prerequisites ---
PYTHON=""

# Prefer D: drive venv (where user has torch installed)
for candidate in \
    "/mnt/d/venv/tensorbit/bin/python3" \
    "/mnt/d/venv/tensorbit/bin/python"; do
    if [[ -x "$candidate" ]]; then
        PYTHON="$candidate"
        break
    fi
done

# Fallback to system python3
if [[ -z "$PYTHON" ]]; then
    for cmd in python3 python; do
        if command -v "$cmd" &>/dev/null; then
            PYTHON="$cmd"
            break
        fi
    done
fi

if [[ -z "$PYTHON" ]]; then
    echo "[ERROR] No Python found. Install: sudo apt install python3"
    exit 1
fi

echo "[INFO] Using Python: $($PYTHON --version 2>&1)"

# Try importing required packages (auto-install if missing)
# Check torch and safetensors (not safetensors.torch — that needs 'packaging')
if ! $PYTHON -c "import torch; import safetensors" 2>/dev/null; then
    echo "[INFO] torch or safetensors not found. Installing..."
    $PYTHON -m pip install numpy packaging --quiet 2>/dev/null || true
    $PYTHON -m pip install torch safetensors --quiet 2>/dev/null || {
        $PYTHON -m pip install torch --index-url https://download.pytorch.org/whl/cpu --quiet 2>/dev/null
        $PYTHON -m pip install safetensors numpy packaging --quiet 2>/dev/null || true
    }
    if ! $PYTHON -c "import torch; import safetensors" 2>/dev/null; then
        echo "[ERROR] Failed to install torch/safetensors."
        echo "  Activate your venv and run: pip install torch safetensors numpy packaging"
        exit 1
    fi
fi
echo "[INFO] torch/safetensors available."

# --- Check tb-prune binary ---

# --- Check tb-prune binary ---
TB_PRUNE="${BUILD_DIR}/bin/tb-prune"
if [[ ! -f "${TB_PRUNE}" ]]; then
    echo "[ERROR] tb-prune not found at ${TB_PRUNE}."
    echo "  Build first: cd build && cmake .. -DTENSORBIT_ENABLE_CUDA=OFF && cmake --build . --target tb-prune --parallel -j4"
    exit 1
fi

# --- Generate multi-tensor safetensors ---
echo "[1/4] Generating multi-tensor .safetensors..."
cd "${TEST_DIR}"
$PYTHON create_multi_model.py --output "${MODEL_FILE}"
echo ""

# --- Clean output ---
rm -rf "${OUTPUT_DIR}"

# --- Run tb-prune ---
echo "[2/4] Running tb-prune on ${MODEL_FILE}..."
"${TB_PRUNE}" \
    --model "${MODEL_FILE}" \
    --sparsity 2:4 \
    --strategy OneShot \
    --output "${OUTPUT_DIR}/"
echo ""

# --- Verify output ---
echo "[3/4] Verifying output..."
TB_COUNT=0
for f in "${OUTPUT_DIR}"/*.tb; do
    [[ -f "$f" ]] || continue
    TB_COUNT=$((TB_COUNT + 1))
    # Check magic bytes
    MAGIC=$(xxd -l4 -ps "$f")
    if [[ "$MAGIC" != "54423031" ]]; then
        echo "  FAIL: $f has bad magic: $MAGIC"
        exit 1
    fi
    echo "  PASS: $(basename "$f") ($(stat -c%s "$f" 2>/dev/null || stat -f%z "$f" 2>/dev/null) bytes)"
done

echo ""
if [[ "$TB_COUNT" -eq 5 ]]; then
    echo "[4/4] SUCCESS — 5 tensors pruned, 5 valid .tb files produced."
else
    echo "[4/4] FAIL — Expected 5 .tb files, found ${TB_COUNT}."
    exit 1
fi

echo ""
echo "Output directory: ${OUTPUT_DIR}"
ls -la "${OUTPUT_DIR}"/*.tb
