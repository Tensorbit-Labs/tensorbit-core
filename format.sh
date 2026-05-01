#!/bin/bash
# ---------------------------------------------------------------------------
# format.sh — Format all C++/CUDA source files with clang-format.
#
# Requires clang-format in PATH.
# Install: sudo apt install clang-format   (Ubuntu)
#       or: pip install clang-format       (any OS)
# ---------------------------------------------------------------------------

set -euo pipefail

if ! command -v clang-format &>/dev/null; then
    echo "[ERROR] clang-format not found in PATH." >&2
    echo "  Install: sudo apt install clang-format   (Ubuntu)" >&2
    echo "       or: pip install clang-format         (any OS)" >&2
    exit 1
fi

echo "Formatting files with clang-format $(clang-format --version)..."
find . -iname "*.cpp" -o -iname "*.hpp" -o -iname "*.cu" -o -iname "*.h" \
    | xargs clang-format -i

echo "Files formatted successfully."
