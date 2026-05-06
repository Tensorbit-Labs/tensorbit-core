#!/usr/bin/env python3
"""fix_shapes.py — Patch tensor shapes in an existing .tbm file.

If a .tbm was produced by an old version of merge_tbm.py that wrote all tensors
as [num_weights, 1], this script infers the correct 2D shapes from tensor names
and num_weights, and patches the JSON index in-place.  The weight/mask blob data
is NOT touched — only the JSON metadata is fixed.

Usage:
  python scripts/fix_shapes.py path/to/model.tbm [--output fixed.tbm]
"""

import argparse
import json
import math
import os
import struct
import sys
from pathlib import Path
from copy import deepcopy


def find_hidden_size(tensors):
    """Find model hidden size from a square attention projection matrix.
       Prioritises q_proj / o_proj (always [hidden, hidden]).
       Falls back to k_proj / v_proj (smaller square in GQA models)."""
    primary = []
    fallback = []
    for t in tensors:
        name = t.get("name", "").lower()
        nw = t.get("num_weights", 0)
        h = int(math.isqrt(nw))
        if h * h != nw or h < 512:
            continue
        if any(k in name for k in ("q_proj", "o_proj")):
            primary.append(h)
        elif any(k in name for k in ("k_proj", "v_proj")):
            fallback.append(h)

    if primary:
        return max(primary)  # largest reliable square = hidden_size
    if fallback:
        return min(fallback)  # k/v_proj may be smaller in GQA
    return 0


def infer_shape(tensor, hidden):
    """Infer correct 2D shape for a tensor."""
    name = tensor.get("name", "").lower()
    nw = tensor.get("num_weights", 0)

    if hidden <= 0 or nw == 0:
        return [nw, 1]

    # Norm tensors are 1D
    if "norm" in name or "layernorm" in name:
        return [nw, 1]

    if nw % hidden != 0:
        return [nw, 1]

    d2 = nw // hidden

    # Embedding / LM head: [vocab, hidden]
    if any(k in name for k in ("embed", "wte", "lm_head")):
        return [d2, hidden]

    # MLP down projection: [hidden, intermediate]
    if "down" in name or "fc_out" in name:
        return [hidden, d2]

    # Everything else: [dim, hidden]
    return [d2, hidden]


def patch_tbm(input_path, output_path):
    """Read .tbm, patch shapes in JSON index, write output."""
    with open(input_path, "rb") as f:
        fsize = os.fstat(f.fileno()).st_size
        if fsize < 4:
            print("ERROR: file too small")
            return False

        # Read 4-byte tail
        f.seek(-4, os.SEEK_END)
        idx_len = struct.unpack("<I", f.read(4))[0]
        if idx_len == 0 or idx_len > fsize - 4:
            print("ERROR: invalid index length")
            return False

        # Read JSON index
        json_start = fsize - 4 - idx_len
        f.seek(json_start)
        data = json.loads(f.read(idx_len))

        # Read blob data
        f.seek(0)
        blob_data = f.read(json_start)

    # Find hidden size and patch shapes
    tensors = data.get("tensors", [])
    if not tensors:
        print("ERROR: no tensors in .tbm")
        return False

    hidden = find_hidden_size(tensors)
    if hidden > 0:
        print(f"Inferred hidden_size = {hidden}")

    fixed = 0
    for t in tensors:
        old_shape = t.get("shape", [])
        new_shape = infer_shape(t, hidden)
        if old_shape != new_shape:
            old_s = "x".join(str(d) for d in old_shape)
            new_s = "x".join(str(d) for d in new_shape)
            print(f"  {t['name']}: [{old_s}] → [{new_s}]")
            t["shape"] = new_shape
            fixed += 1

    if fixed == 0:
        print("No shapes needed fixing — already correct.")
        return True

    # Write output
    with open(output_path, "wb") as f:
        f.write(blob_data)
        new_json = json.dumps(data, separators=(",", ":"))
        json_bytes = new_json.encode("utf-8")
        f.write(json_bytes)
        f.write(struct.pack("<I", len(json_bytes)))

    print(f"\nFixed {fixed} tensors. Output: {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Patch tensor shapes in a .tbm file")
    parser.add_argument("input", nargs="?", help="Input .tbm file")
    parser.add_argument("--output", "-o", help="Output .tbm file (default: patch in-place)")
    args = parser.parse_args()

    if not args.input:
        parser.print_help()
        sys.exit(1)

    out_path = args.output or args.input
    if not patch_tbm(args.input, out_path):
        sys.exit(1)


if __name__ == "__main__":
    main()
