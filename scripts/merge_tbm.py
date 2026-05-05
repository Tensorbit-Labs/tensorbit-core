#!/usr/bin/env python3
"""merge_tbm.py — Merge .tb directories into a unified .tbm container.

After pruning multi-shard models (e.g., Mistral 7B sharded into 2 files),
each shard produces a directory of .tb files (one per tensor) and a model.tbm
containing only that shard's tensors.  This script merges multiple such
directories into a SINGLE .tbm container that tensorbit-run can load.

Usage:
  python scripts/merge_tbm.py \
      --input ./pruned/shard1/ ./pruned/shard2/ \
      --output ./pruned/full/model.tbm

The script reads every .tb file from the input directories, concatenates
them, and writes a unified JSON index with correct byte offsets.
"""

import argparse
import json
import struct
import sys
from pathlib import Path

from typing import BinaryIO, List, Dict, Any

TB_HEADER_SIZE = 4096
TB_MAGIC = 0x31304254  # "TB01" in LE


def parse_tb_header(path: Path) -> dict:
    """Read the TBHeader from a .tb file and return key fields."""
    with open(path, "rb") as f:
        header = f.read(TB_HEADER_SIZE)
        if len(header) < TB_HEADER_SIZE:
            raise ValueError(f"Truncated .tb header in {path}")

        magic = struct.unpack_from("<I", header, 0)[0]
        if magic != TB_MAGIC:
            raise ValueError(f"Bad magic 0x{magic:08X} in {path}")

        version = struct.unpack_from("<I", header, 4)[0]
        nm_n = struct.unpack_from("<I", header, 8)[0]
        nm_m = struct.unpack_from("<I", header, 12)[0]
        num_weights = struct.unpack_from("<Q", header, 16)[0]
        num_mask_bytes = struct.unpack_from("<Q", header, 24)[0]
        weights_offset = struct.unpack_from("<Q", header, 32)[0]
        masks_offset = struct.unpack_from("<Q", header, 40)[0]
        precision = header[48]

        return {
            "version": version,
            "nm_n": nm_n,
            "nm_m": nm_m,
            "num_weights": num_weights,
            "num_mask_bytes": num_mask_bytes,
            "weights_offset": weights_offset,
            "masks_offset": masks_offset,
            "precision": precision,
        }


def infer_shape(name: str, num_weights: int) -> List[int]:
    """Infer weight matrix shape from num_weights and naming convention."""
    # Default: assume 2D square-ish matrix based on num_weights
    # For layer names like "q_proj.weight", "k_proj.weight", "v_proj.weight"
    # the shape is typically [hidden, hidden] or [hidden, 3*hidden]
    # We infer from the tensor name if possible.
    # Fallback: [num_weights, 1] which run can handle.

    # Try to determine from naming convention
    name_lower = name.lower()
    if "embed" in name_lower or "lm_head" in name_lower:
        # Embedding / LM head: [vocab, hidden] — can't infer, use flat
        return [num_weights, 1]
    return [num_weights, 1]


def build_json_tensors(tb_files: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build the JSON tensor index entries from parsed .tb files."""
    tensors = []
    offset = 0
    for entry in tb_files:
        path = entry["path"]
        name = entry["name"]
        hdr = entry["header"]

        weight_size = hdr["num_weights"] * 4  # FP32 = 4 bytes
        mask_size = hdr["num_mask_bytes"]
        file_size = TB_HEADER_SIZE + weight_size + mask_size

        shape = entry.get("shape") or infer_shape(name, hdr["num_weights"])

        dtype_str = "fp32"
        if hdr["precision"] == 1:
            dtype_str = "fp16"
        elif hdr["precision"] == 2:
            dtype_str = "bf16"
        elif hdr["precision"] == 3:
            dtype_str = "fp64"

        tensors.append({
            "name": name,
            "offset": offset,
            "shape": shape,
            "nm_n": hdr["nm_n"],
            "nm_m": hdr["nm_m"],
            "dtype": dtype_str,
            "num_weights": hdr["num_weights"],
            "num_mask_bytes": hdr["num_mask_bytes"],
        })

        offset += file_size

    return tensors


def write_tbm(output_path: Path, tb_files: List[Dict[str, Any]],
               tensors: List[Dict[str, Any]], architecture: str = "llama"):
    """Write the merged .tbm file."""
    with open(output_path, "wb") as f:
        # Write each .tb file's contents in order
        for entry in tb_files:
            path = entry["path"]
            with open(path, "rb") as tb:
                file_data = tb.read()
            f.write(file_data)

        # Build and write JSON index
        json_index = json.dumps({
            "architecture": architecture,
            "config": {
                "num_layers": len(tensors),
            },
            "tensors": tensors,
        }, separators=(",", ":"))

        # Validate: tensor names must not contain characters that break JSON
        for t in tensors:
            name = t["name"]
            for c in name:
                if ord(c) < 32 or ord(c) == 127:
                    raise ValueError(
                        f"Tensor name contains unprintable char 0x{ord(c):02X}: {name}")

        # Write JSON as UTF-8
        json_bytes = json_index.encode("utf-8")
        f.write(json_bytes)

        # Write 4-byte tail (JSON length, little-endian)
        f.write(struct.pack("<I", len(json_bytes)))

    total_files = len(tb_files)
    total_size = output_path.stat().st_size
    print(f"[merge_tbm] Merged {total_files} .tb files into '{output_path}' "
          f"({total_size:,} bytes, {len(tensors)} tensors)")


def main():
    parser = argparse.ArgumentParser(
        description="Merge .tb directories into a unified .tbm container")
    parser.add_argument("--input", required=True, nargs="+",
                        help="Input directories containing .tb files")
    parser.add_argument("--output", required=True,
                        help="Output .tbm file path")
    parser.add_argument("--architecture", default="llama",
                        help="Model architecture name (default: llama)")
    args = parser.parse_args()

    # Collect all .tb files with their metadata
    tb_files: List[Dict[str, Any]] = []
    seen_names: set = set()

    for input_dir in args.input:
        dir_path = Path(input_dir)
        if not dir_path.is_dir():
            print(f"[WARN] Not a directory, skipping: {input_dir}")
            continue

        for tb_path in sorted(dir_path.glob("*.tb")):
            if tb_path.name == "model.tbm":
                continue

            # Derive tensor name from filename (remove .tb)
            name = tb_path.stem.replace(".tbm", "").replace(".tb", "")

            # Handle duplicates by appending a suffix
            while name in seen_names:
                print(f"[WARN] Duplicate tensor name: {name} (skipping)")
                break
            else:
                seen_names.add(name)

                try:
                    header = parse_tb_header(tb_path)
                    tb_files.append({
                        "path": tb_path,
                        "name": name,
                        "header": header,
                    })
                except ValueError as e:
                    print(f"[WARN] {e}")

    if not tb_files:
        print("[ERROR] No valid .tb files found")
        sys.exit(1)

    # Build JSON tensor index
    tensors = build_json_tensors(tb_files)

    # Write merged .tbm
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_tbm(output_path, tb_files, tensors, args.architecture)


if __name__ == "__main__":
    main()
