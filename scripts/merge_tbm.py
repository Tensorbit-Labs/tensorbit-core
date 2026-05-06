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

Tensor shapes are inferred from naming conventions and per-shard .tbm JSON
indexes.  The merged .tbm preserves the full model config from the first shard.
"""

import argparse
import json
import math
import os
import struct
import sys
from pathlib import Path

from typing import Dict, Any, List, Optional

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

        nm_n = struct.unpack_from("<I", header, 8)[0]
        nm_m = struct.unpack_from("<I", header, 12)[0]
        num_weights = struct.unpack_from("<Q", header, 16)[0]
        num_mask_bytes = struct.unpack_from("<Q", header, 24)[0]
        precision = header[48]

        return {
            "nm_n": nm_n,
            "nm_m": nm_m,
            "num_weights": num_weights,
            "num_mask_bytes": num_mask_bytes,
            "precision": precision,
        }


def read_shard_metadata(input_dir: str) -> Dict[str, dict]:
    """Read a shard's model.tbm JSON index and return name→metadata map."""
    src_tbm = Path(input_dir) / "model.tbm"
    if not src_tbm.is_file():
        return {}

    with open(src_tbm, "rb") as f:
        fsize = os.fstat(f.fileno()).st_size
        if fsize < 4:
            return {}
        f.seek(-4, os.SEEK_END)
        idx_len = struct.unpack("<I", f.read(4))[0]
        if idx_len == 0 or idx_len > fsize - 4:
            return {}
        f.seek(-4 - idx_len, os.SEEK_END)
        data = json.loads(f.read(idx_len))

    meta = {}
    for t in data.get("tensors", []):
        name = t.get("name", "")
        if name:
            meta[name] = {
                "shape": t.get("shape", []),
                "nm_n": t.get("nm_n"),
                "nm_m": t.get("nm_m"),
                "dtype": t.get("dtype", "fp32"),
            }
    return meta


def infer_hidden_size(tb_files: List[Dict[str, Any]], shard_meta: Dict[str, dict]) -> int:
    """Find the model hidden size from a square attention projection matrix.
       Prioritises q_proj / o_proj (always [hidden, hidden]).
       Falls back to k_proj / v_proj (smaller square in GQA models)."""
    primary = []
    fallback = []
    for entry in tb_files:
        name = entry["name"].lower()
        nw = entry["header"]["num_weights"]
        h = int(math.isqrt(nw))
        if h * h != nw or h < 512:
            continue
        if any(k in name for k in ("q_proj", "o_proj")):
            primary.append(h)
        elif any(k in name for k in ("k_proj", "v_proj")):
            fallback.append(h)

    if primary:
        return max(primary)
    if fallback:
        return min(fallback)
    return 0


def infer_shape(name: str, num_weights: int, hidden: int) -> List[int]:
    """Infer 2D tensor shape from naming convention and known hidden size."""
    name_low = name.lower()

    if hidden <= 0:
        return [num_weights, 1]

    # 1D tensors
    if "norm" in name_low or "layernorm" in name_low or "_norm" in name_low:
        return [num_weights, 1]

    if num_weights == 0:
        return [0, 1]

    # 2D tensors — one dimension is hidden
    if num_weights % hidden != 0:
        # Fallback: try to factor
        return [num_weights, 1]

    d2 = num_weights // hidden

    # Embedding / LM head: [vocab, hidden]
    if any(k in name_low for k in ("embed", "wte", "lm_head")):
        return [d2, hidden]

    # MLP down projection: [hidden, intermediate]
    if "down" in name_low or "fc_out" in name_low or "dense_4h_to_h" in name_low:
        return [hidden, d2]

    # Attention output projection: [hidden, hidden] (same as q_proj)
    # Gate/Up/FC in: [intermediate, hidden] or [dim, hidden]
    return [d2, hidden]


def build_json_tensors(tb_files: List[Dict[str, Any]], shard_meta: Dict[str, dict],
                       hidden: int) -> List[Dict[str, Any]]:
    """Build the JSON tensor index entries from parsed .tb files."""
    tensors = []
    offset = 0
    for entry in tb_files:
        name = entry["name"]
        hdr = entry["header"]
        nw = hdr["num_weights"]

        weight_size = nw * 4
        mask_size = hdr["num_mask_bytes"]
        file_size = TB_HEADER_SIZE + weight_size + mask_size

        # Prefer shape from shared t.btm index, then infer
        shape = None
        if name in shard_meta:
            s = shard_meta[name].get("shape")
            if s and len(s) >= 2 and s[0] > 0 and s[1] > 0:
                shape = [int(s[0]), int(s[1])]
        if shape is None:
            shape = infer_shape(name, nw, hidden)

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
            "num_weights": nw,
            "num_mask_bytes": mask_size,
        })

        offset += file_size

    return tensors


def write_tbm(output_path: Path, tb_files: List[Dict[str, Any]],
              tensors: List[Dict[str, Any]], architecture: str,
              config: dict):
    """Write the merged .tbm file."""
    with open(output_path, "wb") as f:
        for entry in tb_files:
            with open(entry["path"], "rb") as tb:
                f.write(tb.read())

        json_index = json.dumps({
            "architecture": architecture,
            "config": config,
            "tensors": tensors,
        }, separators=(",", ":"))

        json_bytes = json_index.encode("utf-8")
        f.write(json_bytes)
        f.write(struct.pack("<I", len(json_bytes)))

    total_files = len(tb_files)
    total_size = output_path.stat().st_size
    print(f"[merge_tbm] Merged {total_files} .tb files → '{output_path}' "
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

    # Read per-shard .tbm JSON for accurate tensor metadata
    combined_meta: Dict[str, dict] = {}
    for input_dir in args.input:
        shard_meta = read_shard_metadata(input_dir)
        for name, meta in shard_meta.items():
            if name not in combined_meta:
                combined_meta[name] = meta

    # Collect all .tb files
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

            name = tb_path.stem
            if name in seen_names:
                print(f"[WARN] Duplicate tensor name: {name} (skipping)")
                continue
            seen_names.add(name)

            try:
                header = parse_tb_header(tb_path)
                tb_files.append({"path": tb_path, "name": name, "header": header})
            except ValueError as e:
                print(f"[WARN] {e}")

    if not tb_files:
        print("[ERROR] No valid .tb files found")
        sys.exit(1)

    hidden = infer_hidden_size(tb_files, combined_meta)
    if hidden > 0:
        print(f"[merge_tbm] Inferred hidden_size = {hidden}")

    tensors = build_json_tensors(tb_files, combined_meta, hidden)

    # Read config from first shard that has one
    config = {"num_layers": len(tensors)}
    for input_dir in args.input:
        src_tbm = Path(input_dir) / "model.tbm"
        if src_tbm.is_file():
            with open(src_tbm, "rb") as f:
                fsize = os.fstat(f.fileno()).st_size
                if fsize < 4:
                    continue
                f.seek(-4, os.SEEK_END)
                src_idx_len = struct.unpack("<I", f.read(4))[0]
                if src_idx_len == 0 or src_idx_len > fsize - 4:
                    continue
                f.seek(-4 - src_idx_len, os.SEEK_END)
                src_json = json.loads(f.read(src_idx_len))
                if "config" in src_json:
                    config = src_json["config"]
                if "architecture" in src_json and src_json["architecture"]:
                    args.architecture = src_json["architecture"]
                break

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_tbm(output_path, tb_files, tensors, args.architecture, config)


if __name__ == "__main__":
    main()
