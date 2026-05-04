#!/usr/bin/env python3
"""create_test_tb.py — Generate small .tb files for testing merge_tbm.py.

Creates N .tb files with sequential tensor names in a temp directory.
Each .tb has a valid TBHeader, FP32 weight blob, and mask blob.
"""

import argparse
import os
import struct
import tempfile

TB_HEADER_SIZE = 4096
TB_MAGIC = 0x31304254


def create_tb(path: str, name: str, num_weights: int, nm_n: int = 2, nm_m: int = 4):
    """Create a valid .tb file with the given tensor name and size."""
    weight_byte_size = num_weights * 4  # FP32
    num_groups = num_weights // nm_m
    mask_byte_size = num_groups

    with open(path, "wb") as f:
        # Write header (4096 bytes)
        hdr = bytearray(TB_HEADER_SIZE)
        struct.pack_into("<I", hdr, 0, TB_MAGIC)          # magic
        struct.pack_into("<I", hdr, 4, 1)                 # version
        struct.pack_into("<I", hdr, 8, nm_n)              # nm_n
        struct.pack_into("<I", hdr, 12, nm_m)             # nm_m
        struct.pack_into("<Q", hdr, 16, num_weights)      # num_weights
        struct.pack_into("<Q", hdr, 24, mask_byte_size)   # num_mask_bytes (LE -> actually this is fine)
        struct.pack_into("<Q", hdr, 32, TB_HEADER_SIZE)   # weights_offset
        struct.pack_into("<Q", hdr, 40, TB_HEADER_SIZE + weight_byte_size)  # masks_offset
        hdr[48] = 0  # precision = FP32
        f.write(hdr)

        # Write weight blob (all 1.0f)
        f.write(struct.pack(f"<{num_weights}f", *([1.0] * num_weights)))

        # Write mask blob (all 0xFF = all weights kept)
        f.write(b'\xFF' * mask_byte_size)


TENSORS = {
    "model.embed_tokens.weight": 32000,
    "model.layers.0.self_attn.q_proj.weight": 4096,
    "model.layers.0.self_attn.v_proj.weight": 4096,
    "model.layers.0.mlp.gate_proj.weight": 8192,
    "model.norm.weight": 4096,
    "lm_head.weight": 32000,
}


def main():
    parser = argparse.ArgumentParser(description="Generate test .tb files")
    parser.add_argument("--output", "-o", default="/tmp/tb_test",
                        help="Output directory for .tb files")
    parser.add_argument("--count", "-c", type=int, default=len(TENSORS),
                        help="Number of tensors to generate")
    args = parser.parse_args()

    out_dir = args.output
    os.makedirs(out_dir, exist_ok=True)

    items = list(TENSORS.items())[:args.count]
    for i, (name, num_weights) in enumerate(items):
        path = os.path.join(out_dir, f"{name.replace('/', '_')}.tb")
        create_tb(path, name, num_weights)

    total_size = sum(os.path.getsize(os.path.join(out_dir, f"{name.replace('/', '_')}.tb"))
                     for name, _ in items)
    print(f"Created {len(items)} .tb files in '{out_dir}' ({total_size:,} bytes total)")


if __name__ == "__main__":
    main()
