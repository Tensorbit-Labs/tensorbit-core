#!/usr/bin/env python3
"""create_multi_model.py — Generate a small .safetensors file with multiple tensors.

Creates `multi_model.safetensors` containing 5 tensors of different shapes
to test multi-tensor pruning with `tb-prune --model ... --output <dir>/`.

Usage:
  python3 create_multi_model.py [--output PATH]
"""

import argparse
from pathlib import Path

try:
    import torch
    from safetensors.torch import save_file
except ImportError as exc:
    raise SystemExit(
        "Missing packages. Install: pip install torch safetensors"
    ) from exc

TENSORS = {
    "model.embed_tokens.weight": torch.randn(32000, 512, dtype=torch.float32),
    "model.layers.0.self_attn.q_proj.weight": torch.randn(512, 512, dtype=torch.float32),
    "model.layers.0.self_attn.k_proj.weight": torch.randn(512, 512, dtype=torch.float32),
    "model.layers.0.mlp.gate_proj.weight": torch.randn(2048, 512, dtype=torch.float32),
    "lm_head.weight": torch.randn(32000, 512, dtype=torch.float32),
}

parser = argparse.ArgumentParser(
    description="Generate a small multi-tensor .safetensors for tb-prune testing"
)
parser.add_argument(
    "--output", default="multi_model.safetensors", help="Output path"
)
args = parser.parse_args()

out_path = Path(args.output)
save_file(TENSORS, str(out_path))

total_elems = sum(t.numel() for t in TENSORS.values())
total_bytes = out_path.stat().st_size
print(f"Created '{out_path}': {len(TENSORS)} tensors, {total_elems:,} elements, {total_bytes:,} bytes")
