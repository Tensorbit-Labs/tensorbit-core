#!/usr/bin/env python3
"""download_model.py — Fetch model weights from Hugging Face Hub.

Downloads .safetensors weights for a given model and saves them to
the local filesystem for use with `tb-prune`.

Usage:
  python scripts/download_model.py --repo meta-llama/Llama-2-7b-hf \\
                                    --output ./models/llama-2-7b/ \\
                                    [--token hf_xxx] \\
                                    [--quantize {fp32,fp16,bf16}]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional Import Guard
# ---------------------------------------------------------------------------

try:
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    from safetensors import safe_open
except ImportError as exc:
    logger.fatal(
        "Missing required packages. Install with: "
        "pip install huggingface_hub safetensors torch"
    )
    raise SystemExit(1) from exc

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download HuggingFace model weights for Tensorbit Core pruning."
    )
    parser.add_argument(
        "--repo",
        required=True,
        help="HuggingFace model repository (e.g. meta-llama/Llama-2-7b-hf)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Destination directory for downloaded files",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN", None),
        help="HuggingFace API token (or set HF_TOKEN env var)",
    )
    parser.add_argument(
        "--quantize",
        choices=["fp32", "fp16", "bf16"],
        default="fp16",
        help="Preferred precision variant to download (default: fp16)",
    )
    parser.add_argument(
        "--max-shard-size",
        type=str,
        default=None,
        help="Filter shards by approximate file size (e.g. '2GB')",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="List available files without downloading",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be downloaded without fetching",
    )
    return parser.parse_args(argv)


def download(repo_id: str,
             output_dir: Path,
             token: Optional[str],
             quantize: str,
             max_shard_size: Optional[str],
             list_only: bool,
             dry_run: bool) -> int:
    """Download .safetensors shards from a HuggingFace repository."""

    output_dir.mkdir(parents=True, exist_ok=True)
    api = HfApi(token=token)

    # --- List repository files ---
    logger.info("Listing files for '%s'...", repo_id)
    try:
        repo_info = api.repo_info(repo_id, files_metadata=True)
    except Exception as exc:
        logger.fatal("Failed to fetch repo info for '%s': %s", repo_id, exc)
        return 1

    # Collect .safetensors files
    safetensors_files = []
    for sibling in getattr(repo_info, "siblings", []):
        if not sibling.rfilename.endswith(".safetensors"):
            continue
        safetensors_files.append(sibling.rfilename)

    if not safetensors_files:
        logger.warning("No .safetensors files found in '%s'.", repo_id)
        # Try listing via the repo listing anyway
        files = api.list_repo_files(repo_id, token=token)
        for f in files:
            if f.endswith(".safetensors"):
                safetensors_files.append(f)

    if not safetensors_files:
        logger.error("Repository '%s' has no .safetensors files.", repo_id)
        return 1

    logger.info("Found %d .safetensors file(s):", len(safetensors_files))
    for f in safetensors_files:
        logger.info("  - %s", f)

    if list_only:
        return 0

    # --- Determine preferred files ---
    # Prefer files matching the requested quantize mode.
    preferred: list[str] = []
    for f in safetensors_files:
        if quantize in f.lower():
            preferred.append(f)

    if not preferred:
        logger.info("No '%s' variant found. Downloading all .safetensors files.", quantize)
        preferred = safetensors_files

    logger.info("Will download %d file(s) matching '%s':", len(preferred), quantize)
    for f in preferred:
        logger.info("  -> %s", f)

    if dry_run:
        logger.info("Dry run — no files downloaded.")
        return 0

    # --- Download ---
    logger.info("Downloading to '%s'...", output_dir)
    for filename in preferred:
        target = output_dir / filename
        if target.exists():
            logger.info("  SKIP %s (already exists)", filename)
            continue
        try:
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=output_dir,
                local_dir_use_symlinks=False,
                token=token,
                resume_download=True,
            )
            logger.info("  OK   %s", filename)
        except Exception as exc:
            logger.error("  FAIL %s: %s", filename, exc)
            return 1

    # --- Validate one shard ---
    shard_path = output_dir / preferred[0]
    try:
        with safe_open(str(shard_path), framework="pt") as f:
            keys = list(f.keys())
            logger.info(
                "Validated '%s': %d tensors (e.g. '%s')",
                shard_path.name, len(keys),
                keys[0] if keys else "<none>",
            )
    except Exception as exc:
        logger.error("Failed to validate '%s': %s", shard_path, exc)
        return 1

    logger.info("=== Download complete: %d file(s) in %s ===",
                len(preferred), output_dir)
    return 0


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    sys.exit(download(
        repo_id=args.repo,
        output_dir=Path(args.output),
        token=args.token,
        quantize=args.quantize,
        max_shard_size=args.max_shard_size,
        list_only=args.list_only,
        dry_run=args.dry_run,
    ))
