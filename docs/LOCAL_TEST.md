# Local Testing Guide (No GPU Required)

This guide describes how to build, test, and run Tensorbit Core on a laptop or
desktop without an NVIDIA GPU, using the CPU-only path via WSL/Ubuntu.

---

## Prerequisites

WSL2 Ubuntu 24.04 or bare Ubuntu 22.04+ with:

```bash
sudo apt update
sudo apt install -y build-essential cmake libeigen3-dev
```

Verify with:

```bash
bash verify_ubuntu.sh
```

All checks should show **[OK]** except nvcc (expected — you have no GPU).

---

## Step 1: Run the Unit Tests

```bash
cd /mnt/d/Dev/tensorbit_labs/tensorbit-core
bash tests/test_all.sh --skip-gpu --clean
```

Expected: **14/14 tests passed** (7 EHAP + 7 CORING).

If any fail, see `how_to_run.txt` for troubleshooting.

---

## Step 2: Build the CLI Tool

The test script builds `test_ehap` and `test_coring` but not `tb-prune`.
Build it separately:

```bash
cd build
cmake .. -DTENSORBIT_ENABLE_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build . --target tb-prune --parallel -j4
```

The binary lands at `build/bin/tb-prune`.

---

## Step 3: Run a Mock Pruning Job

The CLI can operate on randomly-generated "mock" tensors — no model download needed:

```bash
cd build
./bin/tb-prune --mock-size 16384 --sparsity 2:4 --output demo.tb
```

This runs the full pipeline:
1. Generate a random 16,384-element weight tensor
2. EHAP: accumulate Fisher → compute importance → select mask → apply
3. CORING: generate 2:4 mask → apply → redistribute
4. Save to `demo.tb`

**Expected output:**

```
[INFO] Tensorbit Core v0.2.0 — Pruning Pipeline
[INFO]   Output: demo.tb
[INFO] [Load] Generating mock weight tensor (16384 elements)
[INFO]   Total weights: 16384 (0.06 MB FP32)
[INFO] [EHAP] Computing importance scores...
[INFO]   EHAP pruned: 8192 weights (50.0%)
[INFO] [CORING] Applying N:M structured sparsity...
[INFO]   CORING (2/4): 8192 weights pruned (50.0% sparsity)
[INFO] [Save] Writing .tb file...
[INFO]   Saved to 'demo.tb' (68.06 KB)
[INFO] [Verify] .tb file valid: magic=0x31304254, v1, 2/4 sparsity, 16384 weights
[INFO] Done.
```

---

## Step 4: Verify the .tb File

The output is a valid Tensorbit Binary file:

```bash
ls -la demo.tb
# Expected: 73728 bytes (72 KB)

# Inspect the binary header:
xxd demo.tb | head -8
```

**Reading the header:**

| Bytes | Field | Expected | What it means |
|-------|-------|----------|---------------|
| `54 42 30 31` | magic | TB01 | Valid .tb file |
| `01 00 00 00` | version | 1 | Format version 1 |
| `0N 00 00 00` | nm_n | N | Kept per group |
| `0M 00 00 00` | nm_m | M | Group size |
| `XX XX 00 00` | num_weights | N_elements | Total weights |
| `YY YY 00 00` | num_mask_bytes | N_elements/M | Mask data size |

The file size should be exactly `4096 + N_elements * 4 + N_elements / M` bytes
(4096 header + FP32 weights + 1 byte per mask group).

For `--sparsity 2:4` with 16384 elements: 4096 + 65536 + 4096 = 73728 bytes. ✅

---

## Advanced Mock Options

| Flag | Description | Example |
|------|-------------|---------|
| `--mock-size N` | Elements in mock tensor | `--mock-size 65536` |
| `--sparsity N:M` | N:M pattern | `--sparsity 1:4` |
| `--strategy NAME` | Pruning strategy | `--strategy BlockOBS` |
| `--damping VAL` | Fisher damping | `--damping 0.02` |
| `--output PATH` | Output path | `--output my_model.tb` |

**BlockOBS (research-grade):**

```bash
./bin/tb-prune --mock-size 16384 --sparsity 2:4 --strategy BlockOBS --output demo_obs.tb
```

**Iterative pruning:**

```bash
./bin/tb-prune --mock-size 16384 --sparsity 2:4 --strategy Iterative --output demo_iter.tb
```

---

## Memory Limits on a Laptop

The mock tensor size is limited by your system RAM. On an 8 GB laptop:

| `--mock-size` | Weight memory | Total peak | Safe? |
|--------------|--------------|------------|-------|
| 4,096 | 16 KB | ~1 MB | Yes |
| 65,536 | 256 KB | ~5 MB | Yes |
| 1,048,576 | 4 MB | ~50 MB | Yes |
| 16,777,216 | 64 MB | ~500 MB | Yes |
| 67,108,864 | 256 MB | ~2 GB | Maybe |

For real LLM pruning (7B+ parameters), you need a cloud GPU — see `prompt3.txt`.

---

## Cleaning Up

```bash
# Delete build artifacts
cd /mnt/d/Dev/tensorbit_labs/tensorbit-core
rm -rf build/

# Delete generated .tb files
rm -f demo*.tb

# Fresh rebuild
bash tests/test_all.sh --skip-gpu --clean
```
