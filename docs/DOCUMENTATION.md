# tb-prune — User Manual

`tb-prune` is the command-line interface for Tensorbit Core. It loads model weights,
applies Hessian-aware pruning (EHAP) and N:M structured sparsity (CORING), then saves
the compressed result as a `.tb` (Tensorbit Binary) file.

---

## Prerequisites

**Ubuntu 22.04+, WSL2, or Debian-based Linux.** GCC 13+ required for C++20 support.

```bash
sudo apt update
sudo apt install -y build-essential cmake libeigen3-dev
```

Verify:

```bash
bash scripts/verify_ubuntu.sh
```

All checks should show **[OK]** except CUDA (`nvcc`) — that is expected unless you
have a GPU.

---

## Building (CPU-Only — Laptop / WSL)

```bash
cd tensorbit-core
bash tests/test_all.sh --skip-gpu --clean      # Run 14 unit tests

cd build
cmake .. -DTENSORBIT_ENABLE_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build . --target tb-prune --parallel -j4
```

The binary is at `build/bin/tb-prune`.

### Building with CUDA

See `docs/CLOUD.md` for cloud GPU setup with A100/H100.

---

## Quick Demo (Mock Mode — No Model Needed)

Mock mode generates random weights. No download required. Runs on any laptop.

```bash
cd build
./bin/tb-prune --mock-size 16384 --sparsity 2:4 --output demo.tb
```

Output:

```
[INFO] [Load] Generating mock weight tensor (16384 elements)
[INFO]   Total weights: 16384 (0.06 MB FP32)
[INFO] [EHAP] Computing importance scores...
[INFO]   EHAP pruned: 8192 weights (50.0%)
[INFO] [CORING] Applying N:M structured sparsity...
[INFO]   CORING (2/4): 8192 weights pruned (50.0% sparsity)
[INFO] [Save] Writing .tb file...
[INFO]   Saved to 'demo.tb' (72.00 KB)
[INFO] [Verify] .tb file valid: magic=0x31304254, v1, 2/4 sparsity, 16384 weights
[INFO] Done.
```

Verify the output:

```bash
ls -la demo.tb           # 73728 bytes
xxd demo.tb | head -8    # First bytes: 5442 3031 ("TB01")
```

---

## CLI Reference

```
tb-prune --model <PATH> [OPTIONS]
         --mock-size <N> [OPTIONS]
```

### Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--model PATH` | string | *(none)* | Path to a `.safetensors` model file |
| `--mock-size N` | integer | — | Random mock tensor of N elements |
| `--sparsity N:M` | pattern | `2:4` | N:M structured sparsity (N < M) |
| `--output PATH` | string | `output.tb` | Output path — a single `.tb` file for mock mode, or an output directory for real mode |
| `--method NAME` | string | `EHAP` | `EHAP` or `Magnitude` |
| `--strategy NAME` | string | `OneShot` | `OneShot`, `Iterative`, or `BlockOBS` |
| `--damping VAL` | float | `0.01` | Fisher damping λ |

### Strategy Guide

| Strategy | Speed | Quality | Best for |
|----------|-------|---------|----------|
| `OneShot` | Fast | Baseline | Quick runs, sparsity ≤ 80% |
| `Iterative` | Medium | Better | Sparsity > 80%, gradual schedule |
| `BlockOBS` | Slow | Best | Maximum accuracy, exact OBS updates |

### Mock Mode Examples

```bash
# Basic
./bin/tb-prune --mock-size 16384 --sparsity 2:4 --output demo.tb

# Larger tensor
./bin/tb-prune --mock-size 131072 --sparsity 2:4 --output demo_large.tb

# With BlockOBS
./bin/tb-prune --mock-size 16384 --sparsity 2:4 --strategy BlockOBS --output demo_obs.tb

# Magnitude only (no Hessian)
./bin/tb-prune --mock-size 16384 --method Magnitude --output demo_mag.tb
```

### Multi-Tensor Testing

```bash
# Install Python deps (use D: drive venv to save C: space):
python3 -m venv /mnt/d/venv/tensorbit
source /mnt/d/venv/tensorbit/bin/activate
pip install torch safetensors numpy packaging --quiet

# Run multi-tensor test:
bash tests/multi_tensor/test_multi.sh
# Expected: 5 .tb files produced, all with valid TB01 magic bytes
```

---

## Real Mode — Pruning .safetensors Models

### 1. Download a Model

```bash
source /opt/tensorbit-venv/bin/activate
python scripts/download_model.py \
    --repo meta-llama/Llama-2-7b-hf \
    --output ./models/llama-2-7b/ \
    --token hf_YOUR_TOKEN
```

### 2. Prune

```bash
./bin/tb-prune \
    --model ./models/llama-2-7b/model-00001-of-00002.safetensors \
    --sparsity 2:4 \
    --strategy BlockOBS \
    --output llama-2-7b-2of4.tb
```

---

## The .tb Format

| Section | Offset | Size | Content |
|---------|--------|------|---------|
| Header | 0 | 4096 | Magic (TB01), version, N:M ratios, offsets |
| Weights | 4096 | N × 4 | FP32 pruned weights (little-endian) |
| Masks | 4096 + N×4 | N/M | 1 byte per group, bit i = keep element i |

### Reading .tb Files

```cpp
#include "tensorbit/core/serialization.hpp"

TBReader reader;
auto hdr = reader.open("model.tb");
if (hdr) {
    std::vector<float> weights(hdr->num_weights);
    reader.read_weights<float>(weights);

    std::vector<uint8_t> masks(hdr->num_mask_bytes);
    reader.read_masks(masks);
    reader.close();
}
```

### Using the Safetensors Loader

```cpp
#include "tensorbit/core/loader.hpp"

loader::SafeTensorsFile sf;
sf.open("model.safetensors");

for (auto& t : sf.tensors())
    printf("  %s: %zu elements\n", t.name.c_str(), t.numel);

auto* meta = sf.find("model.layers.0.weight");
if (meta) {
    std::vector<std::byte> buf(meta->length);
    sf.read_tensor_data(*meta, buf);
}
```

---

## Memory Limits

### Mock Mode (Laptop)

| `--mock-size` | Peak memory |
|--------------|-------------|
| 4,096 | ~1 MB |
| 65,536 | ~5 MB |
| 1,048,576 | ~50 MB |
| 16,777,216 | ~500 MB |

### Real Mode (Cloud GPU)

| Model size | Weights (FP32) | Peak memory | GPU needed |
|-----------|---------------|-------------|------------|
| 7B params | 28 GB | ~58 GB | 1× A100-80GB |
| 13B params | 52 GB | ~104 GB | 2× A100-80GB |

See `docs/CLOUD.md` for cloud deployment.

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `cmake: command not found` | `sudo apt install cmake` |
| `Eigen3 not found` | `sudo apt install libeigen3-dev` |
| `cannot bind non-const lvalue reference to rvalue` | Store expressions in local variables before passing to `TENSORBIT_LOG_*` |
| `free(): invalid pointer` | Mask buffer size must be `N_elements / M`, not `N_elements` |
| `std::expected is only available from C++23` | Use `Result<T,E>`, not `std::expected` |
| `static assertion failed: TBHeader must be exactly 4096 bytes` | `reserved` array must be 4047 bytes |

---

## See Also

- `docs/CLOUD.md` — Cloud GPU deployment
- `docs/ARCHITECTURE.md` — Project internals
- `docs/EHAP.md` — EHAP algorithm
- `docs/CORING.md` — CORING algorithm
