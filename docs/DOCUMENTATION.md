# tb-prune — User Manual

`tb-prune` is the command-line interface for Tensorbit Core. It loads model weights,
applies Hessian-aware pruning (EHAP) and N:M structured sparsity (CORING), then saves
the compressed result as a `.tb` (Tensorbit Binary) file.

---

## Prerequisites

- **Linux** (Ubuntu 22.04+) or **WSL2** (Windows 11 with Ubuntu 24.04)
- **GCC 12+** or **Clang 16+** with C++20 support
- **CMake 3.22+**
- **Eigen3** (header-only linear algebra library)
- **CUDA 12** (optional — required only for GPU acceleration on A100/H100)

Quick setup on WSL/Ubuntu:

```bash
sudo apt update
sudo apt install -y build-essential cmake libeigen3-dev
```

Verify your environment:

```bash
bash verify_ubuntu.sh
```

---

## Building

### CPU-only (laptop / CI)

```bash
cd tensorbit-core
mkdir -p build && cd build
cmake .. -DTENSORBIT_ENABLE_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build . --target tb-prune --parallel -j4
```

The binary is at `build/bin/tb-prune`.

### With CUDA (cloud GPU)

```bash
sudo ./scripts/setup_cloud.sh        # Install CUDA 12, Eigen3, Python
mkdir -p build && cd build
cmake .. -DEIGEN3_ROOT=/usr/local/include/eigen3 -GNinja -DCMAKE_BUILD_TYPE=Release
ninja
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
| `--mock-size N` | integer | — | Create a random mock tensor of N elements for testing |
| `--sparsity N:M` | pattern | `2:4` | N:M structured sparsity ratio (N < M) |
| `--output PATH` | string | `output.tb` | Path for the output `.tb` file |
| `--method NAME` | string | `EHAP` | Importance method: `EHAP` or `Magnitude` |
| `--strategy NAME` | string | `OneShot` | Pruning strategy: `OneShot`, `Iterative`, or `BlockOBS` |
| `--damping VAL` | float | `0.01` | Fisher diagonal damping factor λ |
| `--help, -h` | — | — | Print usage |
| `--version` | — | — | Print version |

### Strategy Details

| Strategy | Speed | Accuracy | When to use |
|----------|-------|----------|-------------|
| `OneShot` | Fastest | Baseline | Quick experiments, moderate sparsity (< 80%) |
| `Iterative` | Medium | Better | Higher sparsity, gradual schedule avoids accuracy cliffs |
| `BlockOBS` | Slow | Best | Maximum accuracy, exact OBS weight compensation |

---

## Mock Mode — Testing Without a Real Model

Mock mode generates a random weight tensor and mock gradients. No model download
required. Useful for development, CI, and verifying the pipeline.

```bash
# Basic mock run
./bin/tb-prune --mock-size 16384 --sparsity 2:4 --output demo.tb

# Larger mock tensor
./bin/tb-prune --mock-size 131072 --sparsity 2:4 --output demo_large.tb

# With BlockOBS strategy
./bin/tb-prune --mock-size 16384 --sparsity 2:4 --strategy BlockOBS --output demo_obs.tb

# Magnitude-only pruning (no Hessian)
./bin/tb-prune --mock-size 16384 --method Magnitude --output demo_mag.tb
```

Expected output for a mock run:

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
[INFO]   Saved to 'demo.tb' (72.00 KB)
[INFO] [Verify] .tb file valid: magic=0x31304254, v1, 2/4 sparsity, 16384 weights
[INFO] Done.
```

---

## Real Mode — Pruning a Downloaded Model

### 1. Download Model Weights

```bash
source /opt/tensorbit-venv/bin/activate
python scripts/download_model.py \
    --repo meta-llama/Llama-2-7b-hf \
    --output ./models/llama-2-7b/ \
    --token hf_YOUR_TOKEN
```

### 2. Prune a Single Safetensors Shard

```bash
./bin/tb-prune \
    --model ./models/llama-2-7b/model-00001-of-00002.safetensors \
    --sparsity 2:4 \
    --strategy BlockOBS \
    --output llama-2-7b-2of4.tb
```

### 3. Prune with Magnitude Fallback

```bash
./bin/tb-prune \
    --model ./models/llama-2-7b/model-00001-of-00002.safetensors \
    --sparsity 1:4 \
    --method Magnitude \
    --output llama-2-7b-1of4.tb
```

> **Note:** Magnitude mode skips Fisher accumulation and Hessian computation.
> It is faster but less accurate than EHAP at high sparsity.

### 4. Using the Safetensors Loader Programmatically

```cpp
#include "tensorbit/core/loader.hpp"

loader::SafeTensorsFile sf;
sf.open("model.safetensors");

// List all tensors
for (auto& t : sf.tensors()) {
    printf("  %s: [", t.name.c_str());
    for (auto d : t.shape) printf("%zu ", d);
    printf("] (%zu elements)\n", t.numel);
}

// Read a specific tensor
auto* meta = sf.find("model.layers.0.weight");
if (meta) {
    std::vector<std::byte> buf(meta->length);
    sf.read_tensor_data(*meta, buf);
}
```

---

## Output: The .tb Format

The `.tb` file is a self-contained binary with three sections:

| Section | Offset | Size | Content |
|---------|--------|------|---------|
| Header | 0 | 4096 | Magic, version, N:M ratios, element counts, byte offsets |
| Weights | 4096 | N × sizeof(float) | Pruned weight data (FP32, little-endian) |
| Masks | 4096 + N×4 | N/M bytes | N:M bitmask (1 byte per group, bit i = keep element i) |

### Verifying a .tb File

```bash
ls -la output.tb
# Expected: 73728 bytes for 16384-element 2:4 model

xxd output.tb | head -8
# First bytes should be: 5442 3031 ("TB01")
```

The program also performs round-trip verification after each write — the log line
`[Verify] .tb file valid` confirms the header was read back correctly.

### Reading a .tb Programmatically

```cpp
#include "tensorbit/core/serialization.hpp"

TBReader reader;
auto hdr = reader.open("model.tb");
if (hdr) {
    printf("%d weights, %d/%d sparsity\n",
           hdr->num_weights, hdr->nm_n, hdr->nm_m);

    std::vector<float> weights(hdr->num_weights);
    reader.read_weights<float>(weights);

    std::vector<uint8_t> masks(hdr->num_mask_bytes);
    reader.read_masks(masks);

    reader.close();
}
```

---

## Configuration Reference

### EHAPConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `damping` | float | 0.01 | Fisher diagonal damping λ |
| `use_diagonal_fisher` | bool | true | Enable Fisher accumulation |
| `sparsity_ratio` | float | 0.5 | Fraction of weights to **retain** (0–1) |
| `ema_decay` | float | 0.99 | Fisher EMA decay β |
| `importance_mode` | enum | kOBD | kOBD / kOBS / kNormalized |
| `prune_strategy` | enum | kOneShot | kOneShot / kIterative / kBlockOBS |
| `prune_rounds` | size_t | 5 | Rounds for iterative pruning |
| `compensation_mode` | enum | kNone | kNone / kBias / kRedist |
| `obs_block_size` | size_t | 128 | Block size for BlockOBS |
| `gradient_history_size` | size_t | 4 | Gradient snapshots for low-rank Hessian |

### CORINGConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `N` | int | 2 | Elements kept per group |
| `M` | int | 4 | Group size |
| `mask_strategy` | enum | kTopN | kTopN / kOptimal / kIterative |
| `redist_mode` | enum | kNone | kNone / kProportional / kUniform |
| `iterative_rounds` | int | 3 | Rounds for iterative swap-refine |
| `hardware_aware_layout` | bool | false | 2:4 Ampere GEMM alignment |

---

## Memory Limits

### Mock Mode (laptop)

| `--mock-size` | Peak memory | Status |
|--------------|-------------|--------|
| 4,096 | ~1 MB | Always safe |
| 65,536 | ~5 MB | Always safe |
| 1,048,576 | ~50 MB | Safe on 8 GB |
| 16,777,216 | ~500 MB | Safe on 8 GB |
| 67,108,864 | ~2 GB | May work |

### Real Mode (requires cloud GPU for LLMs)

| Model size | FP32 weights | Fisher + buffers | Total peak | GPU needed |
|-----------|-------------|-----------------|------------|-------------|
| 7B params | 28 GB | 28 GB | ~58 GB | 1× A100-80GB |
| 13B params | 52 GB | 52 GB | ~104 GB | 2× A100-80GB |
| 70B params | 280 GB | 280 GB | ~560 GB | 8× A100-80GB |

---

## Troubleshooting

### `cmake: command not found`
```bash
sudo apt install cmake
```

### `Eigen3 not found`
```bash
sudo apt install libeigen3-dev
# or pass: cmake .. -DEIGEN3_ROOT=/path/to/eigen3
```

### `cannot bind non-const lvalue reference to rvalue`
All arguments to `TENSORBIT_LOG_*` macros must be named variables, not arithmetic
expressions. Extract computed values to local variables first:
```cpp
auto pct = 100.0 * pruned / total;  // lvalue
TENSORBIT_LOG_INFO("{} %", pct);
```

### `free(): invalid pointer`
Buffer overflow — check that mask buffer is sized for number of groups
(N_elements / M), not N_elements.

### `std::expected is only available from C++23`
Tensorbit Core uses a custom `Result<T,E>` type. Ensure all code references
`Result` not `std::expected`.

### `static assertion failed: TBHeader must be exactly 4096 bytes`
The `reserved` array in TBHeader needs exactly 4047 bytes to pad the struct
to 4096 total when `#pragma pack(1)` is used.

---

## See Also

- `docs/ARCHITECTURE.md` — Project structure and design decisions
- `docs/ALGORITHMS.md` — High-level algorithm overview
- `docs/EHAP.md` — Complete EHAP mathematical exposition
- `docs/CORING.md` — Complete CORING mathematical exposition
- `docs/LOCAL_TEST.md` — Step-by-step local testing guide
- `how_to_run.txt` — Build and run reference
- `prompt3.txt` — Implementation summary and cloud deployment guide
