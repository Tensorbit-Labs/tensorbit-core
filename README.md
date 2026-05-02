# Tensorbit Core

High-performance C++20/CUDA 12 engine for **Hessian-aware structured pruning**
of large language models and vision transformers. Reads industry-standard model
weights, identifies load-bearing parameters using second-order gradient
information, enforces hardware-friendly N:M sparsity patterns, and serializes
the result to a compact `.tb` binary format ready for high-speed inference.

Part of the **Tensorbit Labs P-D-Q pipeline**:
```
.safetensors → [tensorbit-core: Prune] → .tb → [tensorbit-distill] → [tensorbit-quant] → [tensorbit-run]
```

## Quick Start

```bash
# Install prerequisites (Ubuntu / WSL2)
sudo apt install -y build-essential cmake libeigen3-dev

# Build and test
bash tests/test_all.sh --skip-gpu --clean

# Run a demo pruning job (no GPU needed)
cd build
cmake .. -DTENSORBIT_ENABLE_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build . --target tb-prune --parallel -j4
./bin/tb-prune --mock-size 16384 --sparsity 2:4 --output demo.tb
```

## Key Capabilities

| Feature | Description |
|---------|-------------|
| **EHAP** | Fisher-based EMA Hessian approximation, OBD/OBS/Normalized importance scoring, iterative cubic-schedule pruning, blockwise exact OBS with Woodbury inverse and gradient-covariance low-rank Hessian |
| **CORING** | N:M structured sparsity (2:4 optimal for Ampere Sparse Tensor Cores), top-N / optimal C(M,N) / iterative swap-refine mask selection, absolute-magnitude redistribution |
| **.tb format** | 4096-byte header, FP32/FP16/BF16 weight storage, packed N:M bitmasks, round-trip verification |
| **Safetensors** | Header-only parser for HuggingFace models (F32/F16/BF16/I64) |
| **GPU** | 6 CUDA kernels optimized for A100 (SM80) / H100 (SM90), CPU-only fallback |
| **C++20** | Custom `Result<T,E>` type, `FloatingPoint` concepts, `std::span` CLI parsing, non-template Logger with `vformat` |

## Tech Stack

C++20, CUDA 12, Eigen3, GCC 13+ / Clang 16+ / MSVC 2022

## Documentation

| Document | Purpose |
|----------|---------|
| [`docs/DOCUMENTATION.md`](docs/DOCUMENTATION.md) | Complete user manual — installation, building, CLI flags, examples, troubleshooting |
| [`docs/CLOUD.md`](docs/CLOUD.md) | Cloud GPU deployment — A100/H100 setup, providers, pruning real LLMs |
| [`docs/EHAP.md`](docs/EHAP.md) | EHAP algorithm — mathematical derivation, all equations, 10 references |
| [`docs/CORING.md`](docs/CORING.md) | CORING algorithm — N:M sparsity design, mask strategies, 8 references |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Project internals — directory layout, dependency graph, design decisions, format specs |
| [`docs/ALGORITHMS.md`](docs/ALGORITHMS.md) | High-level algorithm overview and pipeline walkthrough |

## License

Apache License 2.0 — Tensorbit Labs
