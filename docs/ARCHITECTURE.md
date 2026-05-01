# Architecture Overview

## Directory Layout

```
tensorbit-core/
├── CMakeLists.txt                    # Build system (CUDA 12 + Eigen3)
├── .clang-format                     # C++20 code style (Google-based, 4-space indent)
├── .gitignore                        # Build artifacts, logs, .tb files, model weights
├── README.md                         # Project overview and usage
├── format.sh                         # Clang-format runner
│
├── include/
│   └── tensorbit/
│       └── core/
│           ├── common.hpp            # CUDA_CHECK, TENSORBIT_CHECK, thread-safe Logger
│           ├── tensor.hpp            # TensorDense<F>, FloatingPoint/TensorType concepts
│           ├── ehap.hpp              # EHAPPruner<F> — Hessian-aware importance scoring
│           ├── coring.hpp            # CORINGPruner<F> — N:M structured sparsity
│           ├── kernels.hpp           # CUDA kernel declarations (launch_*)
│           └── serialization.hpp     # TBWriter/TBReader — .tb binary format
│
├── src/
│   ├── main.cpp                      # CLI entry point (`tb-prune`)
│   ├── ehap.cpp                      # EHAPPruner<float>/<double> implementations
│   ├── coring.cpp                    # CORINGPruner<float>/<double> implementations
│   └── kernels.cu                    # CUDA kernels (fisher_diagonal, nm_mask, apply_mask)
│
├── tests/
│   ├── test_ehap.cpp                 # EHAP pruner unit tests
│   ├── test_coring.cpp               # CORING pruner unit tests
│   └── test_all.sh                   # Test runner (CMake ctest wrapper)
│
├── scripts/
│   ├── setup_cloud.sh                # Ubuntu 22.04 provisioning (CUDA 12, Eigen3, Python)
│   └── download_model.py             # HuggingFace .safetensors downloader
│
├── docs/
│   └── ARCHITECTURE.md               # This file
│
└── third_party/                      # Reserved for non-vcpkg dependencies
```

## Dependency Graph

```
tb-prune (executable)
  ├── tensorbit-core-cuda (static lib)     // src/kernels.cu
  │     ├── CUDA::cudart, CUDA::cublas
  │     └── tensorbit-core (static lib)
  │
  └── tensorbit-core (static lib)          // src/ehap.cpp, src/coring.cpp
        ├── Eigen3::Eigen                  // header-only linear algebra
        └── include/ headers               // common.hpp, tensor.hpp, ...
```

## Key Architecture Decisions

### 1. C++20 Concepts for Tensor Type Safety
`tensor.hpp` defines `FloatingPoint` and `TensorType` concepts. All pruner
templates (e.g., `EHAPPruner<F>`) are constrained by `FloatingPoint<F>`,
preventing accidental instantiation with integer or complex types.

### 2. Diagonal Fisher Approximation (O(N) memory)
Rather than storing the full O(N^2) Hessian, EHAP uses the empirical Fisher
diagonal: `F_ii = E[(∂L/∂w_i)^2]`. This is computed incrementally by `accumulate_fisher()`
and stored alongside weights at O(N) memory cost.

### 3. N:M Structured Sparsity via CORING
N:M sparsity maps directly to NVIDIA Ampere Sparse Tensor Cores. The CORING
pruner generates hardware-friendly masks that yield 2× throughput on A100/H100
GPUs. Mask generation is delegated to CUDA kernels in `kernels.cu`.

### 4. Explicit Template Instantiation
Both `EHAPPruner` and `CORINGPruner` use explicit instantiation (`extern template class`)
for `float` and `double` to control compile times and prevent implicit instantiation
from pulling in non-CUDA code paths.

### 5. Thread-Safe Logging
`common.hpp` provides a singleton `Logger` with severity levels and timestamped output.
All logging is mutex-guarded and safe to call from any thread.

## The .tb Binary Format

| Offset | Size    | Field           | Description                         |
|--------|---------|-----------------|-------------------------------------|
| 0      | 4       | magic           | `0x31304254` ("TB01" big-endian)    |
| 4      | 4       | version         | Format version (1)                  |
| 8      | 4       | nm_n            | N in N:M sparsity                   |
| 12     | 4       | nm_m            | M in N:M sparsity                   |
| 16     | 8       | num_weights     | Total weight elements               |
| 24     | 8       | num_masks       | Total mask bytes                    |
| 32     | 8       | weights_offset  | Byte offset to weight data          |
| 40     | 8       | masks_offset    | Byte offset to mask data            |
| 48     | 1       | precision       | 0=FP32, 1=FP16, 2=BF16             |
| 49     | 2047    | reserved        | Padding (future extensions)         |
| 4096   | varies  | weights_data    | Pruned weight buffer                |
| offset | varies  | masks_data      | Packed N:M bitmask buffer           |

## Roadmap: Implementation Phases

| Phase | Component            | Status     |
|-------|----------------------|------------|
| P1    | Boilerplate + build  | Done       |
| P2    | EHAP fisher kernel   | Stub       |
| P2    | CORING mask kernels  | Stub       |
| P2    | Safetensors parser   | Planned    |
| P3    | Multi-GPU support    | Planned    |
| P3    | FP16/BF16 precision  | Planned    |
| P4    | Inference runtime    | Planned    |

## License

Apache License 2.0 — Tensorbit Labs
