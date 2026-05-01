# Architecture Overview

## Project Identity

**tensorbit-core** is the high-performance C++20/CUDA 12 "Surgical Engine" for
Hessian-aware structured pruning. It is stage one of the Tensorbit Labs
P-D-Q pipeline (Prune → Distill → Quantize).

---

## Directory Layout

```
tensorbit-core/
├── CMakeLists.txt                    # Build system (CUDA 12 SM80/SM90, Eigen3, C++20)
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
│           │                          #   Host/device allocation, to_device(), to_host()
│           ├── ehap.hpp              # EHAPPruner<F> — Hessian-aware importance scoring
│           ├── coring.hpp            # CORINGPruner<F> — N:M structured sparsity
│           ├── kernels.hpp           # CUDA kernel launch declarations (7 launches)
│           └── serialization.hpp     # TBWriter/TBReader — .tb binary format (stubs)
│
├── src/
│   ├── main.cpp                      # CLI entry point (`tb-prune`) with std::span
│   ├── ehap.cpp                      # EHAPPruner<float>/<double> full implementation
│   │                                  #   GPU/CPU dispatch for accumulate, importance, threshold
│   ├── coring.cpp                    # CORINGPruner<float>/<double> full implementation
│   │                                  #   GPU 2:4/generic paths, CPU fallback, analytical counts
│   └── kernels.cu                    # 6 CUDA kernels: fisher_diagonal, fisher_accumulate,
│                                      #   ehap_importance, nm_mask_2_4, nm_mask_generic, apply_mask
├── tests/
│   ├── test_ehap.cpp                 # EHAP pruner unit tests (7 cases)
│   ├── test_coring.cpp               # CORING pruner unit tests (7 cases)
│   └── test_all.sh                   # Test runner (CMake ctest wrapper)
│
├── scripts/
│   ├── setup_cloud.sh                # Ubuntu 22.04 provisioning (CUDA 12, Eigen3, Python)
│   └── download_model.py             # HuggingFace .safetensors downloader
│
├── docs/
│   ├── ARCHITECTURE.md               # This file
│   └── ALGORITHMS.md                 # Full mathematical exposition (Taylor, Fisher, roofline)
│
└── third_party/                      # Reserved for non-vcpkg dependencies
```

---

## Dependency Graph

```
tb-prune (executable)
  ├── tensorbit-core-cuda (static lib)          // src/kernels.cu — 6 GPU kernels
  │     ├── CUDA::cudart                        // cudaMalloc, cudaFree, cudaMemcpy
  │     ├── CUDA::cublas                        // Reserved for future cuBLAS integration
  │     └── tensorbit-core (static lib)         // Host headers + pruner implementations
  │
  └── tensorbit-core (static lib)               // src/ehap.cpp, src/coring.cpp
        ├── Eigen3::Eigen                       // Header-only linear algebra (reserved)
        ├── include/tensorbit/core/common.hpp   // Logging, error-checking macros
        ├── include/tensorbit/core/tensor.hpp   // TensorDense<F>, concepts, device memory
        ├── include/tensorbit/core/ehap.hpp     // EHAPPruner interface
        ├── include/tensorbit/core/coring.hpp   // CORINGPruner interface
        ├── include/tensorbit/core/kernels.hpp  // CUDA kernel launch declarations
        └── include/tensorbit/core/serialization.hpp  // .tb format stubs
```

### Header Dependency Order (acyclic)

```
common.hpp                          (independent — macros, Logger)
    ↑
tensor.hpp                          (includes common.hpp for CUDA_CHECK, TENSORBIT_CHECK)
    ↑
ehap.hpp  coring.hpp  kernels.hpp   (each includes tensor.hpp for TensorDense<F>)
    ↑           ↑           ↑
ehap.cpp   coring.cpp   kernels.cu  (implementation files)
    ↘           ↙           ↓
  main.cpp                  tensorbit-core-cuda
```

---

## Key Architecture Decisions

### 1. C++20 Concepts for Tensor Type Safety

`tensor.hpp` defines two core concepts:

```cpp
template<typename T>
concept FloatingPoint = std::is_floating_point_v<T>;

template<typename T>
concept TensorType = requires(T t) {
    typename T::value_type;
    { t.data() }  -> std::same_as<typename T::value_type*>;
    { t.size() }  -> std::convertible_to<std::size_t>;
    { t.shape() } -> std::convertible_to<std::span<const std::size_t>>;
    { t.rank() }  -> std::convertible_to<std::size_t>;
};
```

All pruner templates (`EHAPPruner<F>`, `CORINGPruner<F>`) are constrained by
`FloatingPoint<F>`, preventing accidental instantiation with integer or complex
types. The `TensorType` concept defines the minimum contract any tensor class
must satisfy to interoperate with the pruning pipeline — enabling future
zero-copy wrappers around PyTorch tensors or NumPy arrays.

### 2. TensorDense<F> — Dual Device Memory Management

The `TensorDense<F>` class owns its buffer on either host or device through a
`std::unique_ptr<F[], void(*)(F*)>` with compile-time selectable deleters:

| Allocator | Deallocator | Guard |
|-----------|-------------|-------|
| `new F[n]()` | `delete[] ptr` | Always available |
| `cudaMalloc` | `cudaFree(ptr)` | `#ifdef __CUDACC__` |

**Memory transfer** is explicit via two methods:
- `to_device()` — `cudaMemcpy(host→device)`, returns new device-owned `TensorDense`.
- `to_host()` — `cudaMemcpy(device→host)`, returns new host-owned `TensorDense`.

These are used by the CORING pruner's CPU fallback path: when double-precision
tensors reside on the GPU (where only float kernels exist), the implementation
transparently copies to host, processes on CPU, and copies back.

**Move semantics** are implemented (copy is deleted) — tensors can be efficiently
returned from functions without double-free risk.

**Maximum rank** is 8 — covers transformer tensors up to
`[batch × seq_len × num_heads × head_dim × ...]`.

### 3. Diagonal Fisher Approximation — O(N) Memory

Rather than storing the full O(N²) Hessian, EHAP uses the **empirical Fisher
Information diagonal**:

$$\boxed{F_{ii} = \mathbb{E}_{x \sim D}\left[\left(\frac{\partial\mathcal{L}}{\partial w_i}\right)^2\right]}$$

This is accumulated incrementally via `accumulate_fisher(gradients, alpha)`:

$$F_{ii} \leftarrow F_{ii} + \alpha \cdot g_i^2$$

**GPU path**: `fisher_accumulate_kernel` — 1 thread per element, uses `fmaf()`
fused multiply-add for 1-ULP precision. Zero shared memory, ~100% occupancy.

**CPU path**: Element-wise `__restrict__` loop (autovectorizable by GCC/Clang 12+).

The importance score then couples magnitude with curvature:

$$\boxed{s_i = w_i^2 \cdot (F_{ii} + \lambda)}$$

where λ is the damping factor (default 0.01) for numerical stability.

### 4. EHAP Three-Stage Pipeline

| Stage | Method | GPU Kernel | CPU Fallback |
|-------|--------|------------|--------------|
| **Accumulate** | `accumulate_fisher(grad, α)` | `fisher_accumulate_kernel` | `__restrict__` loop |
| **Compute** | `compute_importance(w, out)` | `ehap_importance_kernel` | Direct loop |
| **Select** | `select_pruning_mask(imp, mask)` | *(host-only)* | `std::nth_element` O(N) |

The **selection stage** is host-only by design — sorting/partitioning on GPU
requires complex warp-level reductions and is dominated by PCIe transfer cost
for mask export anyway. `std::nth_element` finds the (1−sparsity_ratio)·N
percentile in O(N) time with cache-friendly memory access.

### 5. N:M Structured Sparsity — Dual-Path CORING Engine

CORING enforces hardware-friendly patterns for NVIDIA Ampere Sparse Tensor
Cores (instruction `mma.sp`):

| Path | N:M Pattern | Kernel | Thread Model | Shared Memory |
|------|------------|--------|-------------|---------------|
| **Fast** | 2:4 (Ampere native) | `nm_mask_2_4_kernel` | 1 thread/group | 0 bytes |
| **Generic** | Any N:M, M ≤ 32 | `nm_mask_generic_kernel` | M threads/group | ~128 bytes |

**2:4 fast path** — One thread loads 4 importance values into registers, finds
the top-2 magnitudes via a fixed comparison tree (fully unrolled, branchless
after PTX compilation), and writes a packed mask byte. Near-100% theoretical
occupancy on SM80/SM90.

**Generic N:M path** — A thread block of size M processes one group. Each thread
holds one element. Cooperative ranking via shared memory: each thread counts how
many others have strictly higher value. Tie-breaking is deterministic (lower
index wins). Thread 0 assembles the mask byte from the rank array.

**CPU fallback** — For double-precision tensors or non-CUDA builds, each group
is processed via `std::nth_element` operating on a `vector<pair<F, int>>` of
size M. Since M ≤ 32, this is negligible overhead.

**Mask format** — Packed as 1 byte per group:
```
  Byte g:  bit 0 = keep element (g·M + 0)?
           bit 1 = keep element (g·M + 1)?
           ...
           bit (M-1) = keep element (g·M + M−1)?
```

**Analytical pruned count** — Since `validate_config` ensures the tensor size is
divisible by M, the number of pruned weights is exact:

$$N_{\text{pruned}} = \frac{N_{\text{elements}}}{M} \cdot (M - N)$$

This eliminates the need for atomic counters in the CUDA mask kernel.

### 6. GPU/CPU Device Dispatch Strategy

Every pruner method checks `tensor.device()` at runtime and routes accordingly:

```
if (tensor.device() == DeviceLocation::kDevice && use_cuda)
    → launch CUDA kernel + CUDA_SYNC_CHECK()
else
    → CPU loop (host-resident data or CUDA unavailable)
```

Double-precision GPU tensors fall back to CPU transparently:
```
weights (device, double) → to_host() → CPU processing → cudaMemcpy back
```

This provides a uniform API regardless of whether CUDA is available, while
still benefiting from GPU acceleration when possible.

### 7. Explicit Template Instantiation

Both `EHAPPruner` and `CORINGPruner` use explicit instantiation (`extern template
class`) for `float` and `double`. This:

- **Controls compile times** — specializations are compiled once in each `.cpp`.
- **Prevents ODR violations** — one definition rule across translation units.
- **Isolates CUDA dependencies** — `kernels.cu` is linked separately; the `.cpp`
  files only call host launch wrappers declared in `kernels.hpp`.

### 8. Thread-Safe Logging

`common.hpp` provides a singleton `Logger` with 6 severity levels (`kTrace` →
`kFatal`), timestamped output via `std::format` (C++20), and `std::mutex`-guarded
write access. Convenience macros (`TENSORBIT_LOG_INFO`, etc.) include
`std::source_location` for automatic file:line attribution.

---

## The .tb Binary Format

| Offset | Size    | Field           | Description                                 |
|--------|---------|-----------------|---------------------------------------------|
| 0      | 4       | magic           | `0x31304254` ("TB01" big-endian)            |
| 4      | 4       | version         | Format version (1)                          |
| 8      | 4       | nm_n            | N in N:M sparsity pattern                   |
| 12     | 4       | nm_m            | M in N:M sparsity pattern                   |
| 16     | 8       | num_weights     | Total weight elements (dense count)         |
| 24     | 8       | num_masks       | Total mask bytes (num_weights / M groups)   |
| 32     | 8       | weights_offset  | Byte offset to start of pruned weight data  |
| 40     | 8       | masks_offset    | Byte offset to start of packed mask data    |
| 48     | 1       | precision       | 0=FP32, 1=FP16, 2=BF16                     |
| 49     | 2047    | reserved        | Padding for future extensions               |
| 4096   | varies  | weights_data    | Pruned weight buffer (dense, pruned=0.0)    |
| offset | varies  | masks_data      | Packed N:M bitmasks (1 byte per M-sized group) |

### Mask Packing Convention

Each group of M weights consumes exactly 1 byte in `masks_data`:
```
Group g at masks_data[g]:
  Bit 0 → weight[g·M + 0] is kept (1) or pruned (0)
  Bit 1 → weight[g·M + 1] is kept (1) or pruned (0)
  ...
  Bit (M-1) → weight[g·M + M−1] is kept (1) or pruned (0)
```

For M ≤ 8, one byte per group. For M = 16 or M = 32, bytes are packed
accordingly (2 or 4 bytes per group respectively). The `.tb` reader uses
`nm_m` from the header to determine the unpacking stride.

---

## CUDA Kernel Reference

| Kernel | Grid | BlockDim | Shared Memory | Compute Intensity | Bound |
|--------|------|----------|---------------|-------------------|-------|
| `fisher_diagonal_kernel` | ⌈N/256⌉ | 256 | 0 B | 2 FLOP/elem | Memory (HBM) |
| `fisher_accumulate_kernel` | ⌈N/256⌉ | 256 | 0 B | 2 FLOP/elem | Memory (HBM) |
| `ehap_importance_kernel` | ⌈N/256⌉ | 256 | 0 B | 3 FLOP/elem | Memory (HBM) |
| `nm_mask_2_4_kernel` | ⌈N₄/256⌉ | 256 | 0 B | 4 FLOP/elem | Compute (ALU) |
| `nm_mask_generic_kernel` | N/M | M | ~128 B | O(M²)/elem | Compute (ALU) |
| `apply_mask_kernel` | ⌈N/256⌉ | 256 | 0 B | 0 FLOP/elem | Memory (HBM) |

All grid/block dimensions are computed by host launch wrappers in
`kernels.hpp`. Kernels use `std::size_t` for element counts to handle
models with >2³¹ parameters.

### Fisher-diagonal kernel detail

`fisher_diagonal_kernel` sums over batch dimension B:
$$\text{fisher\_diag}[i] = \sum_{b=0}^{B-1} \text{grad}[b \cdot N + i]^2$$

Used for initial Fisher buffer population from a full batch of gradients.
Subsequent accumulation uses `fisher_accumulate_kernel` (element-wise +=).

---

## End-to-End Data Flow

```
  HuggingFace Hub
       │
       │ download_model.py
       ▼
  .safetensors ──────────────────────────────────────────┐
  (dense FP32/FP16/BF16 weights)                         │
       │                                                 │
       │ tb-prune --model <path> --sparsity 2:4          │
       ▼                                                 │
  ┌──────────────────────────────────────────┐           │
  │              EHAPPruner                  │           │
  │                                          │           │
  │  accumulate_fisher(gradients, α)         │           │
  │    ├─ GPU: fisher_accumulate_kernel      │           │
  │    └─ CPU: __restrict__ loop             │           │
  │                                          │           │
  │  compute_importance(weights, imp_out)    │           │
  │    ├─ GPU: ehap_importance_kernel        │           │
  │    │   s[i] = w[i]² · (F[i] + λ)        │           │
  │    └─ CPU: direct loop                   │           │
  │                                          │           │
  │  select_pruning_mask(imp, mask)          │           │
  │    └─ CPU: std::nth_element O(N)         │           │
  │        threshold = percentile(r)          │           │
  │        mask[i] = (imp[i] >= threshold)    │           │
  └──────────────┬───────────────────────────┘           │
                 │ importance scores                    │ (Phase 3:
                 ▼                                     │  Safetensors
  ┌──────────────────────────────────────────┐          │  parser)
  │             CORINGPruner                 │          │
  │                                          │          │
  │  validate_config(importance.size())      │          │
  │    ├─ N < M, M is power-of-2            │          │
  │    └─ Size divisible by M                │          │
  │                                          │          │
  │  generate_nm_mask(importance, mask_out)  │          │
  │    ├─ 2:4 → nm_mask_2_4_kernel (fast)   │          │
  │    ├─ N:M → nm_mask_generic_kernel       │          │
  │    └─ CPU: nth_element per group         │          │
  │                                          │          │
  │  apply_mask(weights, mask)               │          │
  │    ├─ GPU: apply_mask_kernel             │          │
  │    └─ CPU: bit-test + zero per group     │          │
  │                                          │          │
  │  N_pruned = (N_elems/M) × (M−N)          │          │
  └──────────────┬───────────────────────────┘          │
                 │ pruned weights + N:M masks           │
                 ▼                                     │
  ┌──────────────────────────────────────────┐          │
  │        TBWriter (serialization.hpp)      │          │
  │                                          │          │
  │  write(pruned_weights, masks, N, M)      │          │
  │    ├─ Header: magic + version + metadata │          │
  │    ├─ Weights blob: dense float buffer   │          │
  │    └─ Masks blob: packed bitmask buffer  │          │
  └──────────────┬───────────────────────────┘          │
                 │                                     │
                 ▼                                     ▼
            output.tb                          (future) inference
         (standalone file,                  runtime → TBReader →
        ready for inference)               sparse matmul on GPU)
```

---

## Memory Budget (7B Parameter Model, 2:4 Sparsity)

| Component | Precision | Size |
|-----------|-----------|------|
| Original weights | FP32 | 28 GB |
| Fisher diagonal (during pruning) | FP32 | 28 GB |
| Importance scores (temporary) | FP32 | 28 GB |
| EHAP mask (uint8, 1 B/elem) | uint8 | 7 GB |
| N:M mask (packed, N/M bytes) | uint8 | 1.75 GB |
| Pruned weights (.tb output) | FP32 | 28 GB |
| **Peak pruning memory** | | **~58 GB** |
| Minimum GPU required | | A100-80GB |

Memory is freed incrementally: after mask selection, importance scores are
released. After N:M mask generation, the EHAP mask is released. The Fisher
diagonal is kept until pruning completes, then freed via `reset()`.

---

## Roadmap: Implementation Phases

| Phase | Component                   | Status     | Notes |
|-------|-----------------------------|------------|-------|
| P1    | Boilerplate + build system  | ✅ Done    | CMake, headers, CLI, Logger, tests |
| P2    | EHAP fisher kernel          | ✅ Done    | `fisher_accumulate_kernel`, `fisher_diagonal_kernel` |
| P2    | EHAP importance + threshold | ✅ Done    | `ehap_importance_kernel`, `select_pruning_mask` |
| P2    | CORING 2:4 kernel           | ✅ Done    | Warp-local register-only `nm_mask_2_4_kernel` |
| P2    | CORING generic N:M kernel   | ✅ Done    | Shared-memory `nm_mask_generic_kernel` (M ≤ 32) |
| P2    | Mask application kernel     | ✅ Done    | `apply_mask_kernel` with analytical pruned count |
| P2    | Device memory management    | ✅ Done    | `cudaMalloc`/`cudaFree`, `to_device()`/`to_host()` |
| P3    | Safetensors parser          | 🔜 Planned | Read HuggingFace models for end-to-end CLI pruning |
| P3    | .tb serialization layer     | 🔜 Planned | `TBWriter`/`TBReader` full implementation |
| P3    | Multi-stream parallelism    | 🔜 Planned | CUDA stream pools for concurrent layer processing |
| P3    | FP16/BF16 precision         | 🔜 Planned | `__half`/`__nv_bfloat16` kernels for reduced memory |
| P4    | cuSPARSELt integration      | 🔜 Planned | Direct 2:4 matmul via `cusparseLtMatmul()` |
| P4    | Inference runtime           | 🔜 Planned | Standalone .tb loader + sparse GEMM executor |
| P4    | Multi-GPU sharding          | 🔜 Planned | NCCL all-reduce for distributed Fisher accumulation |

---

## License

Apache License 2.0 — Tensorbit Labs
