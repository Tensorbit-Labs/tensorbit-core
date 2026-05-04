# TODO.md — Remaining Enhancements (Not Blocking Cloud Test)

This document tracks features and improvements deferred to future releases.
**None of these items prevent the Lambda Mistral 7B cloud test.** The core
pipeline (EHAP → CORING → .tb → .tbm → run inference) is complete.

---

## 1. CORING: Ampere 2:4 layout hook for transposed weight matrices

**File:** `include/tensorbit/core/coring.hpp:359-365` — `apply_ampere_2_4_layout()`

### Current state
The function is a valid no-op for standard row-major weight layout. When
`hardware_aware_layout=true` is set in `CORINGConfig`, the function is called but
does nothing — because tensorbit-core always outputs weights in the exact
group-contiguous order that Ampere Sparse Tensor Cores expect along the GEMM
K-dimension.

### When does this matter?
If a weight matrix is **transposed** (e.g., PyTorch stores weights in `[out, in]`
while some frameworks use `[in, out]`), the 2:4 mask bytes no longer correspond to
contiguous groups along the inner dimension. The Ampere `mma.sp` instruction would
read the wrong mask bits for each group.

### Suggested fix
- Detect when weight shapes indicate a transposed layout.
- For transposed weight matrices, restructure the mask byte ordering to match the
  GEMM K-dimension alignment. Each mask byte corresponds to a group of 4 along the
  **inner** (M dimension) of the matrix multiply, which maps to the **second**
  dimension of a `[out_features, in_features]` weight matrix in row-major storage.
- Add a unit test that validates a known transposed layout produces identically
  pruned weights to the non-transposed version.
- Extend the CORING test suite (`test_coring.cpp`) with a `CORINGPruner_HardwareLayout`
  test case that creates a transposed importance array and verifies mask bytes.
- Check compatibility with cuSPARSELt's expected 2:4 mask layout (4 bits per
  group packed into bytes, groups contiguous along the K dimension of the GEMM).

---

## 2. CORING: Global column permutation optimization

**File:** `include/tensorbit/core/coring.hpp:apply_permutation()`

### Current state
Per-group magnitude sort is implemented (sorts importance within each M-element
group by absolute value). The mask is generated from the sorted copy via
`generate_topn()`. This is a lightweight heuristic.

### What's missing
Pool & Yu (2021) showed that **global column permutation** — shuffling columns
across groups to maximize the sum of kept-weight magnitudes — improves 2:4
sparsity accuracy by 1–3 percentage points. This requires:

- A **permutation matrix** `P` of size `[in_features, in_features]` that reorders
  columns of the weight matrix.
- An algorithm to compute P that maximizes `sum of kept magnitudes` under the
  N:M constraint. Options:
  - Hungarian algorithm (optimal for small blocks, O(n³)).
  - Greedy iterative permutation with local search.
  - Magnitude-based reordering (sort columns by norm, group similar magnitudes).
- A **reversal pass** in `prune()` that applies `P` to the weight matrix before
  serialization, and records `P` in the `.tb` metadata so tensorbit-run can apply
  `P^T` during inference (if needed).
- Extension of `TbmEntry` in `main.cpp` to track permutation state.

### Suggested fix
- Add a `CORINGConfig::enable_global_permutation` boolean.
- Implement the greedy iterative algorithm: sort columns by L2 norm, then apply
  a series of swap-improvement rounds within each N:M block.
- Store the permutation indices in a `.tb` metadata field (use the reserved
  bytes in `TBHeader` or append a dedicated metadata section after the mask blob).
- Update `tensorbit-run`'s loader to parse the permutation and apply it to rows
  of the weight matrix at inference time.

---

## 3. Testing: Algorithmic test coverage

**Files:** `tests/test_ehap.cpp`, `tests/test_coring.cpp`

### Current state
Both test suites validate **only error-path behavior**: config field accessors,
Fisher buffer initialization, empty tensor rejection, shape mismatch detection,
and prune pipeline continuation/failure. 30/30 tests pass.

### What's missing
No test validates the **numerical correctness** of any algorithm:

- `EHAPPruner::select_pruning_mask()` — does it select the correct top-k fraction?
- `EHAPPruner::compute_importance()` — does OBD produce `w²*(F+λ)`? OBS produce `w²/(F+λ)`? Normalized produce the bounded variant?
- `EHAPPruner::compensate_weights()` — does bias compensation add the correct delta? Does redist produce Fisher-weighted outputs?
- `CORINGPruner::generate_nm_mask()` — do TopN, Optimal (Gosper's hack), and Iterative (swap-refine) all agree on the correct mask for a known importance pattern?
- `CORINGPruner::redistribute()` — does proportional and uniform redistribute produce correct magnitude-based deltas?
- BlockOBS `prune_block_obs()` — does Woodbury inversion produce the correct H⁻¹ for a known diagonal matrix? Does the greedy OBS loop select the correct weights and apply Sherman-Morrison updates correctly?

### Suggested fix
- Add a test helper function that creates a small known weight tensor (e.g., `[1, 2, 3, 4, 5, 6, 7, 8]`) and verifies:
  - `compute_importance` with known Fisher values produces expected output.
  - `select_pruning_mask(0.5)` selects exactly the top-half elements.
  - `generate_nm_mask` produces byte `0b00000101` for importance `[5, 1, 4, 2]` (top-N=2 picks indices 2 and 0).
  - `compensate_weights` in bias mode: `weights[2,3,4], mask[1,0,1]` → `weights[6,0,4]` (mask[0]=1 gets delta of 3 from mask[1]).
- Reference the EHAP.md equation numbers to validate constants.
- Same for CORING: known importance, known mask output, known redistribution.

---

## 4. Testing: Migration to GoogleTest

**Files:** `tests/test_ehap.cpp`, `tests/test_coring.cpp`

### Current state
Tests use custom inline macros (`TEST(name)`, `EXPECT_TRUE(expr)`, `EXPECT_EQ(a,b)`)
defined locally in each test file. Test functions self-register via static
constructor objects.

### Suggested fix
- Add `FetchContent` for GoogleTest in `CMakeLists.txt` behind a `TENSORBIT_USE_GTEST` option.
- Replace the custom `TEST()`/`EXPECT_*` macros with standard `gtest/gtest.h` includes.
- Convert existing tests to `TEST(Suite, Name)` format.
- GoogleTest provides better test discovery (`--gtest_list_tests`), filtering (`--gtest_filter`),
  and output formatting. No change to test logic needed.
- Keep the custom harness as a fallback when GoogleTest is unavailable (e.g., minimal
  Docker/CI images without cmake FetchContent support).

---

## 5. I/O: Native C++ `merge_tbm` in tb-prune

**File:** `scripts/merge_tbm.py`

### Current state
The merge script is Python-only and runs as a separate step after pruning each
shard. It works correctly (verified by `test_merge.sh` — reads .tb headers,
concatenates blobs, builds JSON index with correct offsets).

### Suggested fix
- Add a `--merge-output` flag to `main.cpp`.
- When two `--output` directories are detected (both containing .tb files),
  automatically run the merge logic in C++ after the pruning loop completes.
- Use the existing `TBReader` class to read each .tb header, and `TBWriter`
  or raw `std::ofstream` for concatenation.
- Build the JSON index using the same `json_escape()` function and string
  concatenation already in `main.cpp`.
- Remove the separate `merge_tbm.py` execution step from the cloud guide.
- Keep `merge_tbm.py` as a standalone utility for edge cases.

---

## 6. Architecture auto-detection in .tbm JSON index

**File:** `src/main.cpp:477`

### Current state
The .tbm JSON index always writes `"architecture":"llama"` regardless of the
actual model being pruned. This is metadata — tensorbit-run uses it for
informational logging only (it doesn't change the inference logic).

### Suggested fix
- Parse the safetensors JSON metadata in `SafeTensorsFile` for an `"architectures"`
hint (some HuggingFace configs include this).
- Or: add a `--architecture` flag to `tb-prune` that the user can set (default `llama`).
- Write the detected or specified architecture into the `.tbm` JSON index.
- tensorbit-run's `TransformerRunner` already auto-detects naming conventions
  regardless of the architecture string, so this is a cosmetic/documentation fix.

---

## 7. GPU: Real-hardware CUDA kernel validation

**Files:** `src/kernels.cu`, `include/tensorbit/core/kernels.hpp`

### Current state
All 7 GPU kernels are compiled for SM80/SM90 and link correctly when
`TENSORBIT_ENABLE_CUDA=ON`. The CPU-only path passes all tests via
`kernels_stubs.cpp`.

### What's missing
No physical A100/H100 testing has been performed. The Lambda cloud test will
be the first exercise of the GPU kernels on real hardware.

### Suggested fix
- After the cloud test, run `tb-prune` with `nvprof` or `nsys profile` to collect
  kernel execution times and verify occupancy.
- Check that `cudaGetLastError()` returns `cudaSuccess` after every kernel launch.
- Test with CUDA_MEMCHECK (`cuda-memcheck ./tb-prune ...`) to detect out-of-bounds
  device memory access.
- Compare GPU path results against CPU path results for the same tensor to verify
  numerical equivalence.
- The 2:4 mask kernel is register-only, so it should achieve near-100% theoretical
  occupancy on A100 (108 SMs × 2048 threads/SM ÷ 256 threads/block = 864 blocks/SM).
  Verify with `nvprof --metrics achieved_occupancy`.

---

## 8. GPU: NVidia Tools Extension (NVTX) profiler ranges

**Files:** `src/kernels.cu`, `include/tensorbit/core/kernels.hpp`

### Current state
No NVTX range markers, CUDA events, or profiler annotations exist in any kernel
launch wrapper.

### Suggested fix
- Add `#include <nvtx3/nvToolsExt.h>` to `kernels.cu`.
- Wrap each `launch_*()` with:
  ```cuda
  nvtxRangePushA("fisher_accumulate");
  fisher_accumulate_kernel<<<...>>>(...);
  cudaDeviceSynchronize();
  nvtxRangePop();
  ```
- This produces annotated timelines in `nsys-ui` and `nvprof`, making it easy to
  identify which kernel dominates runtime.
- Add a CMake option `TENSORBIT_ENABLE_NVTX` (default OFF) to conditionally compile
  the markers.

---

## Summary

| # | Item | Category | Blocks Cloud Test? |
|---|------|----------|:-:|
| 1 | Ampere layout hook for transposed weights | CORING | No |
| 2 | Global column permutation | CORING | No |
| 3 | Algorithmic test coverage | Testing | No |
| 4 | Migration to GoogleTest | Testing | No |
| 5 | Native C++ merge_tbm in tb-prune | I/O | No |
| 6 | Architecture auto-detection in JSON | I/O | No |
| 7 | Real-hardware CUDA kernel validation | GPU | No |
| 8 | NVTX profiler range markers | GPU | No |

**Last updated:** May 2026 — v0.2.0 pre-cloud-test state.

**Next milestone:** Lambda test completed → tensorbit-distill construction.
