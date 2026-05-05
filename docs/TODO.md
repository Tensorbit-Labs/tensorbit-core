# TODO.md — Remaining Enhancements

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

## 6. GPU: Real-hardware CUDA kernel validation

**Files:** `src/kernels.cu`, `include/tensorbit/core/kernels.hpp`

### Current state
All 7 GPU kernels are compiled for SM80/SM90 and link correctly when
`TENSORBIT_ENABLE_CUDA=ON`. The CPU-only path passes all tests via
`kernels_stubs.cpp`.

### What's missing
No physical GPU testing has been performed. The kernels have only been
syntax-checked and linked — never executed on real hardware.

### Suggested fix
- Run `tb-prune` with `nvprof` or `nsys profile` to collect kernel execution
  times and verify occupancy.
- Check that `cudaGetLastError()` returns `cudaSuccess` after every kernel launch.
- Test with `cuda-memcheck` to detect out-of-bounds device memory access.
- Compare GPU path results against CPU path results for the same tensor to verify
  numerical equivalence.
- Verify achieved occupancy: the 2:4 mask kernel is register-only and should
  achieve near-100% theoretical occupancy.

---

## 7. GPU: NVTX profiler range markers

**Files:** `src/kernels.cu`, `include/tensorbit/core/kernels.hpp`

### Current state
No NVTX range markers, CUDA events, or profiler annotations exist in any kernel
launch wrapper.

### Suggested fix
- Add `#include <nvtx3/nvToolsExt.h>` to `kernels.cu`.
- Wrap each `launch_*()` with `nvtxRangePushA("name")` / `nvtxRangePop()`.
- Add a CMake option `TENSORBIT_ENABLE_NVTX` (default OFF) to conditionally
  compile the markers — produces annotated timelines in `nsys-ui` and `nvprof`.

---

## 8. CMAKE: User-overridable CUDA architectures

**File:** `CMakeLists.txt:46-49`

### Current state (fixed May 2026)
`CMAKE_CUDA_ARCHITECTURES` is now guarded with `if(NOT DEFINED)`, allowing
users to override via `-DCMAKE_CUDA_ARCHITECTURES="86"` on the command line.

### Remaining issue
The default list only targets `80;90` (A100/H100). Consumer GPUs (RTX 30-series
= SM86, RTX 40-series = SM89) require the user to know and specify their
architecture. A more user-friendly approach would be to detect the installed
GPU and auto-select, or at minimum include `75;80;86;89;90` as a broader default.

---

## 9. CLI: `obs_block_size` not configurable

**Files:** `include/tensorbit/core/ehap.hpp:85`, `src/main.cpp:150`

### Current state
The OBS block size is hardcoded to 128 both in the `EHAPConfig` default and
in the CLI parser. Users cannot tune it.

### Suggested fix
Add a `--obs-block-size N` CLI flag to `main.cpp`. For large embeddings
(131M params), a larger block size reduces total blocks and prunes faster
at the cost of slightly degraded accuracy. Default 128 is a good balance.

---

## 10. CLI: `gradient_history_size` not configurable

**File:** `include/tensorbit/core/ehap.hpp:92`

### Current state
The gradient history ring buffer is fixed at 4 gradients. This controls the
rank K of the Woodbury low-rank approximation in BlockOBS. Larger K =
better Hessian approximation but higher CPU cost (O(K³) per block).

### Suggested fix
Add a `--grad-samples N` CLI flag. Default 4 is adequate for most uses;
increase to 8-16 for paper-quality results.

---

## 11. CORING: `use_cuda` always true in CLI

**File:** `src/main.cpp:168`

### Current state
`CORINGConfig::use_cuda` is always set to `true` by the CLI builder with no
way to disable it. If the CPU-only build is used, this flag remains `true`
but is gracefully ignored by the CORING implementation (which checks
`if constexpr (std::is_same_v<F, float>)` before GPU dispatch).

### Suggested fix
Either auto-detect `TENSORBIT_ENABLE_CUDA` at runtime and default accordingly,
or add a `--no-gpu` flag. Low priority since the current behavior is harmless.

---

## 12. I/O: Model config auto-detection from safetensors

**File:** `src/main.cpp:556-573`

### Current state (fixed May 2026)
Config values (`hidden_size`, `num_heads`, etc.) are now configurable via CLI
flags with sensible defaults (Mistral 7B values). Architecture name is
configurable via `--architecture`.

### Remaining issue
The user must manually specify model dimensions via CLI flags. A more robust
approach would parse the HuggingFace `config.json` (if available alongside
the `.safetensors` file) to auto-populate these values. The CLI flags would
serve as overrides. This prevents silent wrong metadata for non-Mistral models.

---

## 13. I/O: JSON built with string concatenation, no structural validation

**File:** `src/main.cpp:552-600`

### Current state
The .tbm JSON index is built with raw `std::string` concatenation and
`+=` operators. A single typo (missing comma, unescaped quote) produces
silently invalid JSON that downstream tensorbit-run cannot parse.

### Suggested fix
Replace string concatenation with a lightweight JSON builder that validates
structure. Either use a minimal JSON utility class or adopt structured
serialization (e.g., write key-value pairs through a wrapper that enforces
proper delimiters and escaping). This also simplifies adding new fields.

---

## 14. Main: Duplicate pipeline logic (mock vs real mode)

**File:** `src/main.cpp:312-503`

### Current state
The EHAP+CORING pipeline code is duplicated between mock mode (lines 316-355)
and real mode (lines 444-498). Any fix to one must be applied to the other.
Currently ~80 lines of identical logic with different data sources.

### Suggested fix
Extract the pipeline into a function `prune_and_save(EHAPPruner&, CORINGPruner&, TensorDense<float>&, const CliConfig&)` that both modes call.
Reduces code duplication and eliminates divergence risk.

---

## 15. Main: Mock gradients used in real mode

**File:** `src/main.cpp:225-231, 440`

### Current state
In real mode, the pipeline uses **synthetic mock gradients** instead of
actual gradient data from the model. The Fisher information is computed
from `weights[i] * 0.01` (magnitude-based proxy), which degrades pruning
quality to essentially magnitude-based selection rather than true
loss-landscape-aware pruning.

### Suggested fix
Extract real gradient statistics from HuggingFace models using PyTorch
(either via embedded Python or a separate gradient extraction step).
Alternatively, support loading a pre-computed `.fisher` file containing
the Fisher diagonal (computed offline via `torch.autograd`). The mock
gradients remain as a fallback for testing.

---

## 16. CMAKE: Eigen3 search paths are Debian/Ubuntu-only

**File:** `CMakeLists.txt:68-72`

### Current state
Eigen3 `find_path` PATHS only include `/usr/include/eigen3`,
`/usr/local/include/eigen3`, `/usr/include`, `/usr/local/include`,
and `D:/eigen3`. Missing `/opt/homebrew/include/eigen3` (macOS
Apple Silicon), `/usr/local/opt/eigen/include/eigen3` (macOS Intel
Homebrew), and vcpkg/Conan install prefixes.

### Suggested fix
- Add `/opt/homebrew/include/eigen3` and `/usr/local/opt/eigen/include/eigen3`
  to PATHS list.
- Document `-DEIGEN3_ROOT=<path>` as the recommended approach for
  non-standard installations.
- Remove the Windows-specific `D:/eigen3` from PATHS (it's already captured
  by the `HINTS ${EIGEN3_ROOT}` directive).

---

## 17. setup_cloud.sh: Platform assumptions

**File:** `scripts/setup_cloud.sh:85, 118, 122, 154`

### Current state
- Toolchain PPA hardcodes `jammy` (Ubuntu 22.04). Ubuntu 24.04 noble users
  need a different PPA or manual GCC 13 install.
- CUDA keyring URL hardcodes `ubuntu2204`. Should use `${UBUNTU_CODENAME}`
  or auto-detect.
- CUDA toolkit version hardcoded to `cuda-toolkit-12-6`.
- Python 3.11 assumed primary; Ubuntu 24.04 ships 3.12.

### Suggested fix
- Use `${UBUNTU_CODENAME}` for all PPA and package URLs (already done
  for Kitware repo, should extend to toolchain and CUDA).
- Detect available Python version dynamically: `apt list python3.1*` and
  pick the highest.
- Make CUDA version configurable via script argument or env var.

---

## 18. verify_ubuntu.sh: WSL disk check priority

**File:** `scripts/verify_ubuntu.sh:193`

### Current state
Disk check tries `/mnt/d` first (WSL mount) before falling back to `/`.
On real multi-disk Linux systems, `/mnt/d` might legitimately exist and
report the wrong disk.

### Suggested fix
Check `/` first, then `/mnt/d` only if WSL is detected (i.e., inside
the `grep -qi microsoft /proc/version` block).

---

## Summary

| # | Item | Category | Blocks Cloud Test? |
|---|------|----------|:-:|
| 1 | Ampere layout hook for transposed weights | CORING | No |
| 2 | Global column permutation | CORING | No |
| 3 | Algorithmic test coverage | Testing | No |
| 4 | Migration to GoogleTest | Testing | No |
| 5 | Native C++ merge_tbm in tb-prune | I/O | No |
| 6 | Real-hardware CUDA kernel validation | GPU | No |
| 7 | NVTX profiler range markers | GPU | No |
| 8 | User-overridable CUDA architectures (partial) | Build | No |
| 9 | obs_block_size CLI flag | CLI | No |
| 10 | gradient_history_size CLI flag | CLI | No |
| 11 | use_cuda always true in CLI | CORING | No |
| 12 | Model config auto-detection | I/O | No |
| 13 | JSON string concatenation validation | I/O | No |
| 14 | Duplicate pipeline logic (mock vs real) | Main | No |
| 15 | Mock gradients in real mode | EHAP | No |
| 16 | Eigen3 cross-platform search paths | Build | No |
| 17 | setup_cloud.sh platform assumptions | Scripts | No |
| 18 | verify_ubuntu.sh disk check priority | Scripts | No |

**Last updated:** May 2026 — v0.2.0 post-audit state.

**Next milestone:** Lambda test confirmed → tensorbit-distill construction.
