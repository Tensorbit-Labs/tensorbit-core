# CORING: N:M Structured Sparsity — Complete Mathematical Exposition

## 1. Overview

CORING enforces **N:M fine-grained structured sparsity** on neural network weight
tensors. For a given N:M pattern (e.g., 2:4), the weight tensor is partitioned into
contiguous groups of M elements; within each group, exactly N elements are retained
(non-zero) and M−N are pruned (zeroed).

This structured pattern maps directly to the **NVIDIA Ampere Sparse Tensor Core**
instruction set (`mma.sp`), delivering up to 2× matrix-multiply throughput on A100
and H100 GPUs.

**Reference implementation:** `include/tensorbit/core/coring.hpp`

---

## 2. Problem Statement

Given weight vector `w ∈ R^N` with per-element importance scores `s ∈ R^N`
(typically from an upstream EHAP pruner) and sparsity pattern N:M:

**Objective:** Partition into `G = N/M` groups of size M. Within each group `g`,
select exactly N weights to retain such that the sum of retained importance
is maximised:

$$\max_{K_g \subset \{0,\ldots,M-1\}, |K_g| = N} \sum_{i \in K_g} s_{b_g + i} \quad (1)$$

where `b_g = g · M` is the group base index.

---

## 3. Mask Selection Strategies

### 3.1 Top-N (kTopN) — O(G·M log M)

For each group, sort or partial-sort the M importance values and select the top N.
Uses `std::nth_element` (Hoare 1961, quickselect) for O(M) partial sorting.

**Mathematical property:** Top-N is provably optimal when importance scores are
drawn i.i.d. from a continuous distribution (Hubara et al. 2022, Theorem 1).

### 3.2 Optimal Enumeration (kOptimal) — O(G·C(M,N))

Enumerate all `C(M,N) = M! / (N! · (M−N)!)` possible mask patterns per group
using **Gosper's hack** (Knuth 2011, Sec. 7.2.1.3) for bitwise iteration.
Selects the pattern with maximum total importance.

| N:M | C(M,N) | Feasibility |
|-----|--------|-------------|
| 2:4 | 6 | Always |
| 4:8 | 70 | Always |
| 8:16 | 12,870 | Always |
| >16 | — | Falls back to kTopN |

### 3.3 Iterative Swap-Refine (kIterative) — O(G·M²·R)

1. **Initialise** with top-N mask.
2. **For each round** (up to `iterative_rounds`, default 3):
   - For each group, attempt to swap one pruned element with one kept element.
   - Accept the swap if `s_pruned > s_kept` (improves total importance).
   - Stop early if no swaps improve any group (convergence).

This is a **local search** heuristic (2-opt style). It can escape locally-optimal
top-N solutions and approaches the optimal mask with high probability.

---

## 4. Curvature-Aware Importance

CORING operates on **per-weight importance scores**. These can come from:

| Source | Formula | When to use |
|--------|---------|-------------|
| EHAP OBD | `s_i = w_i² · (F_ii + λ)` | Fisher diagonal available |
| EHAP OBS | `s_i = w_i² / (F_ii + λ)` | Inverse-Hessian diagonal available |
| Magnitude | `s_i = |w_i|` | No Fisher/history, quick pass |

When coupled with EHAP's Fisher accumulation and `store_gradient()`, the
importance scores passed to CORING are **curvature-aware**: they encode
both the weight magnitude AND the local loss-landscape curvature.

This means CORING's mask selection naturally preserves "load-bearing" weights
(high Fisher) even when their magnitude is moderate — the defining advantage
of second-order pruning.

---

## 5. Weight Redistribution

After N:M pruning, the remaining weights can be adjusted to compensate for
removed parameters.

### 5.1 Absolute-Magnitude Sum (avoiding sign cancellation)

Earlier implementations used raw weight sums for redistribution:
`Δ_g = Σ_{pruned} w_i`. This suffers from **signed-weight cancellation**:
a group with `w = [+5, −4]` (both pruned) would have `Δ_g = +1`, severely
underestimating the actual removed magnitude (9).

**Fix:** Use absolute magnitude:
`Δ_g = Σ_{pruned} |w_i|`

### 5.2 Proportional Redistribution (kProportional)

Distribute pruned magnitude to kept weights proportionally to their importance
scores (OBS-inspired):

$$w_i^{(\text{kept})} \leftarrow w_i^{(\text{kept})} + \text{sign}(w_i) \cdot \Delta_g \cdot \frac{s_i}{\sum_{j \in \text{kept}} s_j} \quad (2)$$

The `sign(w_i)` factor ensures the redistribution preserves the direction of
each weight, avoiding sign flips.

### 5.3 Uniform Redistribution (kUniform)

$$w_i^{(\text{kept})} \leftarrow w_i^{(\text{kept})} + \text{sign}(w_i) \cdot \frac{\Delta_g}{N} \quad (3)$$

Mean-preserving: the group's total absolute magnitude is conserved.

---

## 6. Permutation Optimization

When `permute_weights = true`, weights within each group are sorted by
absolute magnitude in descending order **before** mask selection:

$$w_{gM}, w_{gM+1}, \ldots, w_{gM+M-1} \leftarrow \text{sort\_descending}(|w_{gM}|, \ldots, |w_{gM+M-1}|)$$

This concentrates large weights in early group positions, making them more
likely to survive the N:M constraint. The caller is responsible for tracking
the index permutation to reverse it after pruning, maintaining the original
weight ordering.

**Reference:** Pool & Yu (2021) showed that channel-wise permutation of weight
columns before 2:4 sparsity application improves accuracy by 1–3 percentage
points at the same sparsity ratio.

---

## 7. Hardware-Aware 2:4 Layout

When `hardware_aware_layout = true` and the pattern is 2:4, the mask buffer
is structured to match NVIDIA's Ampere Sparse Tensor Core requirements:

- Each group of 4 weights maps to one mask byte (4 bits used)
- Groups are contiguous along the **inner GEMM K-dimension**
- The `mma.sp` instruction expects masks in this exact organisation

In row-major weight layout, groups are naturally contiguous along the K
dimension, so no reordering is needed. The `apply_ampere_2_4_layout()`
hook exists for future transpose/reshape support.

---

## 8. Mask Format

All CORING strategies use a **packed bitmask** format:

```
Byte g:   bit 0 → weight[g·M + 0] is kept (1) or pruned (0)
          bit 1 → weight[g·M + 1] is kept (1) or pruned (0)
          ...
          bit (M−1) → weight[g·M + M−1] is kept (1) or pruned (0)
```

One byte per group. Groups are stored consecutively: `mask_data[g]` for group `g`.

**Analytical pruned count** (deterministic, no runtime counting needed):

$$N_{\text{pruned}} = G \cdot (M - N) = \frac{N_{\text{elements}}}{M} \cdot (M - N) \quad (4)$$

Valid when `N_elements` is divisible by M (enforced by `validate_config`).

---

## 9. GPU Acceleration

| Kernel | Pattern | Thread Model | Shared Memory | Ops/Element |
|--------|---------|-------------|---------------|-------------|
| `nm_mask_2_4_kernel` | 2:4 | 1 thread/group | 0 B | O(1) |
| `nm_mask_generic_kernel` | Any N:M, M ≤ 32 | M threads/group | 256 B | O(M²) |
| `apply_mask_kernel` | Any M | 1 thread/element | 0 B | O(1) |

**GPU strategy support:** `kTopN` is available on GPU via these kernels.
`kOptimal` and `kIterative` run on CPU (they require combinatorial/heuristic
search not yet implemented in CUDA). When GPU is enabled, `kTopN` automatically
dispatches; other strategies fall back to CPU with automatic host/device transfers.

---

## 10. Implementation Reference

### 10.1 Source

`include/tensorbit/core/coring.hpp`

### 10.2 Key Types

| Type | Purpose |
|------|---------|
| `CORINGConfig` | N/M, CUDA toggle, mask strategy, redistribution mode, iterative rounds, permutation toggle, hardware layout toggle |
| `CORINGPruner<F>` | Template class for `float`/`double` |
| `MaskStrategy` | `kTopN`, `kOptimal`, `kIterative` |
| `RedistMode` | `kNone`, `kProportional`, `kUniform` |

### 10.3 Methods

| Method | Algorithm | Complexity |
|--------|----------|------------|
| `generate_nm_mask(s, mask)` | Mask selection per strategy + optional permutation + HW layout | O(G·C(M,N)) worst |
| `apply_mask(w, mask)` | Element-wise bit-test + zero | O(N) |
| `redistribute(w, mask, s)` | Group-local absolute-magnitude redistribution | O(N) |
| `prune(s, w)` | Full pipeline: mask → apply → redistribute | O(N) |

---

## 11. Configurable Behaviour Summary

| Setting | Effect |
|---------|--------|
| `mask_strategy = kTopN` | Fast partial-sort, GPU-accelerated |
| `mask_strategy = kOptimal` | Exact C(M,N) enumeration, CPU only |
| `mask_strategy = kIterative` | Locally-optimal swap-refine, CPU only |
| `iterative_rounds = R` | Max refinement rounds for kIterative |
| `redist_mode = kProportional` | Importance-weighted redistribution |
| `redist_mode = kUniform` | Equal-share redistribution |
| `permute_weights = true` | Group-local magnitude sort before masking |
| `hardware_aware_layout = true` | 2:4 mask alignment for Ampere GEMM |

---

## 12. Known Limitations

- **GPU parity**: Only `kTopN` runs on GPU. `kOptimal` and `kIterative` are
  CPU-only. A future release could implement warp-level combinatorial enumeration
  in CUDA for kOptimal.
- **Permutation reversal**: `permute_weights` reorders weights but does not
  track the index mapping for reversal. Caller must handle this.
- **M > 32**: The generic GPU kernel uses shared-memory ranking with `__shared__
  float[32]`. For M > 32, the CPU path processes groups serially. This is
  acceptable since practical N:M patterns (2:4, 1:4, 2:8, 4:8) all have M ≤ 8.
- **Cross-group coupling**: Groups are processed independently. Blockwise
  coupling (sharing sparsity budget across groups) is deferred to a future Phase.

---

## 13. References

1. **Mishra, A., Latorre, J. A., Pool, J., Stosic, D., Stosic, D., Venkatesh, G., Yu, C., & Micikevicius, P. (2021).** "Accelerating Sparse Deep Neural Networks." *arXiv:2104.08378*.

2. **Hubara, I., Chmiel, B., Island, M., Banner, R., Naor, J., & Soudry, D. (2022).** "Accelerated Sparse Neural Training." *NeurIPS 35*.

3. **Pool, J., & Yu, C. (2021).** "Channel Permutations for N:M Sparsity." *NeurIPS 34*.

4. **Frantar, E., & Alistarh, D. (2023).** "SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot." *ICML 2023*.

5. **NVIDIA Corporation (2021).** "Accelerating Inference with Sparsity Using the NVIDIA Ampere Architecture." *NVIDIA Technical Blog*.

6. **Hoare, C. A. R. (1961).** "Algorithm 65: Find." *Communications of the ACM, 4(7)*, pp. 321–322.

7. **Knuth, D. E. (2011).** *The Art of Computer Programming, Volume 4A*. Addison-Wesley.
