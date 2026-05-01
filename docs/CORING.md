# CORING: N:M Structured Sparsity — Complete Mathematical Exposition

## 1. Overview

CORING is a structured sparsity engine that enforces **N:M fine-grained sparsity**
patterns on neural network weight tensors. For a given N:M pattern (e.g., 2:4), the
weight tensor is partitioned into contiguous groups of M elements; within each group,
exactly N elements are retained (non-zero) and M−N are pruned (zeroed). This structured
pattern maps directly to the **NVIDIA Ampere Sparse Tensor Core** instruction set
(`mma.sp`), delivering up to 2× matrix-multiply throughput on A100 and H100 GPUs.

Unlike random or unstructured pruning, N:M sparsity guarantees **regular memory access
patterns** and **predictable warp utilization**, eliminating the warp-divergence
penalties that plague unstructured sparsity at inference time.

The name "CORING" derives from the idea of extracting the **core** (highest-importance)
elements from each structural group — analogous to geological coring where the most
valuable material is extracted in a structured cylindrical pattern.

---

## 2. Mathematical Formulation

### 2.1 Problem Statement

Given a weight vector `w ∈ R^N` with associated importance scores `s ∈ R^N` (typically
computed by the upstream EHAP pruner), and a sparsity pattern `N:M` (where `N < M`):

**Objective**: Partition the weights into `G = N/M` contiguous groups of size M, and
within each group select exactly N weights to retain, such that the total importance
of the retained weights is maximized.

Formally, for group `g` with base index `b_g = g · M`:

$$\text{maximize}_{K_g \subset \{0,\ldots,M-1\}, |K_g| = N} \quad \sum_{i \in K_g} s_{b_g + i} \quad (1)$$

subject to the constraint that each group has exactly N kept elements.

### 2.2 Optimal Selection (Exhaustive)

For group size M, there are `C(M, N) = M! / (N! · (M-N)!)` possible mask patterns.
The exhaustive-search strategy (`MaskStrategy::kOptimal`) evaluates every pattern
and selects the one with maximum total importance:

$$K_g^* = \arg\max_{K \subset \{0,\ldots,M-1\}, |K| = N} \sum_{i \in K} s_{b_g + i} \quad (2)$$

This is tractable for small M:
- 2:4 → C(4,2) = 6 patterns per group
- 4:8 → C(8,4) = 70 patterns per group
- 8:16 → C(16,8) = 12,870 patterns per group

For M > 16, exhaustive enumeration becomes impractical and the implementation falls
back to top-N selection.

The exhaustive enumeration uses **Gosper's hack** (a bit-twiddling algorithm that
iterates over all n-bit numbers with exactly k bits set), providing O(C(M,N)) time
per group with zero heap allocation.

**References**: The combinatorial formulation of N:M mask selection is discussed in
Mishra et al. (2021) and formalized in Hubara et al. (2022).

### 2.3 Top-N Selection (Heuristic)

The top-N strategy (`MaskStrategy::kTopN`) approximates the optimal solution by
sorting (or partially sorting) the M importance values within each group and selecting
the top N:

$$K_g^{(\text{topN})} = \{ \text{indices of the N largest } s_{b_g + i} \text{ for } i = 0,\ldots,M-1 \} \quad (3)$$

This is O(M log M) per group using `std::nth_element` (Hoare 1961, quickselect partial
sort). For most practical importance distributions, top-N produces solutions within
1-5% of the optimal total importance (Hubara et al., 2022, Table 2).

### 2.4 Iterative Refinement

The iterative strategy (`MaskStrategy::kIterative`) alternates between mask selection
and local improvement:

1. **Initialization**: Compute top-N mask for each group.
2. **Refinement**: For each group, attempt to swap one pruned element with one kept
   element. If the swap increases the total-group importance, accept it.
3. **Repeat** for `iterative_rounds` iterations (default 3) or until convergence.

This is a **local search** heuristic (akin to 2-opt for the traveling salesman problem)
that can escape locally-suboptimal top-N solutions. The search is O(M²) per group per
round — tractable because M ≤ 32.

### 2.5 Weight Permutation Optimization

When `permute_weights = true`, the weights within each group are sorted by magnitude
before mask selection. This is a lightweight heuristic that improves the N:M constraint
by grouping similar-magnitude weights together:

$$w_{b_g}, w_{b_g+1}, \ldots, w_{b_g+M-1} \leftarrow \text{sort\_descending}(|w_{b_g}|, \ldots, |w_{b_g+M-1}|)$$

Sorting by magnitude before N:M selection ensures that the largest weights are
concentrated in early positions within each group, making it more likely that they
survive the N:M constraint. This is a practical approximation to the **permutation
optimization** studied by Pool & Yu (2021) and Frantar & Alistarh (2023).

**References**: Pool & Yu (2021) showed that, for 2:4 sparsity, permuting weight
columns to maximize the sum of kept-weight magnitudes can improve accuracy by
1-3 percentage points at the same sparsity.

---

## 3. Mask Format

### 3.1 Packed Bitmask Convention

CORING uses a **packed bitmask** representation: each group of M weights consumes one
byte, where bit `i` (0-indexed, LSB) indicates whether weight `i` within the group
is kept (1) or pruned (0):

```
Byte g:   bit 0 → weight[g·M + 0]
          bit 1 → weight[g·M + 1]
          ...
          bit (M-1) → weight[g·M + M−1]
```

For `M = 2^k`, exactly one byte per group. Groups are stored consecutively:
`mask_data[g]` corresponds to the `g`-th group.

This format aligns with the NVIDIA cuSPARSELt 2:4 mask convention where each group of
4 is encoded as a 4-bit pattern. The CORING format generalizes this to arbitrary N:M.

### 3.2 Analytical Pruned Count

Since each group has exactly M − N pruned elements, the total pruned count is
deterministic:

$$N_{\text{pruned}} = G \cdot (M - N) = \frac{N_{\text{elements}}}{M} \cdot (M - N) \quad (4)$$

This formula is exact when `N_elements` is divisible by M (enforced by `validate_config`).

---

## 4. Weight Redistribution

### 4.1 Proportional Redistribution (kProportional)

After N:M pruning, the total magnitude of pruned weights within each group is
redistributed to the kept weights proportionally to their Fisher/importance values.
For group `g` with kept index set `K_g`:

$$w_{b_g + i} \leftarrow w_{b_g + i} + \Delta_g \cdot \frac{s_{b_g + i}}{\sum_{j \in K_g} s_{b_g + j}}, \quad \forall i \in K_g \quad (5)$$

where `Δ_g = Σ_{i ∉ K_g} w_{b_g + i}` is the total pruned magnitude.

This is a **group-local OBS approximation**: it preserves the total contribution of the
group while directing the compensation to the weights with the highest Fisher sensitivity.
Unlike full OBS (which requires the inverse Hessian), redistribution uses the per-weight
importance scores as a proxy for sensitivity.

### 4.2 Uniform Redistribution (kUniform)

A simpler alternative: the pruned magnitude is distributed equally among all kept weights:

$$w_{b_g + i} \leftarrow w_{b_g + i} + \frac{\Delta_g}{N}, \quad \forall i \in K_g \quad (6)$$

This is a **mean-preserving** correction: the group's total weight sum is unchanged
after pruning + redistribution. Uniform redistribution is appropriate when importance
scores are unavailable or unreliable.

---

## 5. GPU Acceleration

### 5.1 CUDA Kernels

For `F = float` with CUDA enabled, CORING dispatches to specialized CUDA kernels:

| Kernel | Group Size | Thread Model | Shared Memory | Time Complexity |
|--------|-----------|-------------|---------------|-----------------|
| `nm_mask_2_4_kernel` | M = 4 | 1 thread/group (register-only) | 0 B | O(1) |
| `nm_mask_generic_kernel` | M ≤ 32 | M threads/group (cooperative) | 256 B | O(M²) |
| `apply_mask_kernel` | Any M | 1 thread/element | 0 B | O(1) |

The 2:4 kernel is the "gold standard" for Ampere hardware — it processes one group
per thread using local registers only, achieving near-100% theoretical occupancy.
The generic kernel uses shared-memory cooperative ranking for arbitrary N:M patterns
up to M = 32.

### 5.2 Optimal Mask Selection on GPU

The exhaustive (`kOptimal`) and iterative (`kIterative`) mask strategies currently
execute on CPU. GPU-accelerated versions using warp-level combinatorial enumeration
are planned for Phase 4.

---

## 6. Implementation Reference

### 6.1 Source Location

`include/tensorbit/core/coring.hpp`

### 6.2 Key Classes

| Class | Purpose |
|-------|---------|
| `CORINGConfig` | Configuration with N, M, mask strategy, redistribution mode |
| `CORINGPruner<F>` | Template class for `float`/`double` precision |
| `MaskStrategy` | Enum: `kTopN`, `kOptimal`, `kIterative` |
| `RedistMode` | Enum: `kNone`, `kProportional`, `kUniform` |

### 6.3 Core Methods

| Method | Algorithm | Complexity |
|--------|----------|------------|
| `generate_nm_mask(s, mask)` | Mask selection per strategy | O(G · C(M,N)) or O(G · M) |
| `apply_mask(w, mask)` | Element-wise zeroing | O(N_elements) |
| `redistribute(w, mask, s)` | Group-local weight redistribution | O(N_elements) |
| `prune(s, w)` | Full pipeline: mask → apply → redistribute | O(N_elements) |

---

## 7. Usage Example

```cpp
CORINGConfig cfg;
cfg.N            = 2;
cfg.M            = 4;
cfg.mask_strategy = MaskStrategy::kOptimal;
cfg.redist_mode  = RedistMode::kProportional;
cfg.permute_weights = true;

CORINGPruner<float> pruner(cfg);

// importance comes from upstream EHAP pruner
auto result = pruner.prune(importance_scores, model_weights);
if (result.has_value())
    std::printf("Applied 2:4 sparsity, pruned %zu weights\n", result.value());
```

---

## 8. References

1. **Mishra, A., Latorre, J. A., Pool, J., Stosic, D., Stosic, D., Venkatesh, G., Yu, C., & Micikevicius, P. (2021).** "Accelerating Sparse Deep Neural Networks." *arXiv:2104.08378*. — Introduces the N:M transposable fine-grained sparsity concept and the hardware motivation for structured sparsity on Ampere GPUs.

2. **Hubara, I., Chmiel, B., Island, M., Banner, R., Naor, J., & Soudry, D. (2022).** "Accelerated Sparse Neural Training: A Provable and Efficient Method to Find N:M Transposable Masks." *Advances in Neural Information Processing Systems (NeurIPS) 35*. — Provides the combinatorial formulation of N:M mask selection (Equation 1) and proves that top-N is near-optimal for most distributions.

3. **Pool, J., & Yu, C. (2021).** "Channel Permutations for N:M Sparsity." *Advances in Neural Information Processing Systems (NeurIPS) 34*. — Studies permutation optimization for 2:4 sparsity and demonstrates accuracy improvements from weight reordering before N:M application.

4. **Frantar, E., & Alistarh, D. (2023).** "SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot." *International Conference on Machine Learning (ICML) 2023*. — Demonstrates that one-shot N:M pruning of LLMs is feasible with proper importance estimation, validating the CORING pipeline design.

5. **NVIDIA Corporation (2021).** "Accelerating Inference with Sparsity Using the NVIDIA Ampere Architecture and NVIDIA TensorRT." *NVIDIA Technical Blog*. — Specifications for the 2:4 Sparse Tensor Core mask format and the `mma.sp` instruction.

6. **LeCun, Y., Denker, J., & Solla, S. (1990).** "Optimal Brain Damage." *NeurIPS 2*. — The OBD diagonal-Hessian framework that underlies per-weight importance estimation (used as input to CORING).

7. **Hoare, C. A. R. (1961).** "Algorithm 65: Find." *Communications of the ACM, 4(7)*, pp. 321–322. — Quickselect algorithm used for O(M) top-N selection within groups.

8. **Knuth, D. E. (2011).** *The Art of Computer Programming, Volume 4A: Combinatorial Algorithms, Part 1*. Addison-Wesley. — Gosper's hack for bitwise enumeration of k-combinations (Section 7.2.1.3).
