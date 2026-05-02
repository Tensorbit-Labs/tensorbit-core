# EHAP: Efficient Hessian-Aware Pruning — Complete Mathematical Exposition

## 1. Overview

Efficient Hessian-Aware Pruning (EHAP) is a second-order model compression algorithm
that identifies and removes redundant neural-network parameters by estimating each
parameter's impact on the loss function. Unlike magnitude-based pruning, which
discards weights solely by size, EHAP couples the **magnitude** of a weight with the
**local curvature** of the loss landscape.

EHAP is grounded in the **Optimal Brain Damage (OBD)** framework of LeCun et al.
(1990) and the **Optimal Brain Surgeon (OBS)** of Hassibi & Stork (1993), combined
with a low-rank + diagonal Hessian approximation and Woodbury-accelerated inversion
for efficient exact OBS updates.

**Reference implementation:** `include/tensorbit/core/ehap.hpp`

---

## 2. Mathematical Foundation

### 2.1 Second-Order Taylor Expansion

Let `L(w; D)` denote the loss of a neural network with weight vector `w ∈ R^N`.
Perturbing a single weight `w_i` by pruning it (`δw_i = -w_i`) gives the loss change
via second-order Taylor expansion:

$$L(w + \delta w) \approx L(w) + \nabla L(w)^\top \delta w + \frac{1}{2} \delta w^\top H(w) \delta w \quad (1)$$

At a local minimum, ∇L ≈ 0, so:

$$\Delta L_i \approx \frac{1}{2} w_i^2 H_{ii} \quad (2)$$

This is the **fundamental OBD relationship** (LeCun et al. 1990, Eq. 4): importance
scales with squared magnitude times Hessian diagonal.

### 2.2 Fisher Diagonal (O(N) memory)

The full Hessian is O(N²). The **Empirical Fisher Information diagonal** is O(N):

$$F_{ii} = \mathbb{E}_{x \sim D}\left[ \left( \frac{\partial L}{\partial w_i} \right)^2 \right] \quad (3)$$

Under standard asymptotic conditions, `F_ii → H_ii` at the optimum (Kunstner et al. 2019).

### 2.3 Importance Scores

Three importance formulations are available:

**OBD** (kOBD): `s_i = w_i^2 · (F_ii + λ)` — LeCun et al. (1990)

**OBS-style** (kOBS): `s_i = w_i^2 / (F_ii + λ)` — Hassibi & Stork (1993) diagonal

**Normalized** (kNormalized): `s_i = w_i^2·(F_ii+λ) / (1 + w_i²)` — scale-robust variant

---

## 3. Hessian Approximation (Blockwise OBS)

The blockwise OBS path (`PruneStrategy::kBlockOBS`) uses a **low-rank + diagonal**
Hessian approximation to apply exact OBS compensation weights.

### 3.1 Low-Rank + Diagonal Hessian

For a block of size B, the Hessian is decomposed as:

$$H_B = \text{diag}(F_B + \lambda) + U \cdot U^\top \quad (4)$$

where U is B×K. Two sources for U:

| Source | K | Description |
|--------|---|-------------|
| Gradient snapshots | configurable (default 4) | `U_k = w_k · g^{(k)}` with EMA decay `w_k = exp(-age/τ)`, τ = 2.0 |
| Weight fallback | 1 | `U = sqrt(α) · w_B` — weight self-correlation regulariser |

The EMA decay ensures recent snapshots (which reflect current loss-landscape
geometry) dominate H, while older snapshots contribute diminishing weight.
This matches WoodFisher's approach (Singh & Alistarh 2020) of maintaining an
online low-rank Fisher approximation.

**Key insight:** The off-diagonal structure comes from gradient covariance
`g·g^T`, NOT from the weights themselves. When gradients are available, H
captures true data-dependent curvature. The weight fallback is a structural
regulariser only, active when no gradient history exists.

### 3.2 Woodbury Inversion

The diagonal + low-rank form permits exact inversion via Woodbury:

$$H^{-1} = D^{-1} - D^{-1} \cdot U \cdot (I_K + U^\top D^{-1} U)^{-1} \cdot U^\top \cdot D^{-1} \quad (5)$$

- `D^{-1}` is O(B) (element-wise on diagonal)
- The K×K inner matrix `M = I + U^T·D^{-1}·U` is inverted via LDL^T with
  regularisation `M += ε·I` (ε = 1e-8) for numerical stability
- Building the full H^{-1} costs **O(B²·K²)** — for B = 128, K = 4:
  ~262K ops vs O(B³) = ~2M ops for Cholesky (8× speedup)

Once H^{-1} is built, the OBS loop uses **Sherman-Morrison rank-1 deflation**
(Hassibi & Stork 1993, Eq. 12) after each pruned weight:

$$H^{-1} \leftarrow H^{-1} - \frac{H^{-1}_{:,i} \cdot H^{-1}_{i,:}}{H^{-1}_{i,i}} \quad (6)$$

This is O(B²) per step — inherent to any OBS implementation.

### 3.3 OBS Compensation (Exact Under the Approximation)

Given the Hessian approximation H ≈ diag(F+λ) + U·U^T, the OBS update is:

$$\delta w_j = -\frac{w_i}{[H^{-1}]_{ii}} \cdot [H^{-1}]_{ji} \quad \forall j \quad (7)$$

This minimizes `‖w − w′‖²_H` under the constraint that weight `i` is pruned.
The compensation is **exact under the chosen Hessian approximation** — it is not
a heuristic. When K > 0 and gradient snapshots are available, this provides
cross-weight compensation driven by genuine data-dependent curvature.

When K = 0 (no gradients) and α = 0 (no weight regulariser), Eq. (7) reduces
to pure OBD: `δw_j = 0` for `j ≠ i` — no cross-weight compensation.

### 3.4 Adaptive Sparsity Allocation

Per-block importance statistics drive a softmax-weighted sparsity distribution:

1. Compute block mean importance `μ_b = mean(w_i^2 · (F_ii + λ))` for block b
2. Normalise: `r_b = μ_b / max({μ_b})`
3. Weight: `w_b = exp(-4 · r_b)`
4. Allocate: `prune_b = prune_total · w_b / Σ w_b`
5. Normalise and distribute remainder evenly

Blocks with **lower mean importance** receive **more pruning** — this is
data-adaptive and preserves model accuracy compared to uniform allocation.

---

## 4. Fisher Accumulation (EMA)

The Fisher diagonal is maintained via Exponential Moving Average:

$$F_{ii}^{(t)} = \beta \cdot F_{ii}^{(t-1)} + (1-\beta) \cdot g_i^2 \quad (8)$$

- β ∈ (0, 1] is the EMA decay factor (default 0.99)
- First step uses α (scaling factor) directly
- GPU path: `fisher_accumulate_kernel` followed by CUDA sync

---

## 5. Weight Compensation (Heuristic Modes)

For non-block-OBS pruning strategies, two compensation modes are available:

**Bias compensation** (kBias): `b += Σ(pruned w_i)` — LeCun et al. (1990, Sec. 3)

**Redistribution** (kRedist): Distribute pruned magnitude to kept weights
proportionally to Fisher values within groups of 8 — OBS-inspired diagonal
approximation.

Note: In `kBlockOBS` mode, compensation is exact via Eq. (7). The heuristic
compensation modes are NOT used with block-OBS.

---

## 6. Computational Complexity

### 6.1 Per-Block Costs (B = block size, K = low-rank factor ≤ 4)

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| Build U matrix | O(B·K) | Extract block from snapshot ring buffer |
| Woodbury H^{-1} construction | O(B²·K²) | Via outer-product summation |
| OBS saliency scan | O(B) | Per pruning step |
| OBS weight update | O(B) | δw_j for all j ≠ i |
| Sherman-Morrison deflation | O(B²) | Per pruning step (inherent to OBS) |
| Total per block (B steps) | **O(B³)** | Dominated by Sherman-Morrison |

For B = 128: ~2M FP32 ops per block. For a 7B-parameter model: ~0.1 sec/block.

### 6.2 Global Costs

| Component | Complexity |
|-----------|-----------|
| Fisher EMA accumulation | O(N) per batch |
| Importance computation | O(N) |
| One-shot pruning | O(N log N) or O(N) (nth_element) |
| Blockwise OBS (kBlockOBS) | O(N·B²) |
| Iterative pruning | O(T·N log N) for T rounds |

---

## 7. Implementation Reference

### 7.1 Source Location

`include/tensorbit/core/ehap.hpp`

### 7.2 Key Types

| Type | Purpose |
|------|---------|
| `EHAPConfig` | damping, sparsity, EMA decay, importance/strategy/compensation modes, block-OBS & gradient-history settings |
| `EHAPPruner<F>` | Template class for `float`/`double` precision |
| `ImportanceMode` | `kOBD`, `kOBS`, `kNormalized` |
| `PruneStrategy` | `kOneShot`, `kIterative`, `kBlockOBS` |
| `CompensationMode` | `kNone`, `kBias`, `kRedist` |

### 7.3 Methods

| Method | Algorithm | Complexity |
|--------|----------|------------|
| `store_gradient(g)` | Store EMA-weighted gradient snapshot into ring buffer | O(N) |
| `accumulate_fisher(g, α)` | EMA Fisher diagonal: F ← βF + (1-β)g² | O(N), GPU-accelerated |
| `compute_importance(w, out)` | Score per configured ImportanceMode | O(N), GPU-accelerated |
| `select_pruning_mask(s, mask)` | O(N) threshold via nth_element (Hoare 1961) | O(N) |
| `apply_mask(w, mask)` | Element-wise zeroing | O(N) |
| `compensate_weights(w, m, s)` | kBias or kRedist | O(N) |
| `prune(w)` | Dispatcher → one-shot / iterative / block-OBS | Varies |
| `prune_one_shot(w)` | importance → mask → apply → compensate | O(N log N) |
| `prune_iterative(w)` | Cubic schedule (Zhu & Gupta 2017), T rounds | O(T·N log N) |
| `prune_block_obs(w)` | Woodbury H^{-1}, EMA-weighted grad-covar, adaptive sparsity, Sherman-Morrison | O(N·B²) |
| `gradient_history()` | Returns current snapshot ring buffer | O(1) |

### 7.4 GPU Acceleration

When `F = float` and CUDA is enabled:
- `accumulate_fisher` dispatches to `fisher_accumulate_kernel`
- `compute_importance` dispatches to `ehap_importance_kernel`
- Blockwise OBS (`kBlockOBS`) runs on CPU (Eigen3 linear algebra)
- Double-precision (`F = double`) uses CPU path exclusively

---

## 8. Known Scope & Limitations

### 8.1 What EHAP Does

- Second-order pruning using Fisher diagonal + low-rank gradient covariance
- Exact OBS weight compensation under the low-rank + diagonal Hessian model
- Adaptive per-block sparsity allocation based on data-dependent importance
- EMA decay on both Fisher diagonal and gradient history

### 8.2 What EHAP Does NOT Do (Future Work)

- **Structured pruning** (channel/head/filter pruning): CORING handles N:M
  fine-grained sparsity; coarse structural pruning is Future Work.
- **Quantization coupling**: The pruning pipeline is precision-agnostic.
  INT4/INT8-aware pruning requires modifying the importance score to account
  for quantisation error (Nagel et al. 2020) — Future Work.
- **Layer sensitivity scaling**: Different transformer layers have different
  pruning tolerances (attention early layers vs MLP late layers). This would
  require per-layer calibration passes — Future Work.
- **Activation-aware Hessian**: SparseGPT computes H = X^T·X from input
  activations, which is not available in offline pruning. Our gradient-covariance
  approach is the best proxy without data.
- **Full H^{-1} materialisation**: For B ≤ 256, the O(B²) memory of H^{-1}
  is negligible. For larger blocks, a truly lazy (column-on-demand) Woodbury
  implementation would be needed.

### 8.3 When to Use Each Strategy

| Strategy | Use case |
|----------|----------|
| `kOneShot` | Quick magnitude-based or diagonal-Fisher pruning |
| `kIterative` | Gradual pruning with cubic schedule, best for moderate sparsity (≤80%) |
| `kBlockOBS` (with gradients) | Maximum accuracy at high sparsity (>80%), when gradient snapshots are available |
| `kBlockOBS` (without gradients) | Intermediate — cross-weight compensation via weight correlation only |

---

## 9. References

1. **LeCun, Y., Denker, J., & Solla, S. (1990).** "Optimal Brain Damage." *NeurIPS 2*, pp. 598–605.

2. **Hassibi, B., & Stork, D. G. (1993).** "Second Order Derivatives for Network Pruning: Optimal Brain Surgeon." *NeurIPS 5*, pp. 164–171.

3. **Theis, L., Korshunova, I., Tejani, A., & Huszar, F. (2018).** "Faster Gaze Prediction with Dense Networks and Fisher Pruning." *arXiv:1801.05787*.

4. **Zhu, M., & Gupta, S. (2017).** "To Prune, or Not to Prune: Exploring the Efficacy of Pruning for Model Compression." *arXiv:1710.01878*.

5. **Kunstner, F., Hennig, P., & Balles, L. (2019).** "Limitations of the Empirical Fisher Approximation for Natural Gradient Descent." *NeurIPS 32*.

6. **Schervish, M. J. (1995).** *Theory of Statistics*. Springer-Verlag.

7. **Hoare, C. A. R. (1961).** "Algorithm 65: Find." *Communications of the ACM, 4(7)*, pp. 321–322.

8. **Frantar, E., & Alistarh, D. (2023).** "SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot." *ICML 2023*.

9. **Singh, S. P., & Alistarh, D. (2020).** "WoodFisher: Efficient Second-Order Approximation for Neural Network Compression." *NeurIPS 33*.

10. **Kurtic, E., Campos, D., Nguyen, T., Frantar, E., Kurtz, M., Fineran, B., Goin, M., & Alistarh, D. (2022).** "The Optimal BERT Surgeon: Scalable and Accurate Second-Order Pruning for Large Language Models." *arXiv:2203.07259*.
