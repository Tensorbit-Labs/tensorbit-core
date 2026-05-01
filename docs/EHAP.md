# EHAP: Efficient Hessian-Aware Pruning — Complete Mathematical Exposition

## 1. Overview

Efficient Hessian-Aware Pruning (EHAP) is a second-order model compression algorithm that
identifies and removes redundant neural-network parameters by estimating each parameter's
impact on the loss function. Unlike magnitude-based pruning, which discards weights solely
by their size, EHAP couples the **magnitude** of a weight with the **local curvature** of
the loss landscape — a weight with small magnitude but large curvature (i.e., a "load-bearing"
weight) is preserved, while a weight with small curvature is pruned even if its magnitude
is moderate.

The algorithm is grounded in the **Optimal Brain Damage (OBD)** framework of LeCun et al.
(1990) and the **Optimal Brain Surgeon (OBS)** of Hassibi & Stork (1993), using the
**Empirical Fisher Information Matrix diagonal** as a computationally tractable proxy for
the full Hessian.

This document provides a rigorous mathematical derivation, implementation details for
the reference C++20 codebase at `include/tensorbit/core/ehap.hpp`, and a discussion of
design choices and their theoretical justification.

---

## 2. Mathematical Foundation

### 2.1 Second-Order Taylor Expansion

Let `L(w; D)` denote the loss of a neural network with weight vector `w ∈ R^N` on dataset
`D`. Consider perturbing a single weight `w_i` by setting it to zero (pruning it). The
change in weight is `δw_i = -w_i`.

The second-order Taylor expansion of the loss around the current weight vector is:

$$L(w + \delta w) \approx L(w) + \nabla L(w)^\top \delta w + \frac{1}{2} \delta w^\top H(w) \delta w \quad (1)$$

where `H(w)` is the Hessian matrix with entries

$$H_{ij} = \frac{\partial^2 L}{\partial w_i \partial w_j} \quad (2)$$

At a local minimum (or after sufficient training), the gradient vanishes: `∇L(w) ≈ 0`.
The first-order term drops out, and the loss change from pruning weight `w_i` is dominated
by the quadratic term:

$$\Delta L_i \approx \frac{1}{2} w_i^2 H_{ii} \quad (3)$$

**This is the fundamental OBD relationship** (LeCun et al., 1990, Equation 4): the
importance of a weight is proportional to its squared magnitude times the corresponding
diagonal entry of the Hessian.

### 2.2 The Fisher Information Approximation

The full Hessian `H` has `O(N²)` entries, making it intractable for large models (7B+
parameters). The **Empirical Fisher Information Matrix** provides a diagonal approximation:

$$F_{ii} = \mathbb{E}_{x \sim D}\left[ \left( \frac{\partial L}{\partial w_i} \right)^2 \right] \quad (4)$$

Under the assumption that the model's predictive distribution matches the true data
distribution (a property of a well-trained model), the Fisher Information is asymptotically
equivalent to the Hessian at the optimum (Schervish, 1995; Kunstner et al., 2019):

$$\lim_{t \to \infty} F_{ii}^{(t)} = H_{ii} \quad (5)$$

The diagonal Fisher thus serves as a tractable proxy: O(N) memory, computed from
first-order gradients alone, without requiring second-order derivatives.

### 2.3 The EHAP Importance Score

Combining the OBD framework (Equation 3) with the Fisher diagonal (Equation 4) and adding
a damping term `λ > 0` for numerical stability, the EHAP importance score is:

$$\boxed{s_i = \frac{1}{2} w_i^2 \cdot (F_{ii} + \lambda)} \quad (6)$$

Where:

| Symbol | Meaning | Reference |
|--------|---------|-----------|
| w_i | Weight value | — |
| F_ii | Diagonal Fisher Information | Equation 4 |
| λ (lambda) | Damping factor (default 0.01) | LeCun et al. 1990, Eq. 8 |

The constant factor `1/2` can be absorbed into the ranking (it does not affect the relative
ordering), so the implementation computes:

$$s_i = w_i^2 \cdot (F_{ii} + \lambda) \quad (6')$$

Weights with the **smallest** `s_i` are pruned first. The damping `λ` prevents the
importance score from collapsing to zero when `F_ii` is numerically zero (a common
occurrence in embedding layers, bias terms, and frozen parameters).

### 2.4 Alternative Importance Formulations

The implementation supports three importance-score variants, selectable via
`ImportanceMode`:

**a) OBD (kOBD)** — LeCun et al. (1990), Equation 4:
$$s_i^{(\text{OBD})} = w_i^2 \cdot (F_{ii} + \lambda) \quad (7)$$
This is the standard formulation. Weights with large magnitude AND large Fisher
(high curvature) are retained.

**b) OBS-style (kOBS)** — Hassibi & Stork (1993), diagonal approximation:
$$s_i^{(\text{OBS})} = \frac{w_i^2}{F_{ii} + \lambda} \quad (8)$$
Derived from the inverse-Hessian formulation of OBS (Equation 8 in Hassibi & Stork).
The OBS criterion directly measures the SAL (Saliency) of each weight — the loss increase
when that weight is pruned. Weights with small SAL are safe to prune. This formulation
is more sensitive to the Fisher value and tends to preserve weights in high-curvature
regions more aggressively than OBD.

**c) Normalized (kNormalized)** — Scale-robust variant:
$$s_i^{(\text{norm})} = \frac{w_i^2 \cdot (F_{ii} + \lambda)}{1 + w_i^2} \quad (9)$$
This formulation prevents extremely large weights from dominating the importance ranking,
which is important across layers with different weight distributions (e.g., attention vs.
feed-forward layers in transformers). The denominator `1 + w_i²` ensures the score is
bounded between `F_ii + λ` (as `w_i → ∞`) and `0` (as `w_i → 0`).

---

## 3. Fisher Accumulation

### 3.1 Exponential Moving Average (EMA)

The Fisher diagonal is accumulated incrementally across training batches using an
Exponential Moving Average:

$$F_{ii}^{(t)} = \beta \cdot F_{ii}^{(t-1)} + (1 - \beta) \cdot g_i^2 \quad (10)$$

where:

- `g_i = ∂L/∂w_i` is the gradient of weight `i` at the current batch.
- `β ∈ (0, 1]` is the EMA decay factor (default 0.99).
- `β = 1.0` corresponds to simple cumulative accumulation.

The scaling factor `α` (typically `1 / accumulation_steps`) controls the learning rate
of the Fisher estimate:

$$F_{ii}^{(t)} \leftarrow F_{ii}^{(t)} + \alpha \cdot g_i^2 \quad (\text{first step, or cumulative mode})$$

**References**: The Fisher EMA formulation follows Theis et al. (2018) who used an
exponential moving average of squared gradients for Fisher pruning in gaze prediction
networks. The `α` scaling is adapted from the original OBD/OBS work where gradient
statistics were computed over the full dataset.

### 3.2 Fisher Normalization

When `normalize_fisher = true`, the Fisher diagonal is L2-normalized per-layer before
computing importance scores:

$$\tilde{F}_{ii} = \frac{F_{ii}}{\|F\|_2}, \quad \|F\|_2 = \sqrt{\sum_j F_{jj}^2}$$

This prevents layers with systematically different gradient scales (e.g., early vs. late
layers, or attention vs. MLP) from having their importance scores dominated by raw Fisher
magnitude differences. Theis et al. (2018, Sec. 3.2) found that normalizing Fisher across
layers improved pruning stability.

---

## 4. Mask Selection

### 4.1 Thresholding via nth_element

Given importance scores `s_i` and a target **retention ratio** `ρ` (fraction of weights
to keep), the EHAP pruner selects weights whose score exceeds the `(1-ρ)·N` percentile:

```
Let K = ρ · N be the number of weights to keep.
Find the threshold τ such that exactly K weights satisfy s_i ≥ τ.
Set mask[i] = 1 if s_i ≥ τ, else 0.
```

The threshold is found using Hoare's **quickselect** (`std::nth_element`), which is O(N)
in the average case, compared to O(N log N) for full sorting.

**Justification**: Quickselect was recommended by Hassibi & Stork (1993) for the OBS
saliency computation, which requires finding the minimum-saliency weight to prune first.
For global pruning, the threshold is computed across all weights simultaneously.

### 4.2 Iterative Pruning with Cubic Schedule

When `prune_strategy = kIterative`, pruning is performed over `T` rounds using the
**Gradual Pruning Schedule** of Zhu & Gupta (2017):

$$\rho_t = \rho_{\text{final}} + (\rho_0 - \rho_{\text{final}}) \cdot \left(1 - \frac{t}{T}\right)^3 \quad (11)$$

where `ρ_0 = 1.0` (all weights kept initially), `ρ_final` is the target sparsity, and
`t ∈ [1, T]` is the current round.

The cubic schedule prunes aggressively in early rounds (when the model can compensate)
and conservatively in later rounds (when remaining weights are critical). After each
round, weight compensation (Section 5) is applied to the remaining weights.

Zhu & Gupta (2017) demonstrated that gradual pruning with a cubic schedule achieves
higher final accuracy than one-shot pruning at the same sparsity level, particularly
for sparsity ratios above 80%.

### 4.3 Blockwise Exact OBS Pruning (kBlockOBS)

When `prune_strategy = kBlockOBS`, the EHAP pruner employs the **blockwise exact OBS**
algorithm, inspired by SparseGPT (Frantar & Alistarh, 2023) and WoodFisher (Singh
& Alistarh, 2020). This is the mathematically correct formulation of Optimal Brain
Surgeon for large models — it computes the **inverse Hessian** for each block and
applies exact cross-weight compensation.

**Algorithm** (per block of size B):

1. **Hessian construction**: Approximate H for the block using the accumulated Fisher
   diagonal plus a low-rank off-diagonal correction estimated from weight
   self-correlation:

   $$H \approx \text{diag}(F_{\text{block}} + \lambda) + \alpha \cdot w_{\text{block}} \cdot w_{\text{block}}^\top \quad (12)$$

   where α is a small scaling factor (default 0.01). When α = 0, this is pure diagonal
   OBD; when α > 0, weight correlations introduce non-zero off-diagonal entries enabling
   true cross-weight compensation.

2. **Cholesky inversion**: Compute H^{-1} via Cholesky decomposition (Eigen::LLT). H is
   guaranteed symmetric positive-definite by construction (diagonal entries are strictly
   positive due to the damping term λ).

3. **Greedy OBS loop**: For each weight to prune within the block:

   a. **Saliency**: Find weight i with minimum `s_i = w_i^2 / [H^{-1}]_{ii}`. This is the
      exact OBS saliency criterion (Hassibi & Stork, 1993, Equation 8).

   b. **Compensation**: Apply the exact OBS correction to all remaining weights:

      $$\delta w_j = -\frac{w_i}{[H^{-1}]_{ii}} \cdot [H^{-1}]_{ji} \quad \forall j \quad (13)$$

      This minimizes the quadratic loss increase ‖w − w′‖²_H under the constraint that
      weight i is pruned.

   c. **Deflation**: Update H^{-1} via the Sherman-Morrison rank-1 formula to remove
      the pruned weight from the precision matrix:

      $$H^{-1} \leftarrow H^{-1} - \frac{H^{-1}_{:,i} \cdot H^{-1}_{i,:}}{H^{-1}_{ii}} \quad (14)$$

      This ensures that subsequent pruning decisions within the same block use the
      correct conditional precision matrix (the inverse Hessian for the remaining weights).

**Complexity**: O(B³) per block (Cholesky initialisation) + O(B³) for pruning
(B steps, each O(B²) for Sherman-Morrison). For B = 128, this is ~4M operations
per block — computationally feasible for models up to billions of parameters.

**When α > 0** (off-diagonal enabled), this is **true second-order pruning**:
the Hessian contains cross-weight interaction terms, the inverse captures the
full conditional precision, and the OBS compensation uses the exact off-diagonal
entries of H^{-1} — not a heuristic approximation.

**When α = 0** (diagonal only), the algorithm reduces to pure OBD within each
block: no cross-weight compensation, H^{-1}_{ij} = 0 for i ≠ j.

**References**: The blockwise formulation follows SparseGPT (Frantar & Alistarh, 2023),
where the Hessian is approximated by X^T X for linear layers. Here, without input data,
we use the Fisher diagonal regularised by weight self-correlation as a proxy. The
Sherman-Morrison deflation step is standard in OBS literature.

---

## 5. Weight Compensation

### 5.1 OBS Framework

The Optimal Brain Surgeon (Hassibi & Stork, 1993) provides an exact formula for the
optimal adjustment to remaining weights when a weight `w_k` is pruned:

$$\delta w_j = -\frac{w_k}{[H^{-1}]_{kk}} \cdot [H^{-1}]_{kj} \quad (12)$$

With the diagonal approximation `H_{kj} ≈ 0` for `k ≠ j`, this reduces to no cross-weight
compensation. However, even the diagonal OBS framework suggests compensating the **bias**
term of a layer — since the bias is connected to all output neurons, pruning a weight
effectively shifts the mean activation, which can be partially corrected by a bias update.

### 5.2 Bias Compensation (kBias)

For each layer, the sum of pruned weights is added to the bias term:

$$b \leftarrow b + \sum_{i : \text{pruned}} w_i$$

This is a first-order correction proposed by LeCun et al. (1990, Sec. 3) under the name
"output bias adjustment." The implementation applies this per-layer: the caller is
responsible for providing a per-layer weight vector, and the compensation adds the total
pruned magnitude to the last weight entry (assumed bias by convention).

### 5.3 Group Redistribution (kRedist)

Inspired by the OBS diagonal correction, the redistribution mode divides weights into
groups (default size 8) and, within each group, redistributes the total magnitude of
pruned weights to the kept weights proportionally to their Fisher values:

$$w_i^{(\text{kept})} \leftarrow w_i^{(\text{kept})} + \Delta_g \cdot \frac{F_{ii}}{\sum_{j \in \text{kept}} F_{jj}} \quad (15)$$

where `Δ_g = Σ_{pruned in group} w_j` is the total removed magnitude. This preserves
the group's average contribution while directing the compensation to the weights with
the highest Fisher sensitivity.

---

## 6. Implementation Reference

### 6.1 Source Location

`include/tensorbit/core/ehap.hpp`

### 6.2 Key Classes

| Class | Purpose |
|-------|---------|
| `EHAPConfig` | Configuration struct with damping, sparsity, EMA decay, importance mode, pruning strategy, compensation mode, block-OBS settings |
| `EHAPPruner<F>` | Template class for `float`/`double` precision |
| `ImportanceMode` | Enum: `kOBD`, `kOBS`, `kNormalized` |
| `PruneStrategy` | Enum: `kOneShot`, `kIterative`, `kBlockOBS` |
| `CompensationMode` | Enum: `kNone`, `kBias`, `kRedist` |

### 6.3 Core Pipeline Methods

| Method | Algorithm | Reference |
|--------|----------|-----------|
| `accumulate_fisher(g, α)` | EMA Fisher diagonal: F ← βF + (1-β)g² | Theis et al. 2018 |
| `compute_importance(w, out)` | s_i per chosen importance mode | LeCun et al. 1990; Hassibi & Stork 1993 |
| `select_pruning_mask(s, mask)` | O(N) threshold via nth_element | Hoare 1961 (quickselect) |
| `apply_mask(w, mask)` | Element-wise zeroing | — |
| `compensate_weights(w, m, s)` | Bias update or group redistribution | LeCun et al. 1990 |
| `prune(w)` | Full pipeline: dispatches to one-shot, iterative, or block-OBS | — |
| `prune_one_shot(w)` | Single-pass: importance → mask → apply → compensate | O(N) |
| `prune_iterative(w)` | Cubic-schedule multi-round pruning | O(T·N log N) |
| `prune_block_obs(w)` | Blockwise exact OBS with Cholesky + Sherman-Morrison | O(N·B²) |

### 6.4 GPU Acceleration

When `F = float` and CUDA is enabled, `accumulate_fisher` and `compute_importance`
dispatch to CUDA kernels in `src/kernels.cu`. The GPU kernels use `fmaf()` fused
multiply-add for IEEE-754 single-precision accuracy. Double-precision (`F = double`)
currently uses the CPU path exclusively.

---

## 7. Usage Example

```cpp
EHAPConfig cfg;
cfg.damping            = 0.01f;
cfg.sparsity_ratio     = 0.5f;       // keep 50%
cfg.ema_decay          = 0.99f;
cfg.importance_mode    = ImportanceMode::kOBD;
cfg.prune_strategy     = PruneStrategy::kIterative;
cfg.prune_rounds       = 5;
cfg.compensation_mode  = CompensationMode::kRedist;

EHAPPruner<float> pruner(cfg);

// Accumulate Fisher from training gradients batch-by-batch
for (auto& batch : batches) {
    auto grads = compute_gradients(batch);
    pruner.accumulate_fisher(grads, 1.0f / batches.size());
}

// Execute iterative pruning with compensation
auto result = pruner.prune(model_weights);
if (result.has_value())
    std::printf("Pruned %zu weights\n", result.value());
```

---

## 8. References

1. **LeCun, Y., Denker, J., & Solla, S. (1990).** "Optimal Brain Damage." *Advances in Neural Information Processing Systems (NeurIPS) 2*, pp. 598–605. — Introduces the diagonal-Hessian pruning framework and the `w_i² · H_ii` saliency criterion.

2. **Hassibi, B., & Stork, D. G. (1993).** "Second Order Derivatives for Network Pruning: Optimal Brain Surgeon." *Advances in Neural Information Processing Systems (NeurIPS) 5*, pp. 164–171. — Extends OBD with the full inverse-Hessian correction (Equation 12) and the SAL saliency measure.

3. **Theis, L., Korshunova, I., Tejani, A., & Huszar, F. (2018).** "Faster Gaze Prediction with Dense Networks and Fisher Pruning." *arXiv:1801.05787*. — Introduces diagonal Fisher pruning for modern CNNs; the EMA Fisher accumulation (Equation 10) and per-layer normalization technique.

4. **Zhu, M., & Gupta, S. (2017).** "To Prune, or Not to Prune: Exploring the Efficacy of Pruning for Model Compression." *arXiv:1710.01878*. — Proposes the cubic Gradual Pruning Schedule (Equation 11) and demonstrates iterative pruning outperforms one-shot pruning.

5. **Kunstner, F., Hennig, P., & Balles, L. (2019).** "Limitations of the Empirical Fisher Approximation for Natural Gradient Descent." *Advances in Neural Information Processing Systems (NeurIPS) 32*. — Rigorous analysis of when the Empirical Fisher approximates the Hessian (Equation 5).

6. **Schervish, M. J. (1995).** *Theory of Statistics*. Springer-Verlag. — Standard reference for Fisher Information and its asymptotic relationship to the Hessian.

7. **Hoare, C. A. R. (1961).** "Algorithm 65: Find." *Communications of the ACM, 4(7)*, pp. 321–322. — Quickselect algorithm used for O(N) threshold selection.

8. **Frantar, E., & Alistarh, D. (2023).** "SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot." *International Conference on Machine Learning (ICML) 2023*. — Introduces the blockwise exact OBS approach for LLM pruning using Cholesky decomposition of the Hessian approximated by X^T X. The `prune_block_obs()` method adapts this to use Fisher diagonal + weight self-correlation when input data X is unavailable.

9. **Singh, S. P., & Alistarh, D. (2020).** "WoodFisher: Efficient Second-Order Approximation for Neural Network Compression." *Advances in Neural Information Processing Systems (NeurIPS) 33*. — Demonstrates that the inverse Fisher can be efficiently approximated and maintained under weight removal via Sherman-Morrison-Woodbury updates. The H^{-1} deflation step in blockwise OBS follows this approach.

10. **Kurtic, E., Campos, D., Nguyen, T., Frantar, E., Kurtz, M., Fineran, B., Goin, M., & Alistarh, D. (2022).** "The Optimal BERT Surgeon: Scalable and Accurate Second-Order Pruning for Large Language Models." *arXiv:2203.07259*. — Extends OBS to transformer architectures with block-diagonal Hessian approximations, motivating the per-block processing strategy.
