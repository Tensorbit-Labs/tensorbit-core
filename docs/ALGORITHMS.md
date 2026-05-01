# Algorithms of Tensorbit Core

## 1. Problem Statement: Why Structured Pruning Matters for LLM Inference

Modern large language models (LLMs) like Llama 2, Mistral, and GPT-4 contain billions of
parameters. During inference on consumer hardware (the target of Tensorbit Labs: 8–16 GB RAM
devices), two bottlenecks dominate:

1. **Memory bandwidth** — reading all weights from RAM/VRAM overwhelms the memory bus.
2. **Compute throughput** — dense matrix multiplications waste cycles on near-zero weights
   that contribute negligibly to the output.

**Structured sparsity** addresses both simultaneously. By enforcing a hardware-friendly
N:M pattern (e.g., 2:4 — exactly 2 non-zero values in every contiguous group of 4), the
GPU can:

- Skip loading pruned weights entirely (2× bandwidth reduction for 2:4).
- Double matrix-multiply throughput via NVIDIA's Sparse Tensor Cores (Ampere/Hopper
  instruction `mma.sp`).

The challenge is **which weights to prune**. Random or naive magnitude-based pruning can
remove "quiet but load-bearing" weights — parameters with small magnitudes whose removal
disproportionately harms model accuracy. The Hessian-aware approach solves this.

---

## 2. EHAP: Efficient Hessian-Aware Pruning

### 2.1 Mathematical Foundation

Given a loss function L(w; D) parameterized by weights w and evaluated on dataset D,
the **local geometry** of the loss landscape around the current weight vector is
characterized by the **Hessian matrix**:

$$H_{ij} = \frac{\partial^2 L}{\partial w_i \partial w_j}$$

A second-order Taylor expansion reveals how much the loss changes when we perturb
the weights by δw:

$$L(w + \delta w) \approx L(w) + \nabla L^T \delta w + \frac{1}{2} \delta w^T H \delta w$$

At a local minimum, ∇L = 0, and the loss change is dominated by the quadratic form:

$$\Delta L \approx \frac{1}{2} \delta w^T H \delta w$$

If we prune weight w_j (i.e., set δw_j = -w_j), the loss increase is:

$$\Delta L_j \approx \frac{1}{2} w_j^2 H_{jj}$$

**This is the fundamental insight**: weights with small `w_j^2 · H_jj` can be removed
with minimal impact on the loss function. The Hessian diagonal encodes per-weight
"load-bearing capacity."

### 2.2 The Fisher Information Diagonal Approximation

Computing the full Hessian is O(N^2) in memory and O(N^3) in time for an N-parameter
model — infeasible for billion-parameter LLMs.

The **Empirical Fisher Information Matrix** is an approximation that replaces the
Hessian with the expectation of squared gradients:

$$F_{ij} = \mathbb{E}_{x \sim D} \left[ \frac{\partial L}{\partial w_i} \cdot \frac{\partial L}{\partial w_j} \right]$$

Under the assumption that the model's output distribution matches the true data
distribution, the Fisher is asymptotically equivalent to the Hessian at the optimum.
Crucially, we take only the **diagonal** — O(N) memory:

$$F_{ii} = \mathbb{E}_{x \sim D} \left[ \left(\frac{\partial L}{\partial w_i}\right)^2 \right]$$

### 2.3 The EHAP Importance Score

The EHAP (Efficient Hessian-Aware Pruning) sensitivity score for weight w_i is:

$$\boxed{s_i = w_i^2 \cdot (F_{ii} + \lambda)}$$

Where:

| Symbol | Meaning | Default |
|--------|---------|---------|
| w_i | Weight magnitude | — |
| F_ii | Diagonal Fisher Information | Computed from gradients |
| λ (lambda) | Damping factor for numerical stability | 0.01 |

The damping term λ prevents division-like instabilities when F_ii ≈ 0 (weights that
receive near-zero gradients but must be kept for architectural reasons, like bias terms
or embedding entries).

**Weights with the lowest s_i are pruned.**

### 2.4 Fisher Accumulation Algorithm

During training or fine-tuning, Fisher information is accumulated incrementally:

```
Initialize: F = zero vector of size N
For each batch:
    Compute gradient g = ∇L(w)
    Update: F[i] += α · g[i]^2        # α = 1 / accumulation_steps
```

This is an exponential running average with decay controlled by `accumulation_steps`.
The implementation in `include/tensorbit/core/ehap.hpp` (EHAPPruner<F>::accumulate_fisher) supports both:

- **GPU path**: Launches `fisher_accumulate_kernel` (1 thread per element, `fmaf`
  fused multiply-add for precision).
- **CPU path**: Element-wise loop with `__restrict__` annotations for autovectorization.

### 2.5 Importance Computation

Once Fisher is accumulated, importance scores couple weights with curvature:

```
For each weight w[i]:
    if Fisher diagonal available:
        s[i] = w[i]^2 · (F[i] + damping)
    else (magnitude fallback):
        s[i] = w[i]^2
```

GPU: `ehap_importance_kernel` — one thread per weight, zero shared memory.
CPU: Plain loop, vectorizable.

### 2.6 Mask Selection (Thresholding)

Given a target sparsity ratio r (fraction of weights to keep):

1. Find the (1-r) · N percentile of importance scores via `std::nth_element` (O(N)).
2. All weights with score below this threshold are marked for pruning.
3. Output: binary mask M where M[i] = 1 means "keep."

---

## 3. CORING: N:M Structured Sparsity

### 3.1 Motivation

Global (unstructured) sparsity — dropping the k least-important weights wherever they
are — produces irregular data access patterns that GPUs cannot accelerate. Every pruned
zero still occupies memory and must be skipped at runtime, incurring warp-divergence
penalties.

**N:M structured sparsity** constrains the pattern: divide weights into contiguous
groups of M, keep exactly N per group. This guarantees:

- **Regular memory access** — hardware can predict which elements are zero.
- **No warp divergence** — all threads in a warp follow the same index pattern.
- **2× throughput** on A100/H100 when N=2, M=4 (supported by `mma.sp` instruction).

### 3.2 Mask Generation Algorithm

Given importance scores s[0..N_elements-1] and N:M pattern:

```
For each group g in [0, N_elements/M):
    base = g · M
    Find top-N indices among s[base .. base+M-1]
    Emit mask byte: bit i = 1 if i is in top-N set
```

#### GPU: 2:4 Specialized Path

The 2:4 kernel (`nm_mask_2_4_kernel`) is optimized for Ampere's native instruction:

| Threads per block | Shared memory | Per-thread work | Occupancy bottleneck |
|-------------------|---------------|-----------------|---------------------|
| 256 | 0 bytes | One group per thread | Register pressure (~18 regs) |

Each thread loads 4 importance values into registers, finds the top-2 indices via
a fixed comparison tree (fully unrolled, no branches after compilation), and writes
a packed byte. This achieves near-100% theoretical occupancy on SM80/SM90.

#### GPU: Generic N:M Path

The generic kernel (`nm_mask_generic_kernel`) handles arbitrary N:M patterns for M ≤ 32:

| Threads per block | Shared memory | Per-thread work | Time complexity |
|-------------------|---------------|-----------------|-----------------|
| M (up to 32) | ~256 bytes (32×float + 32×int) | Rank computation | O(M^2) |

Algorithm:
1. Each of M threads loads one importance value into `__shared__ float s_vals[M]`.
2. Each thread counts how many elements have strictly higher value → its rank.
3. Tie-breaking: equal values are resolved by lower index winning (deterministic).
4. Thread 0 assembles the mask byte from ranks: if rank < N, bit is set.

#### CPU Fallback

For non-GPU execution or double-precision tensors:
1. Copy M elements per group into a `std::pair<float, int>` vector (value, original index).
2. Use `std::nth_element` to partition top-N to the front (O(M log M) per group,
   but M is small).
3. Assemble mask byte.

### 3.3 Mask Application

The mask is applied element-by-element:

```
For each weight w[i]:
    group = i / M
    offset = i % M
    if mask_byte[group] bit offset == 0:
        w[i] = 0
```

GPU: `apply_mask_kernel` — 1 thread per element, one division + one bit test + one
conditional store per thread. Divergence is bounded because both paths (keep vs prune)
are trivial.

### 3.4 Pruned Weight Counting

The count is computed analytically — no runtime overhead:

$$N_{\text{pruned}} = \frac{N_{\text{elements}}}{M} \cdot (M - N)$$

This is exact because `validate_config` ensures the tensor size is divisible by M.

---

## 4. End-to-End Pipeline

```
   .safetensors ──► [EHAP] ──► [CORING] ──► .tb file
   (dense weights)    │            │          (pruned + masks)
                      │            │
              Fisher diagonal   N:M bitmask
              (O(N) memory)    (N/M bytes)
```

### Mathematical Invariants

1. **Monotonicity**: If w_a has higher importance than w_b, CORING will never prefer
   w_b over w_a within the same group. The superposition of EHAP importance and
   CORING structural constraints is monotonic.

2. **Sparsity Guarantee**: After the pipeline, every contiguous group of M weights
   contains exactly N non-zero values. The ratio N/M is the **structural sparsity**
   of the model.

3. **Memory Footprint**: During pruning, peak memory is O(N) for weights + O(N) for
   Fisher diagonal + O(N/M) for masks. For a 7B parameter model:
   - FP32 weights: 28 GB
   - FP32 Fisher: 28 GB (during pruning only)
   - 2:4 mask: 1.75 GB
   - **Total peak**: ~58 GB (fits on a single A100-80GB)

4. **Numerical Stability**: The damping term λ prevents the importance score from
   collapsing to zero when Fisher information is near-zero (common in embedding
   layers or frozen parameters).

---

## 5. Why This Beats Magnitude-Based Pruning

Consider two weights in the same 2:4 group:

| Weight | |w| | F_ii | w^2 · F_ii | Magnitude rank | EHAP rank |
|--------|-----|------|------------|----------------|-----------|
| w_a | 0.05 | 100.0 | 0.25 | 2nd (pruned) | 1st (kept) |
| w_b | 0.10 | 1.0 | 0.01 | 1st (kept) | 2nd (pruned) |

Magnitude pruning keeps w_b (larger |w|) and prunes w_a. But w_a's high Fisher value
(100.0) indicates that small changes to w_a cause large changes in loss — it is
**load-bearing**. Pruning it would tank accuracy.

EHAP correctly identifies w_a as critical despite its small magnitude. This is the
essence of Hessian-aware pruning: coupling **size** (magnitude) with **sensitivity**
(curvature) to make informed decisions.

---

## 6. CUDA Kernel Performance Analysis

### 6.1 Roofline Model (A100-SXM4, 80 GB)

| Kernel | Arithmetic Intensity | Bound By | Theoretical Throughput |
|--------|---------------------|----------|----------------------|
| fisher_accumulate | 2 FLOP/element | Memory (HBM2e, 2 TB/s) | 500M elements/ms |
| ehap_importance | 3 FLOP/element | Memory | 333M elements/ms |
| nm_mask_2_4 | 4 FLOP/element | Compute (312 TFLOPS) | 78B elements/s |
| nm_mask_generic | O(M^2) FLOP/element | Compute | M=4: 1.3B elements/s |
| apply_mask | 0 FLOP/element (pure I/O) | Memory | 500M elements/ms |

The mask kernels are compute-bound on A100, not memory-bound, which is ideal —
they don't compete with weight I/O for HBM bandwidth during the pruning phase.

### 6.2 Shared Memory Usage

| Kernel | Shared Memory/Block | Max Blocks/SM (A100, 164 KB L1/SHMEM) |
|--------|--------------------|--------------------------------------|
| nm_mask_generic | 256 bytes (s_vals[32] + s_ranks[32]) | 256+ (not a limiting factor) |
| All others | 0 bytes | Limited by registers/threadblocks only |

---

## 7. References

1. LeCun, Denker, & Solla (1990). "Optimal Brain Damage." — Classical OBD framework
   using diagonal Hessian.
2. Hassibi & Stork (1993). "Optimal Brain Surgeon." — Full Hessian inverse for
   pruning. Impractical for LLMs but the theoretical foundation.
3. Theis et al. (2018). "Faster Gaze Prediction with Dense Networks and Fisher
   Pruning." — Introduced diagonal Fisher for modern CNNs.
4. NVIDIA (2021). "Accelerating Inference with Sparsity Using the NVIDIA Ampere
   Architecture." — 2:4 sparse tensor core specification.
5. Mishra et al. (2021). "Accelerating Sparse Deep Neural Networks." — N:M
   transposable fine-grained sparsity (hardware motivation for CORING).
