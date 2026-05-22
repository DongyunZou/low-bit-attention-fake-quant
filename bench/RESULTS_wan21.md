# Fake-quant attention precision on wan21 — fixed against CuTe-DSL Sage2 reference

## Setup
- Hardware: NVIDIA H100 80GB HBM3, CUDA 13.0, Triton 3.7, Torch 2.12.
- Reference: `torch.nn.functional.scaled_dot_product_attention` on raw BF16 Q/K/V.
- Workloads: `/home/dongyun/dataset/v-dit/wan21_p1/layer_0/timestep_{0,3,6,9,29,49}.pt`, each `(B=1, S=69120, H=40, D=128)` BF16.
- Metric: per-element MSE / RMSE / cosine on the flattened output tensor.

## Key bug fix (v3 → v4)

**Earlier versions of this report under-reported the kmeans + Q-smoothing gains by ~10×.** The bug:

```python
# WRONG (v3): qm correction goes through FP8-quantized K
score = (q_centered_deq + qm) @ k_smooth_deq.T
      = q_centered_deq @ k_smooth_deq.T + qm @ k_smooth_deq.T
                                          ^^^^^^^^^^^^^^^^^^^^
                                          K FP8 noise leaks here!

# CORRECT (v4, matches Sage2 CuTe-DSL): qm correction uses un-quantized K
score = q_centered_deq @ k_smooth_deq.T + qm @ k_smooth_unquantized.T
```

K is the dominant noise source in this pipeline (K-quant-only MSE = 1.88e-3 vs Q-quant-only = 1.34e-4). The buggy `(q_deq + qm) @ k_deq.T` factoring reintroduced K noise scaled by qm, drowning out the benefit of Q centering.

Fix: separated `K_smooth` (un-quantized BF16) from `K_deq` (FP8-cast) all the way to the kernel. The Triton kernel now takes both and computes the qm correction term inline against the un-quantized K_smooth.

## Validation vs CuTe-DSL Sage2 reference

| Method | Sage2 reference | This pipeline (P_off) | This pipeline (P_on) |
| ------ | --------------- | --------------------- | -------------------- |
| `k_smooth` only                              | MSE=1.71e-3 | **1.698e-3** ✓ | 1.718e-3 |
| + kmeans + q_smooth                          | MSE=7.21e-4 | **7.26e-4** ✓ | 7.20e-4 |
| MXFP8 + kmeans + q_smooth                    | MSE=6.71e-4 | **6.99e-4** ✓ | 6.92e-4 |
| MXFP8 + kmeans + q_smooth + V smoothing      | (n/a)       | 6.83e-4        | **6.76e-4** |

The pipeline now matches Sage2's CuTe-DSL kernel to within ~5% MSE. The small residual gap is plausibly from BF16 vs FP32 SDPA reference and minor scale formula differences.

## Real per-trick contributions (P_off track, P FP8 cast off)

| Step                                              | MSE       | cum MSE drop | step MSE drop | cos       |
| -------------------------------------------------- | --------- | ------------ | ------------- | --------- |
| baseline (fp8_block / off / no kmeans / no vsmooth) | 1.785e-03 | —            | —             | 0.996301  |
| + `k_only` smoothing (K -= mean(K, S))             | 1.698e-03 | −5%          | −5%           | 0.996482  |
| + `full` smoothing (Q -= per-group mean)            | 7.78e-04  | **−56%**     | **−54%**      | 0.998412  |
| + Q kmeans k=32 (flash-kmeans reorder)              | 7.26e-04  | −59%         | −7%           | 0.998513  |
| + switch QK to MXFP8 (per-D power-of-2 scale)       | 6.99e-04  | −61%         | −4%           | 0.998575  |
| + V per-block smoothing (Alg 2)                     | 6.83e-04  | **−62%**     | −2%           | 0.998590  |

**Q smoothing is the workhorse** (−54% MSE single step). Kmeans/MXFP8/V-smoothing each add a few percent on top.

P_on track (with P FP8 e4m3fn cast, Triton kernel):

| Step                                              | MSE       | cum MSE drop | cos       |
| -------------------------------------------------- | --------- | ------------ | --------- |
| baseline P_on                                       | 1.804e-03 | —            | 0.996260  |
| + k_only smoothing                                  | 1.718e-03 | −5%          | 0.996441  |
| + full smoothing                                    | 7.76e-04  | −57%         | 0.998416  |
| + kmeans k=32                                       | 7.20e-04  | −60%         | 0.998528  |
| + MXFP8 QK                                          | 6.92e-04  | −62%         | 0.998589  |
| + V per-block smoothing (full stack)                | **6.76e-04** | **−63%**  | **0.998605** |

## V smoothing impact (under corrected pipeline)

With the qm bug fixed, V smoothing now contributes a slightly larger fraction of total MSE because K noise no longer dominates:

| Statistic                | v3 (bug) | v4 (fixed) |
| ------------------------ | -------- | ---------- |
| mean Δcosine (V smooth on vs off) | +1.6e-5 | +1.5e-5  |
| mean ΔMSE_rel            | −0.95%   | −2.3%     |
| V smoothing helped       | 72/72    | 72/72     |

## Best configuration (production-realistic, all tricks on, P_on)

```
QK quant     : MXFP8 (e4m3fn, per-D power-of-2 scale, block_d=32)
V  quant     : FP8 per-channel (e4m3fn, one FP32 scale per (B,H,D))
Smoothing    : full   (K -= mean(K,S);  Q -= group_mean(Q, block_q=256))
Q kmeans     : k=32   (flash-kmeans Triton)
V smoothing  : per_block (block_s=64, Algorithm 2)
P requant    : on     (P → e4m3fn → BF16 → PV mma; v_scale[d] post-mul + C correction)
```

Resulting metrics on wan21 layer_0 (mean over 6 timesteps):
- MSE    = **6.76e-04**
- RMSE   = **2.54e-02**
- Cosine = **0.998605**

vs Sage2 CuTe-DSL reference best (MXFP8 + kmeans + q_smooth, w/o V smoothing): MSE=6.71e-4, cos=0.9990. **Our fake-quant matches their real kernel.**

## Reproduce

```bash
uv sync --extra dev --extra bench
uv run pytest                                  # 32 unit + smoke + Triton tests
uv run python bench/eval_wan21.py \
    --data-root /home/dongyun/dataset/v-dit/wan21_p1 \
    --output bench/results_wan21_v4.json
```

End-to-end runtime: ≈ 5 min on 1×H100 for the full 144-config sweep on 6 timesteps.
