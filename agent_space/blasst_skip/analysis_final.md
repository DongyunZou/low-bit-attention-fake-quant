# FP8 + BLASST Block-Skip Attention — Final Analysis (full 60 Wan2.1 workloads)

> Generated from the full 60-workload native sweep (`agent_space/blasst_skip/`)
> and the reordered sweep (`agent_space/blasst_skip_reordered/`). Reorder layout
> selected by the empirical ranking over **241 grids × 6 flatten orders** (1446
> scored): grid `t,h,w = 18×48×80`, block `2×8×8`, native flatten order
> `(t,h,w)` — see `layout_ranking.json`. Static-P/skip rungs normalize by the
> pre-quantization (true) row sum, matching the FP8 attention path; the numerator
> uses FP8-quantized P. Ground truth = `torch` SDPA bf16; all smoothing disabled.
> Prose authored via Codex (`analyze` routing); tables by
> `bench/make_analysis_tables.py` (`degradation_tables.md`).

## Final Analysis

Across all 60 real Wan2.1 non-causal workloads, the ablation ladder separates three effects cleanly. The bf16 PyTorch reference path already matches torch SDPA closely, with mean cosine 1.00000, relRMSE 2.29e-3, and the registered reference check passing 60/60 cases. Moving Q/K/V to per-head static e4m3 is the dominant accuracy cost: mean cosine drops to 0.99877 and relRMSE rises to 4.19e-2. Adding static e4m3 quantization of `P * 256` changes almost nothing, with mean cosine 0.99875 and relRMSE 4.22e-2. Thus, in this pure-PyTorch study with all smoothing disabled, the main precision loss is from static FP8 Q/K/V, not from the static probability path.

The native-layout skip sweep gives a direct safe-lambda read-off. For a conservative cosine floor of 0.998, `lambda <= 0.03` is safe and gives about 18% block skip. Relaxing to cosine >= 0.995 allows `lambda <= 0.2`, reaching about 37% skip. A cosine >= 0.99 budget supports `lambda <= 0.3`, reaching about 45% skip. These thresholds are empirical over the full 60-workload set, using BLASST per-`(M-tile,N-block)` tile skip with all-rows reduction.

Dropped attention mass is a strong predictor of skip-only error. The log-log Pearson correlation between dropped-mass p95 and skip-only relRMSE is `r = 0.97` over 519 points, indicating that the dropped-mass statistic is not merely descriptive: it tracks the numerical error induced by block skipping very tightly.

The space-time reorder is a positive result. The empirical search over 241 grids and 6 flatten orders selected `t,h,w = 18x48x80`, block `2x8x8`, native flatten order `(t,h,w)`. This layout roughly doubles diagonal attention mass, from 0.047 to 0.091. At matched accuracy it consistently increases skip rate: +4.1pp at `lambda=0.03`, +5.5pp at `lambda=0.1`, and +6.4pp at `lambda=0.3`. At higher lambdas, reordered even has both higher skip and higher cosine, e.g. `lambda=0.7` gives native `0.695 / 0.97794` versus reordered `0.750 / 0.97986`. The caveat is that `18x48x80` is empirically top-ranked but the true Wan grid is unconfirmed; nevertheless, the locality gain is data-driven.

Layer sensitivity is nonuniform. At `lambda=0.1`, layer 0 is most sensitive with cosine 0.99558, while layers 10, 20, 30, and 39 are 0.99951, 0.99814, 0.99888, and 0.99694 respectively.

Finally, the default bf16 matmul path is not a confounder: fp32 internal accumulation over exactly representable e4m3 values matches strict fp32 matmul to about `5e-6` cosine.

## Full per-λ degradation tables

### Native order (60 workloads), mean vs SDPA bf16

**Ablation ladder (no skip):**

| rung | cosine | relRMSE | RMSE | MSE | worst-workload cos |
|---|---|---|---|---|---|
| bf16_ref | 1.00000 | 2.293e-03 | 9.490e-04 | 1.020e-06 | 1.00000 |
| fp8_qkv | 0.99877 | 4.187e-02 | 1.747e-02 | 4.844e-04 | 0.99428 |
| fp8_static_p | 0.99875 | 4.217e-02 | 1.760e-02 | 4.904e-04 | 0.99421 |

**Block-skip ladder:**

| λ | skip-rate | cosine | relRMSE | RMSE | MSE | dropped-mass p95 | skip-only relRMSE |
|---|---|---|---|---|---|---|---|
| 0 | 0.000 | 0.99875 | 4.217e-02 | 1.760e-02 | 4.904e-04 | 0.000e+00 | 0.000e+00 |
| 0.001 | 0.064 | 0.99875 | 4.218e-02 | 1.760e-02 | 4.905e-04 | 8.582e-04 | 7.574e-04 |
| 0.003 | 0.087 | 0.99875 | 4.228e-02 | 1.764e-02 | 4.922e-04 | 3.424e-03 | 2.523e-03 |
| 0.01 | 0.127 | 0.99871 | 4.332e-02 | 1.805e-02 | 5.061e-04 | 1.446e-02 | 8.078e-03 |
| 0.03 | 0.183 | 0.99856 | 4.757e-02 | 1.969e-02 | 5.692e-04 | 4.182e-02 | 1.979e-02 |
| 0.1 | 0.284 | 0.99781 | 6.321e-02 | 2.555e-02 | 8.400e-04 | 1.208e-01 | 4.456e-02 |
| 0.2 | 0.374 | 0.99589 | 8.850e-02 | 3.507e-02 | 1.485e-03 | 2.189e-01 | 7.568e-02 |
| 0.3 | 0.446 | 0.99331 | 1.143e-01 | 4.500e-02 | 2.391e-03 | 3.124e-01 | 1.049e-01 |
| 0.5 | 0.575 | 0.98679 | 1.666e-01 | 6.568e-02 | 5.007e-03 | 4.816e-01 | 1.613e-01 |
| 0.7 | 0.695 | 0.97794 | 2.218e-01 | 8.730e-02 | 8.747e-03 | 6.305e-01 | 2.185e-01 |

### Space-time reordered (grid 18×48×80, block 2×8×8, order t,h,w), mean vs SDPA bf16

**Ablation ladder (no skip):** (matches native up to fp rounding — pure reindexing)

| rung | cosine | relRMSE | RMSE | MSE | worst-workload cos |
|---|---|---|---|---|---|
| bf16_ref | 1.00000 | 2.293e-03 | 9.490e-04 | 1.020e-06 | 1.00000 |
| fp8_qkv | 0.99877 | 4.187e-02 | 1.747e-02 | 4.844e-04 | 0.99428 |
| fp8_static_p | 0.99875 | 4.220e-02 | 1.761e-02 | 4.910e-04 | 0.99420 |

**Block-skip ladder:**

| λ | skip-rate | cosine | relRMSE | RMSE | MSE | dropped-mass p95 | skip-only relRMSE |
|---|---|---|---|---|---|---|---|
| 0 | 0.000 | 0.99875 | 4.220e-02 | 1.761e-02 | 4.910e-04 | 0.000e+00 | 0.000e+00 |
| 0.001 | 0.076 | 0.99875 | 4.222e-02 | 1.762e-02 | 4.913e-04 | 1.764e-03 | 1.236e-03 |
| 0.003 | 0.106 | 0.99874 | 4.243e-02 | 1.770e-02 | 4.939e-04 | 6.515e-03 | 3.688e-03 |
| 0.01 | 0.157 | 0.99869 | 4.409e-02 | 1.835e-02 | 5.149e-04 | 2.244e-02 | 1.063e-02 |
| 0.03 | 0.224 | 0.99847 | 4.985e-02 | 2.057e-02 | 6.046e-04 | 5.772e-02 | 2.413e-02 |
| 0.1 | 0.339 | 0.99746 | 6.912e-02 | 2.774e-02 | 9.580e-04 | 1.511e-01 | 5.169e-02 |
| 0.2 | 0.434 | 0.99544 | 9.453e-02 | 3.732e-02 | 1.638e-03 | 2.602e-01 | 8.173e-02 |
| 0.3 | 0.510 | 0.99311 | 1.185e-01 | 4.665e-02 | 2.494e-03 | 3.587e-01 | 1.090e-01 |
| 0.5 | 0.641 | 0.98717 | 1.670e-01 | 6.593e-02 | 4.850e-03 | 5.273e-01 | 1.614e-01 |
| 0.7 | 0.750 | 0.97986 | 2.143e-01 | 8.458e-02 | 7.810e-03 | 6.627e-01 | 2.104e-01 |

### Native vs reordered skip-rate at matched accuracy

| λ | native skip / cos | reordered skip / cos | Δ skip-rate |
|---|---|---|---|
| 0.03 | 0.183 / 0.99856 | 0.224 / 0.99847 | +4.1pp |
| 0.1 | 0.284 / 0.99781 | 0.339 / 0.99746 | +5.5pp |
| 0.3 | 0.446 / 0.99331 | 0.510 / 0.99311 | +6.4pp |
| 0.7 | 0.695 / 0.97794 | 0.750 / 0.97986 | +5.5pp (also higher cos) |

## Caveats
- The static-P and skip rungs normalize by the unquantized (true) softmax row
  sum and quantize only the P·V numerator, matching the FP8 attention path
  (cf. `p_requant_rows`). The `P*256` offset keeps mass loss small, so this
  normalization is numerically close to a self-normalized quantized softmax.
- The reorder grid 18×48×80 (order t,h,w) is the empirically top-ranked layout,
  not a confirmed Wan2.1 latent geometry (DEC-4 open). The positive locality
  result is data-driven (diagonal mass ~2× native); a confirmed grid could shift
  the magnitude.
- "fp8" here is a fake-quant accuracy proxy (quant→dequant, fp32 accumulation),
  not a kernel-accurate hardware result.
