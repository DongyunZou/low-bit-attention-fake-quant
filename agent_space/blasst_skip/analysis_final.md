# FP8 + BLASST Block-Skip Attention — Final Analysis (full 60 Wan2.1 workloads)

> Generated from the full 60-workload native sweep (`agent_space/blasst_skip/`)
> and the reordered sweep (`agent_space/blasst_skip_reordered/`, empirically
> selected grid t,h,w = 4×108×160, block 1×4×32; see `layout_ranking.json`).
> Ground truth = `torch` SDPA bf16. All smoothing disabled. Authored via Codex
> (`analyze` task routing).

## Final Analysis

Across all 60 real Wan2.1 non-causal workloads (`seqlen=69120`, 40 heads, `head_dim=128`), the ablation ladder shows a clean separation between quantization error and skip-induced error. The bf16 no-quant PyTorch baseline is effectively identical to SDPA bf16 ground truth, with mean cosine `1.00000`, relRMSE `2.29e-3`, and AC-2 passing `60/60`. Moving to per-head static e4m3 Q/K/V quantization introduces the dominant accuracy loss: mean cosine drops to `0.99877` and relRMSE rises to `4.19e-2`. Adding static e4m3 quantization of `P*256` changes almost nothing, yielding cosine `0.99875`, relRMSE `4.21e-2`, and worst-workload cosine `0.99421`. Thus, static probability quantization is not the limiting factor in this setup; the practical floor is already set by FP8 Q/K/V.

The BLASST tile-skip ladder gives a direct safe-lambda read-off. Because the no-skip FP8 floor is only `0.99875`, a `cos >= 0.999` target is unreachable without improving quantization. For `cos >= 0.998`, the safe region is `lambda <= 0.03`, giving about `18%` skip with mean cosine `0.99857` and relRMSE `4.76e-2`. Relaxing to `cos >= 0.995` permits `lambda <= 0.2`, about `37%` skip, while `cos >= 0.99` permits `lambda <= 0.3`, about `45%` skip. Beyond that, accuracy degrades rapidly: at `lambda=0.5`, skip reaches `57.5%`, but cosine falls to `0.98679`.

Dropped attention mass is a strong predictor of skip-only error. The p95 dropped mass rises monotonically from `8.6e-4` at `lambda=0.001` to `6.3e-1` at `lambda=0.7`, and correlates with skip-only relRMSE at Pearson `r=0.95`, or `0.97` on a log-log scale over `n=519` points. This supports dropped mass as a useful diagnostic for BLASST safety.

The native-vs-reordered comparison does not show a meaningful locality win. The empirically selected `4x108x160` reorder, using `1x4x32` blocks, tracks native within roughly `0.5pp` skip rate and `0.001` cosine at every lambda. At `lambda=0.1`, native gives skip `0.284`, cosine `0.99781`; reordered gives skip `0.286`, cosine `0.99775`. Attention is diffuse: the own-128 block holds only about `5%` of mass, and best reordered diagonal mass is `0.048` versus native `0.047`. This reorder therefore does not improve skip rate at matched accuracy.

Layer stratification shows nonuniform sensitivity. At `lambda=0.1`, layer 0 is most fragile with cosine `0.99558`, while layers 10, 20, 30, and 39 reach `0.99951`, `0.99814`, `0.99888`, and `0.99694`, respectively. Finally, the default bf16 matmul path, using fp32 internal accumulation over exactly representable e4m3 values, matches strict fp32 matmul to about `5e-6` cosine, so reported trends are not a matmul-precision artifact.

## Caveats

- The space-time reorder grid (4×108×160) is the empirically top-ranked genuine
  factorization, not a confirmed Wan2.1 latent geometry (DEC-4 remains open). The
  null reorder result holds for this grid; a different, confirmed grid could
  change it, though the diffuse-attention measurement (own-block mass ≈5%) bounds
  the achievable locality gain regardless.
- "fp8" here is a fake-quant accuracy proxy (quant→dequant, fp32 accumulation),
  not a kernel-accurate hardware result.
