# FP8 + BLASST Block-Skip Attention Accuracy Study On Wan2.1 (Pure-PyTorch Simulator)

## Goal Description

Build a pure-PyTorch, kernel-free, memory-safe **tiled** attention simulator that reproduces the **BLASST** block-skipping algorithm (arXiv:2512.12087, "Dynamic Blocked Attention Sparsity via Softmax Thresholding") **combined with full fp8 quantization** (Q, K, P, V all 8-bit), and measure how output accuracy degrades — relative to `torch` SDPA — as the BLASST skip threshold `λ` is swept, across **all 60 real Wan2.1 non-causal attention workloads** at `~/dataset/v-dit/wan21`. Optionally evaluate the user's **space-time token reordering** to test whether spatially/temporally coherent 128-token blocks raise the skip rate at equal accuracy.

This is a measurement study, not a kernel-shipping task. The deliverable is a faithful simulator plus a results dataset (RMSE, MSE, cosine, relative RMSE) and degradation-vs-`λ` curves, produced over the real workloads with `torch` SDPA as the numerical ground truth.

The combination being studied is novel: **BLASST is bf16-only in the paper and uses no token reordering**; layering fp8 (Q/K/V/P) and the space-time reordering on top of BLASST sparsity is the user's contribution under test.

### Fixed constraints (deterministic design — these are specified by the user/paper and are NOT open choices)

- QK attention tile size is exactly **128 × 128** (M-tile = 128 query rows, N-block = 128 keys).
- Q, K, V are quantized to **fp8 e4m3** with **per-head static amax scaling** (`scale = amax_per_head / 448`, quantize `x/scale → e4m3`, dequantize `× scale`). Per-head, not per-channel; this is calibration, not smoothing.
- P (attention probabilities) uses the **static** scheme `P_q = dequant(quant(P × 256, e4m3)) / 256`. **No dynamic P quantization.** (This mirrors the Hopper fp8 path's `MAX_OFFSET=8`, which expands the per-block softmax-weight range to ≈256.)
- All "smoothing"/accuracy tricks are **disabled**: no Hadamard/rotation, no per-channel scale migration, no softcap, no `score_mod`.
- Block-skip follows **BLASST exactly**: skip N-block `j` of M-tile `i` iff `m̃ᵢ⁽ʲ⁾ − mᵢ⁽ʲ⁾ < ln(λ)`, where `m̃` is the block's local max score and `m` is the **running (online) row-max**. Skip is **pre-softmax**: the running denominator (row-sum `l`) is **not** updated for skipped blocks; their softmax, V-load, and P·V matmul are all omitted.
- Attention is **non-causal** (video DiT self-attention).
- Ground truth is **`torch.nn.functional.scaled_dot_product_attention`**, `scale = 1/sqrt(128)`, bf16 inputs.
- Development and execution pinned to **GPU3** (`CUDA_VISIBLE_DEVICES=3`).
- Metrics reported: **RMSE, MSE, cosine** (plus relative RMSE), over the full workload set.

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for deterministic verification.

- AC-1: The simulator is memory-safe and runs to completion on all 60 workloads on a single GPU, never materializing a full `[seqlen, seqlen]` (69120²) score or probability matrix.
  - Positive Tests (expected to PASS):
    - A run over one full `[1, 69120, 40, 128]` workload completes on GPU3 and peak GPU memory stays below a fixed bound (e.g. well under the device capacity), verified via `torch.cuda.max_memory_allocated()`.
    - A run over all 60 files completes and writes a results record per `(file, layer, timestep)`.
  - Negative Tests (expected to FAIL):
    - Any attempt to allocate a tensor whose shape contains `seqlen × seqlen` (e.g. a `[*, 69120, 69120]` scores buffer) is rejected by an explicit guard / assertion.
    - A code path that calls `attention_ref` from `flash_attn/cute/testing.py` on the full seqlen (which materializes full scores and OOMs) is not on the production sweep path.
  - AC-1.1: SDPA ground truth is itself memory-safe.
    - Positive: SDPA is invoked with an explicitly selected memory-efficient/flash backend and completes at full seqlen.
    - Negative: A configuration that forces the math (non-flash) SDPA backend at full seqlen, causing OOM, is flagged rather than silently attempted.

- AC-2: The simulator's reference (no-quant, no-skip) path is validated against ground truth with pre-registered, falsifiable thresholds.
  - Positive Tests (expected to PASS):
    - Full-seqlen, bf16, no-quant/no-skip simulator output (L1) vs SDPA (L0): global cosine ≥ 0.999 AND relative RMSE ≤ 2e-2 on every workload.
    - On a cropped sequence (length 4096), the fp32 no-quant/no-skip simulator matches `attention_ref(..., upcast=True)` with cosine ≥ 0.9999 AND relative RMSE ≤ 1e-3.
  - Negative Tests (expected to FAIL):
    - A simulator with an intentionally wrong softmax scale (e.g. `1/d` instead of `1/sqrt(d)`) fails the L1-vs-L0 thresholds.
    - A simulator with a transposed/incorrect online-softmax rescale fails the cropped fp32 vs `attention_ref` check.

- AC-3: The block-skip predicate reproduces BLASST exactly.
  - Positive Tests (expected to PASS):
    - The skip condition is `m̃ᵢ⁽ʲ⁾ − mᵢ⁽ʲ⁾ < ln(λ)` using the running (online) row-max, processing N-blocks in natural sequence order.
    - Skipped blocks contribute nothing to the running denominator `l` and nothing to the output accumulator (pre-softmax skip), confirmed by a unit test on a tiny hand-checkable case.
    - At `λ = 0` (equivalently `ln(λ) = −∞`) no block is skipped and L4 output equals the L3 (no-skip fp8) output bit-for-bit.
  - Negative Tests (expected to FAIL):
    - A variant that updates `l` for skipped blocks (i.e. keeps them in the denominator) diverges from the BLASST reference beyond tolerance on the unit case.
    - A variant that thresholds on the final row-max instead of the running row-max produces a different skip mask on a case where running and final maxima differ, and is flagged as non-BLASST.
  - AC-3.1: Empty-row safeguard is defined and reported.
    - Positive: If a threshold would skip every block for an M-tile, the single largest-`m̃` N-block for that M-tile is force-kept; the count of such force-keeps is recorded.
    - Negative: A run that produces a row with zero kept blocks (NaN/inf output) is rejected by an assertion.

- AC-4: The fp8 numerics honor the exact rules.
  - Positive Tests (expected to PASS):
    - Q, K, V are quantized to e4m3 with per-head static amax scaling; round-trip error and per-tensor saturation rate are reported for Q, K, V, and `P×256` separately.
    - P is quantized as `dequant(quant(P × 256, e4m3)) / 256` (static), applied to per-block online probabilities `exp(score − m_running)`.
    - QK^T and P·V matmuls accumulate in fp32 over the dequantized fp8 values (fp8 inputs, fp32 accumulate); no `torch` float8 matmul kernel is used.
    - softcap, `score_mod`, and any Hadamard/per-channel smoothing are absent from the active path.
  - Negative Tests (expected to FAIL):
    - A configuration enabling softcap, `score_mod`, or per-channel/Hadamard smoothing is rejected by a guard.
    - A configuration using dynamic per-block P scaling (instead of the static ×256) is rejected.
    - A tile size other than 128 for M or N is rejected.

- AC-5: The threshold sweep and ablation ladder produce the full metrics dataset over all 60 workloads, written to `agent_space/`.
  - Positive Tests (expected to PASS):
    - For each `λ` in the sweep grid and each workload, the run records RMSE, MSE, cosine, and relative RMSE for the ablation ladder {L0 SDPA bf16; L1 no-quant/no-skip; L2 fp8-QKV no static-P no skip; L3 fp8-QKV + static-P no skip; L4 fp8 + static-P + skip@λ}, all measured vs L0.
    - Skip-induced error is reported as L4(λ) vs L3, and total fp8-quant error as L3 vs L0.
    - Each record also includes: skip-rate (fraction of (M-tile, N-block) pairs skipped), per-row total dropped softmax mass (mean/median/p95/max), force-keep count, and Q/K/V/`P×256` saturation and fp8-underflow-zero rates.
    - Metrics are aggregated globally AND stratified per layer and per timestep (mean, median, p95, worst-case), and cosine is reported both as a per-query-row distribution and as a global flattened value.
    - All artifacts (per-workload table as CSV/JSON + a summary) are written under `agent_space/`.
  - Negative Tests (expected to FAIL):
    - A run that reports only a single global mean (no p95/worst-case, no per-layer/per-timestep stratification) does not satisfy AC-5.
    - A run missing the L3 no-skip fp8 baseline (so skip error cannot be isolated) does not satisfy AC-5.
    - A cosine computed only as a single flattened scalar (hiding per-row collapse) does not satisfy AC-5.

- AC-6: (Conditional on the reordering arm, see DEC-4) Space-time reordering is correct and validated.
  - Positive Tests (expected to PASS):
    - The reorder uses a configurable `(t, h, w)` with `t·h·w == 69120` asserted (default `20×48×72`), permutes Q, K, V identically, and inverse-permutes the output before any metric is computed.
    - With reordering applied but skip disabled (`λ=0`), the output equals the native-order no-skip output after inverse permutation (within fp tolerance) — proving the permutation is a pure reindexing.
    - An empirical layout-validation step ranks candidate `(t,h,w)` factorizations (and flatten order) by near-diagonal attention-mass concentration and reports which maximizes locality.
  - Negative Tests (expected to FAIL):
    - A reorder that permutes Q/K/V inconsistently, or omits the output inverse permutation, fails the `λ=0` equivalence check.
    - A `(t,h,w)` whose product ≠ 69120 is rejected by the assertion.

- AC-7: Reporting is exploratory (no hard pass/fail accuracy bar baked into criteria).
  - Positive Tests (expected to PASS):
    - The deliverable presents degradation curves (RMSE/MSE/cosine/relRMSE vs `λ`) and the skip-rate-vs-accuracy trade-off, allowing the safe threshold to be read off; native vs reordered skip-rate at matched accuracy is shown when the reordering arm runs.
  - Negative Tests (expected to FAIL):
    - A run that hard-fails (errors/aborts) merely because accuracy at some `λ` is poor — instead of recording it as a data point — violates the exploratory intent.

## Path Boundaries

Path boundaries define the acceptable range of implementation quality and choices. The user's specification is highly deterministic, so several bounds converge.

### Upper Bound (Maximum Acceptable Scope)
A pure-PyTorch tiled simulator that: reproduces BLASST skipping exactly; implements the full fp8 pipeline (e4m3 per-head Q/K/V, static `P×256` e4m3); runs the full ablation ladder (L0–L4) over all 60 workloads across the `λ` grid; reports RMSE/MSE/cosine/relRMSE with global + per-layer + per-timestep stratification (mean/median/p95/worst-case) plus skip-rate, dropped-mass, force-keep, and saturation diagnostics; validates correctness against SDPA and cropped `attention_ref`; runs the optional space-time reordering arm with empirical `(t,h,w)` locality validation and native-vs-reordered comparison; and writes a clean results dataset and summary to `agent_space/`.

### Lower Bound (Minimum Acceptable Scope)
A pure-PyTorch tiled simulator that: validates against SDPA (AC-2); implements e4m3 per-head Q/K/V + static `P×256` and BLASST skipping exactly (AC-3, AC-4); runs the ablation ladder including the mandatory L3 no-skip fp8 baseline over all 60 workloads across the `λ` grid (AC-5); reports RMSE/MSE/cosine/relRMSE aggregated globally and stratified per layer/timestep with p95/worst-case, plus skip-rate and dropped-mass; and writes results to `agent_space/`. The reordering arm (AC-6) may be deferred per DEC-4.

### Allowed Choices
- Can use: PyTorch (CPU loaders + GPU compute), `einops`, `numpy`/`pandas` for aggregation, `matplotlib` for optional plots, and existing repo helpers (`attention_ref` for cropped validation, the fp8 cast/descale patterns from `benchmark_flash_attention_fp8.py`, FLOPS helpers from `bench_utils.py`). May choose the internal tiling granularity (e.g. one M-tile vs all keys at once) as long as memory stays bounded and the BLASST online semantics are preserved. May choose the results serialization format (CSV/JSON) and the exact `λ` grid points (a geometric grid covering no-skip through a failure region).
- Cannot use: any CuTe/CUDA kernel modification (this is a pure-PyTorch study); dynamic P quantization; any smoothing/softcap/`score_mod`; tile sizes other than 128×128; the final row-max (instead of running row-max) in the skip predicate; a single-global-mean-only reporting scheme; materialization of the full 69120² matrix.

> **Note on Deterministic Designs**: The numerics (e4m3 per-head Q/K/V, static `P×256`), the skip predicate (BLASST `m̃ − m < ln(λ)`, pre-softmax, running max), the tile size (128), the ground truth (`torch` SDPA bf16), and the metrics (RMSE/MSE/cosine) are fixed by the user and the paper. Upper and lower bounds differ mainly in stratification depth and whether the optional reordering arm is included.

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are conceptual suggestions, not prescriptive requirements.

### Conceptual Approach

Per `(file, head)`, stream over the query tiles and keys without materializing full attention. One workable streaming form per M-tile of 128 query rows:

```
load Q_tile (128 x 128), all K (69120 x 128), all V (69120 x 128) for this head
fake-quant Q_tile, K, V to e4m3 with per-head amax scale; dequantize to fp32
m = -inf (128,)   ;  l = 0 (128,)   ;  acc = 0 (128 x 128)
for j in range(540):                 # N-blocks of 128 keys, natural order
    S_j = (Q_tile @ K_block_j^T) * (1/sqrt(128))   # 128 x 128, fp32 accumulate
    m_tilde = rowmax(S_j)                           # block local max (128,)
    m_new = max(m, m_tilde)                          # running max (BLASST)
    if (m_tilde - m_new) < ln(lambda):               # BLASST skip: pre-softmax
        continue                                     #   no update to l, acc
    # rescale running state to the new max
    alpha = exp(m - m_new); l *= alpha; acc *= alpha[:, None]
    P_j = exp(S_j - m_new[:, None])                  # online (un-normalized) probs in (0,1]
    P_q = dequant(quant(P_j * 256, e4m3)) / 256      # STATIC P quantization
    l += rowsum(P_q)
    acc += P_q @ V_block_j                           # fp32 accumulate
    m = m_new
# empty-row safeguard: if a row never accepted a block, force-keep its argmax block
O_tile = acc / l[:, None]                            # normalize at the end
```

Ablation ladder is the same loop with switches: L1 = skip the fake-quant and the `P×256` step and the skip test; L2 = fake-quant Q/K/V only; L3 = fake-quant + `P×256`, skip test off; L4 = everything on. The dropped-mass diagnostic is computed from a separate no-skip pass's final `(m, l)` so dropped mass is measured against the true normalization.

Memory note: `Q_tile @ K^T` for one M-tile against all keys is `128 × 69120` fp32 (≈35 MB) — safe. Never form `69120 × 69120`. SDPA ground truth is called with a pinned memory-efficient/flash backend so it also avoids the full matrix.

Validation: confirm L1 ≈ SDPA at full seqlen, and confirm the fp32 no-quant path matches `attention_ref(upcast=True)` on a cropped length-4096 case (since `attention_ref` materializes full scores and OOMs at 69120).

### Relevant References
- `flash_attn/cute/testing.py` — `attention_ref(...)` fp32 oracle with `q/k/v_descale`, `return_lse`, `upcast`; use for CROPPED validation only (it materializes full scores → OOM at 69120).
- `flash_attn/cute/bench_utils.py` — minimal `attention_ref(q,k,v,causal)` and `flops(...)` / bandwidth helpers.
- `flash_attn/cute/softmax.py` — online-softmax `row_max`/`row_sum` and `finalize()` (LSE) semantics to mirror in pure PyTorch.
- `flash_attn/cute/benchmark_flash_attention_fp8.py` — e4m3/e5m2 cast and per-`(batch,nheads)` descale broadcast pattern; PyTorch baseline `attention_pytorch`.
- `tests/cute/test_mask_mod.py` — `F.scaled_dot_product_attention(q,k,v,scale=...)` call pattern (layout `(b,h,s,d)`) and the `assert_fwd_matches_reference` tolerance/metric helpers.
- `hopper/softmax.h` — `MAX_OFFSET=8` (expands per-block softmax-weight range to ≈256), the hardware analogue of the static `P×256` rule.
- `~/dataset/v-dit/wan21_p{1,2}/layer_{0,10,20,30,39}/timestep_{0,3,6,9,29,49}.pt` — 60 files, dict `{query,key,value}`, each `[1, 69120, 40, 128]` bf16, non-causal.
- `agent_space/` — destination for results tables, summaries, and any plots (per the repo scratch convention).

## Dependencies and Sequence

### Milestones
1. Foundations: data + numerics primitives.
   - Phase A: Confirm the BLASST Algorithm-1 skip criterion verbatim and lock its formula (`m̃ − m < ln(λ)`, running max, pre-softmax).
   - Phase B: Build the wan21 loader (CPU, mmap) and the per-head amax fp8 fake-quant utilities (e4m3, `scale=amax/448`, quant/dequant) with saturation/zero diagnostics.
2. Correct tiled core + ground truth.
   - Phase A: Implement the memory-safe tiled online-softmax core (L1 path), fp32 accumulate.
   - Phase B: Implement the SDPA ground-truth harness (pinned backend, bf16) and pass AC-2 validation (full-seqlen L1≈L0; cropped fp32 vs `attention_ref`).
3. fp8 + BLASST integration.
   - Phase A: Add the fp8 pipeline (L2, then static `P×256` for L3).
   - Phase B: Add the BLASST skip predicate (L4), the empty-row safeguard, and the dropped-mass diagnostic.
4. Sweep + reporting.
   - Phase A: Metrics module (RMSE/MSE/cosine/relRMSE, per-row cosine, stratified aggregation).
   - Phase B: Threshold-sweep driver over all 60 files × `λ` grid on GPU3, writing results to `agent_space/`.
5. Optional reordering arm + analysis (gated by DEC-4).
   - Phase A: Configurable `(t,h,w)` reorder + inverse permutation + empirical locality validation.
   - Phase B: Final analysis — safe-`λ` read-off, native-vs-reordered skip-rate at matched accuracy, dropped-mass-vs-error correlation.

Dependencies: Milestone 2 depends on Milestone 1. Milestone 3 depends on the validated core from Milestone 2 (so quant/skip error is measured against a trusted baseline). Milestone 4 depends on Milestone 3 (needs all ladder rungs). Milestone 5 depends on Milestone 4 (reuses the sweep driver and metrics, compares against native-order results).

## Task Breakdown

Each task includes exactly one routing tag (`coding` = implemented by Claude; `analyze` = executed via Codex `/humanize:ask-codex`).

| Task ID | Description | Target AC | Tag (`coding`/`analyze`) | Depends On |
|---------|-------------|-----------|----------------------------|------------|
| task1 | Extract and lock BLASST Algorithm-1 skip criterion verbatim (formula, running-max usage, pre-softmax denominator handling, λ sweep convention); reconcile with the planned predicate | AC-3 | analyze | - |
| task2 | wan21 loader (CPU/mmap) + per-head static amax fp8 fake-quant helpers (e4m3, scale=amax/448, quant/dequant) + Q/K/V/P×256 saturation & underflow-zero diagnostics | AC-4 | coding | task1 |
| task3 | Memory-safe tiled online-softmax attention core (L1), fp32 accumulate, with full-matrix-allocation guard | AC-1 | coding | task2 |
| task4 | SDPA ground-truth harness (pinned memory-efficient/flash backend, bf16, scale=1/sqrt(128)) + AC-2 validation: full-seqlen L1≈L0 and cropped fp32 vs attention_ref | AC-2 | coding | task3 |
| task5 | fp8 numerics integration: L2 (fp8 Q/K/V) and L3 (+ static P×256 on per-block online probs) | AC-4 | coding | task4 |
| task6 | BLASST skip predicate integration (L4): m̃−m_running<ln(λ), pre-softmax (no denom update), empty-row force-keep, dropped-mass-per-row diagnostic | AC-3 | coding | task5 |
| task7 | Metrics module: RMSE, MSE, cosine (per-row distribution + global), relative RMSE; global + per-layer + per-timestep aggregation (mean/median/p95/worst) | AC-5 | coding | task4 |
| task8 | Threshold-sweep driver over all 60 files × λ grid on GPU3; emit per-workload CSV/JSON + summary to agent_space/; record env (GPU model, torch/CUDA versions, SDPA backend, allow_tf32) | AC-5, AC-7 | coding | task6, task7 |
| task9 | Optional space-time reorder arm: configurable (t,h,w) (assert product=69120, default 20×48×72), identical Q/K/V permute + output inverse-permute, λ=0 equivalence check, empirical (t,h,w) locality ranking | AC-6 | coding | task8 |
| task10 | Final results analysis: safe-λ read-off from curves, native-vs-reordered skip-rate at matched accuracy, dropped-mass-vs-output-error correlation, write-up to agent_space/ | AC-7 | analyze | task8, task9 |

## Claude-Codex Deliberation

### Agreements
- The ablation ladder is necessary: L3 (no-skip fp8 + static-P baseline) must be reported so skip-induced error (L4 vs L3) is isolated from fp8-quant error (L3 vs L0).
- The full 69120² attention matrix must never be materialized; both the simulator and the SDPA ground truth must stay tiled/streamed, and the SDPA backend must be pinned and recorded.
- Reordering must not contaminate the primary result: it requires a verified `(t,h,w)` layout, an identical Q/K/V permutation, and an output inverse permutation before metrics.
- Per-head static amax scaling is defensible as "not smoothing" provided it is fixed per `(file, head, tensor)` and computed once from values (calibration, not adaptive/per-channel).
- Metrics must be stratified (per layer/timestep) with p95/worst-case and per-row cosine — the 60 workloads are structured, not IID, and a global mean can hide collapse.
- fp8 in pure PyTorch is a fake-quant accuracy proxy (quantize→dequantize, fp32 matmul accumulate), framed as such — not a kernel-accurate result.

### Resolved Disagreements
- Skip-predicate definition & circularity: Codex (round 1) showed that an "offline normalized-mass selector" both mislabels a full-score prepass as deployable and creates a denominator circularity. Resolution: per the user's directive to "do what the paper does," adopt the **exact BLASST online criterion** (`m̃ᵢ⁽ʲ⁾ − mᵢ⁽ʲ⁾ < ln(λ)` on the running row-max, pre-softmax). This is single-pass, well-defined, deployable, and removes the circularity natively. Codex round 2 confirmed R1–R7 resolved with no remaining blockers.
- τ semantics: Codex noted `λ` bounds each individual block's relative weight, not total dropped mass. Resolution: keep BLASST's per-block predicate but add a mandatory per-row total-dropped-mass diagnostic (mean/median/p95/max) measured against the true (no-skip) normalization.
- Static-P ordering: applied to per-block online probabilities `exp(score − m_running) × 256 → e4m3 → /256` (matching the Hopper `MAX_OFFSET=8` path); normalization by `l` is deferred to the end. This is faithful to real fp8 online softmax.
- Matmul precision: "fp8" means storage/input precision; QK^T and P·V accumulate in fp32 over dequantized values (fp8 inputs + fp32 accumulate). Stated explicitly in AC-4.
- AC-2 falsifiability: concrete thresholds pre-registered (full-seqlen L1-vs-L0 cosine ≥ 0.999 / relRMSE ≤ 2e-2; cropped fp32 vs `attention_ref` cosine ≥ 0.9999 / relRMSE ≤ 1e-3).
- Determinism: pin and record `CUDA_VISIBLE_DEVICES=3`, GPU model, torch/CUDA versions, SDPA backend, `torch.backends.cuda.matmul.allow_tf32`.
- Empty-row safeguard granularity (Codex round 2 optional): resolved at M-tile level — if a threshold would skip all blocks for an M-tile, force-keep that tile's single largest-`m̃` N-block and record the count.

### Convergence Status
- Final Status: `converged` (2 convergence rounds; round 2 returned no DISAGREE and no REQUIRED_CHANGES — only carried user decisions remained).

## Pending User Decisions

- DEC-1: Skip-normalization semantics (pre- vs post-softmax).
  - Claude Position: Pre-softmax (skipped blocks excluded from numerator and denominator).
  - Codex Position: Either is defensible; pre-softmax is faithful to "never compute skipped blocks."
  - Tradeoff Summary: Pre-softmax models a real sparse kernel; post-softmax gives cleaner dropped-mass attribution.
  - Decision Status: **RESOLVED — pre-softmax** (user: "do what the paper does"; BLASST is pre-softmax, denominator not updated for skipped blocks).

- DEC-2: Q/K/V fp8 scale granularity.
  - Claude Position: Per-head static amax.
  - Codex Position: Reasonable; results may differ materially from per-tensor.
  - Tradeoff Summary: Per-head matches the kernel's `(batch,nheads)` descale convention and is not smoothing; per-tensor is coarser/more conservative.
  - Decision Status: **RESOLVED — per-head static amax** (user selection).

- DEC-3: fp8 format for Q/K/V/P.
  - Claude Position: e4m3 for Q/K/V and for P (via ×256), with e5m2 as an optional ablation axis.
  - Codex Position: N/A - open question (P likely e4m3 by the Hopper precedent; make explicit).
  - Tradeoff Summary: e4m3 (more mantissa) suits attention forward and matches the Hopper fp8 path; e5m2 (more range) is a secondary comparison.
  - Decision Status: **RESOLVED (default) — e4m3 for Q/K/V/P**; e5m2 remains an optional ablation if the user later wants the comparison.

- DEC-4: Space-time reordering `(t,h,w)` factorization of seqlen 69120.
  - Claude Position: Use a configurable `(t,h,w)` defaulting to `20×48×72` (= 69120; ⇒ 768×1152 px, ~77 frames), with `t·h·w == 69120` asserted and an empirical locality-validation step that ranks candidate factorizations/flatten-orders by near-diagonal attention-mass concentration; run reordering as an additional arm alongside native order.
  - Codex Position: Reordering must stay out of the primary result until a true `(t,h,w)` is known; any wrong factorization contaminates results.
  - Tradeoff Summary: The dataset carries NO `(t,h,w)` metadata. Wan2.1 uses temporal-4× / spatial-16× token strides with `T_latent = 1 + (frames−1)/4`; `20×48×72` is stride-consistent but does NOT match the common 81-frame config (which yields `T=21`, and `21 ∤ 69120`). The exact grid is therefore unconfirmed.
  - Decision Status: **PENDING** — proceed with default `20×48×72` + assertion + empirical locality validation, but the user should confirm the exact Wan2.1 latent grid (resolution + frame count) that produced these 69120-token traces. If unconfirmed, the empirical locality ranking selects the working factorization and the reordering arm is labeled accordingly; the native-order experiment is unaffected.

- DEC-5: Headline claim (accuracy potential vs deployable predicate).
  - Claude Position: Reproduce BLASST's actual deployable online predicate (running-max threshold), so the headline is a real-method accuracy result, not an offline oracle.
  - Codex Position: The plan must commit to one; the offline-mass selector measured potential, not deployability.
  - Tradeoff Summary: BLASST's criterion is itself deployable (online running max), so adopting it makes the result deployable; an oracle (final-max / exact-mass) predicate can still be added as an upper-bound ablation.
  - Decision Status: **RESOLVED — deployable BLASST predicate** (follows the paper); an oracle upper-bound predicate is an optional secondary.

- DEC-6: Success definition (hard accuracy bar vs exploratory curves).
  - Claude Position: Exploratory curves; safe threshold read off the data.
  - Codex Position: N/A - open question.
  - Tradeoff Summary: A hard bar (e.g. cosine ≥ 0.99) would bake pass/fail into ACs; exploratory reporting leaves the threshold judgment to the user.
  - Decision Status: **RESOLVED — exploratory curves** (user selection); no hard accuracy bar in the acceptance criteria.

## Implementation Notes

### Code Style Requirements
- Implementation code and comments must NOT contain plan-specific terminology such as "AC-", "Milestone", "Step", "Phase", or similar workflow markers. These terms are for plan documentation only, not for the resulting codebase.
- Use descriptive, domain-appropriate naming in code instead (e.g. `skip_threshold`, `running_row_max`, `fp8_quantize_per_head`, `dropped_mass`, `ablation_level`).

### Environment & Determinism
- Pin `CUDA_VISIBLE_DEVICES=3`. Record GPU model, torch/CUDA versions, the selected SDPA backend, and `torch.backends.cuda.matmul.allow_tf32`.
- Write all experiment artifacts (per-workload CSV/JSON, summary, optional plots, run-config/version manifest) under `agent_space/` — treat them as disposable experiment output, not product code.
- This is a pure-PyTorch study: do NOT modify any CuTe/CUDA kernel or repository source under `flash_attn/cute/`, `hopper/`, or `csrc/`; reuse them read-only as references.

--- Original Design Draft Start ---

# Pure-Torch FP8 Block-Skip Attention Accuracy Simulator For Video Diffusion

## Original Idea

请使用GPU3进行开发。你读一下https://arxiv.org/pdf/2512.12087的核心算法，我很好奇在non-casual的diffusion模型中，如果QKPV都是8
  bit，是不是有些block可以直接跳过而不需要计算，几乎不引入精度损失。你可以考虑将QKV重新排列：[t * h * w, d] -> [?, t_s, h_s, w_s,
  d]，这样实际计算是由于QK的block size都是128，我们可以保证128都是真实视频中相近的token。更容易出现整个block跳过的情况。我需要你完整测量不同skip阈值下精度损失。

请在进行这个实验的时候关闭所有smooth等精度技巧。并保证计算attention时QK的block size都是128。本试验中P的量化采用静态给P乘以256再量化到fp8。不必使用动态P量化。

我们实验必须在~/dataset/v-dit/wan21的全部真实workload上测量精度。要求用torch的sdpa作为数值ground truth。测量量化kernel的RMSE MSE和cosine

## Primary Direction: Pure-Torch FakeQuant Simulator

### Rationale

Realizes the entire quantized + block-skip attention as a tiled PyTorch reference (fake-quant fp8 for Q/K/V, static P×256→fp8, explicit 128-block masking and skipping) with NO CuTe kernel, so accuracy across skip thresholds can be measured quickly and exactly — distinct from any in-kernel implementation.

### Approach Summary

Build a self-contained PyTorch reference for FP8 block-skipping attention that requires no CuTe kernel, so accuracy across skip thresholds can be measured exactly and fast:

1. **Tile Q, K, V into explicit 128×128 blocks** (matching the fixed block-size constraint), following the sparsity layout used in `flash_attn/cute/block_sparse_utils.py` (which already defines `BlockSparseTensorsTorch` with mask/full block indices).
2. **FP8 fake-quantization pipeline.** Scale Q, K to e4m3fn using computed max values (per-tensor or per-head), then quantize→dequantize with no dynamic smoothing/clipping (per spec). Static P quantization: compute P∈[0,1] → P×256 → fp8_e5m2 → descale by 1/256. Descale outputs in float32.
3. **Block-skipping logic.** For each (Q-block, K-block) pair, compute a cheap score statistic (block max/norm of scaled-Q·scaled-K); skip if below threshold τ; sweep τ (e.g. {0.001, 0.01, 0.1, 1.0}). Apply explicit masking to zero out skipped block contributions.
4. **Online softmax.** Mirror `Softmax.online_softmax()` from `flash_attn/cute/softmax.py` (row_max, row_sum, exp-normalize) in pure-PyTorch loops so that block skipping integrates naturally.
5. **Ground truth.** Use `torch.nn.functional.scaled_dot_product_attention` (already used as the reference in `tests/cute/test_mask_mod.py`) as the numerical oracle.
6. **Metrics.** Compute RMSE, MSE, and cosine similarity per-output / per-batch / per-head, aggregated across the full V-DiT/wan21 workload (`~/dataset/v-dit/wan21_p1/` and `wan21_p2/`).

Affected components: a new pure-PyTorch reference module (no kernel changes); reuses `attention_ref`, `bench_utils`, and the block-sparse tensor structures. Estimated 400–700 LOC (quant helpers ~100–150, online softmax loop ~150–200, block-skip logic ~100–150, metric aggregation ~100–150, wan21 loader ~50–100).

### Objective Evidence

- `flash_attn/cute/testing.py:326` — `attention_ref(...)` defines a full softmax/scaling/masking PyTorch oracle (~488 lines), handling upcast and descaling (lines 361–367), query/key padding masks, causal, and softcap. Precedent for a PyTorch oracle already exists.
- `tests/cute/test_mask_mod.py:113` and `tests/cute/test_mask_mod_varlen.py` — call `F.scaled_dot_product_attention(...)` as the reference output (e.g. `out_ref = F.scaled_dot_product_attention(q, k, v, scale=scale)`). SDPA is the established ground truth.
- `flash_attn/cute/block_sparse_utils.py:39–50` — `BlockSparseTensorsTorch` NamedTuple (`mask_block_cnt`, `mask_block_idx`, `full_block_cnt`, `full_block_idx`, ...). Block-indexing metadata already exists and is tested in `tests/cute/test_block_sparsity.py`.
- `flash_attn/cute/benchmark_flash_attention_fp8.py` — float8 dtype selection (`torch.float8_e4m3fn`, `torch.float8_e5m2`), descaling in the reference baseline, CuDNN fp8 scale/descale tensor patterns for Q/K/V/S/O, and P scaling applied at discrete points (precedent for a static-P scheme).
- `flash_attn/cute/softmax.py:127–192` — `online_softmax()` maintains per-block `row_max`/`row_sum`; classes `Softmax` and `SoftmaxSm100` show the tile-based update pattern reusable in a pure-PyTorch loop.
- `tests/cute/test_mask_mod.py:133–150` — `assert_fwd_matches_reference()` with tolerance computation (`fwd_atol`, `rtol`), element-wise max-error checks, and nan/inf validation. Test-harness pattern is established.
- `~/dataset/v-dit/wan21_p1/` and `wan21_p2/` — real workloads confirmed locally.
- `flash_attn/cute/testing.py:376–388` — einsum Q·K → softmax → V baseline (compact, ~160 LOC across lines 326–465).
- `flash_attn/cute/bench_utils.py:84–100` — `attention_ref(q, k, v, causal)` plus FLOPS/bandwidth calculators, reusable for FP8 measurements.

### Known Risks

- **Block-skipping threshold tuning.** No existing heuristic in the codebase; empirically sweeping [0.001 → 1.0] is required. At higher thresholds, block-max may not correlate well with per-token importance.
- **Online-softmax numerical stability.** Pure-PyTorch row-wise max/sum can diverge slightly from single-pass softmax due to order of operations; validate against full-tensor `torch.softmax()` at tolerance ~1e-5.
- **FP8 rounding semantics.** `torch.float8_e4m3fn` rounding is deterministic but may differ subtly from CuDNN/hardware FP8; either treat PyTorch FP8 as canonical for this study or validate against kernel output.
- **V-DiT shape variety.** Variable seqlen/batch sizes mean 128-tiling may not align (boundary/remainder blocks); pad to block boundary or handle remainders explicitly.
- **Scale-factor selection.** Static P×256 is specified, but Q/K/V scale derivation (per-tensor, per-head, per-block) is undefined; "no smooth" risks extreme scales — recommend independent per-head max-based scales for Q, K, V.

## Alternative Directions Considered

### Alt-1: Cheap Block-Skip Predicate
- Gist: Add a dynamic block-skip predicate inside the forward pass — after QK^T and masking but before softmax, evaluate whether a (Q-block, K-block) pair's pre-softmax max score is below a tunable threshold and, if so, skip the softmax step and V accumulation for that block (treating it as zero contribution). Extend `BlockSparseTensors` to optionally hold a pre-computed skip list (or compute the decision on-the-fly in the consumer loop). In the FP8 context this also reduces error propagation from low-contribution blocks.
- Objective Evidence:
  - `flash_attn/cute/block_info.py:23–55` — `get_n_block_min_max()` block-range decision point.
  - `flash_attn/cute/compute_block_sparsity.py:306–319` — existing three-way full/partial/skip classification; 5-point representative sampling per block (lines 189–216).
  - `flash_attn/cute/block_sparsity.py:17–50` — mask/full block-metadata tuples (`mask_block_cnt`, `mask_block_idx`, `full_block_cnt`, `full_block_idx`).
  - `flash_attn/cute/block_sparse_utils.py` — `consume_block_sparse_loads` iteration over partial/full lists; empty-tile correction precedent (`handle_block_sparse_empty_tile_correction_sm100`).
  - `flash_attn/cute/softmax.py:92–190` — per-row `row_max` available as a pre-softmax bound.
  - `flash_attn/cute/mask.py:176–424` — mask application before softmax; plug-in site for a post-mask, pre-softmax block evaluation.
  - `flash_attn/cute/flash_fwd_sm90.py:766–1175` — block iteration loop with no current score-based skip.
- Why not primary: It modifies the live CuTe kernel (medium confidence, with threshold-vs-quant-scale coupling and an unresolved online-vs-offline tradeoff), whereas the user's immediate goal is an accuracy measurement that the kernel-free simulator answers faster and more controllably.

### Alt-2: Space-Time Locality Reordering
- Gist: Insert a host-side token permutation that reorders Q/K/V from `[batch, seqlen, heads, dim]` to a space-time-coherent tiling `[batch, heads, t_s, h_s, w_s, dim]` (with `t_s·h_s·w_s = 128`) before attention, and applies the inverse permutation to the output — so each 128-token block holds spatially/temporally adjacent video tokens and whole blocks become more skippable. Reuses the `pack_gqa` stride-permutation pattern and feeds permuted tensors into the existing block-sparse path with zero kernel modification.
- Objective Evidence:
  - `flash_attn/cute/pack_gqa.py:15–112` — `pack_gqa_layout`/`unpack_gqa_layout` permute tensor shape via stride manipulation at the host level; direct analogue for a space-time permutation.
  - `flash_attn/layers/patch_embed.py:46–67` — `einops.rearrange("b c (h p1) (w p2) -> b h w (c p1 p2)")` shows codebase familiarity factorizing spatial dims.
  - `flash_attn/cute/block_sparsity.py:17–50, 181–195, 286` — `BlockSparseTensorsTorch` block counts and `get_block_sparse_expected_shapes()` with `m_block_size_effective = q_stage * m_block_size`.
  - `flash_attn/cute/interface.py:92–234` — `_tile_size_fwd_sm90()` with `tile_m=128, tile_n=128` defaults locked in.
  - `flash_attn/cute/seqlen_info.py:17–65` — per-batch/head seqlen/offset metadata for post-permutation recomputation.
  - `~/dataset/v-dit/wan21_p1/layer_0/timestep_{0,3,6,9,29,49}.pt` — real video attention traces for validating the factorization.
  - `flash_attn/cute/benchmark_flash_attention_fp8.py` — PyTorch baseline + FP8 measurement scaffolding.
- Why not primary: It is an enabler that raises the skip rate but does not by itself produce the accuracy measurement, and it requires the true Wan2.1 (t, h, w) factorization — so it is best folded into the primary simulator as a configurable layout rather than pursued standalone.

### Alt-3: FP8 Numerics Pipeline (static-P, no-smoothing)
- Gist: Pin down the exact 8-bit numerics — quantize Q/K/V to e4m3fn with per-batch-head descale factors, expand the P range via the existing `MAX_OFFSET=8` softmax trick (which maps P to ~[0, 256]), statically scale P by 256 before the fp8 cast (no dynamic P scaling), and disable every accuracy trick (score_mod, softcap, fp8 exp2-emulation tuning). No Hadamard/per-channel smoothing exists in the fp8 path today, which keeps the isolated baseline clean.
- Objective Evidence:
  - `hopper/flash_fwd_kernel_sm90.h` + `hopper/softmax.h:64–149` — `scale_apply_exp2<Max_offset>` with `Max_offset=8` expands the exp/P range to ~256, and `finalize()` applies `sum_scale = 1/2^Max_offset` — a direct hardware analogue of the prescribed P×256.
  - `hopper/instantiations/flash_fwd_hdim128_e4m3_sm90.cu` — existing FP8 e4m3 instantiations; `hopper/flash_fwd_launch_template.h:35` `Is_FP8` detection.
  - `flash_attn/cute/flash_fwd_sm100.py:111–117` — `DescaleTensors` (`q_descale`, `k_descale`, `v_descale`); `flash_attn/cute/interface.py:799` constructs them for SM100.
  - `flash_attn/cute/softmax.py:19–51` — `apply_score_mod_inner` (set `score_mod=None`); `Has_softcap=False`; `_FP8_TUNING_CONFIG.ex2_emu_freq` left at default.
  - `flash_attn/cute/mma_sm100_desc.py:48–50` — `MXF8F6F4Format.E4M3`/`E5M2`; `flash_attn/cute/blackwell_helpers.py:13–30` `f8f6f4` MMA kind.
- Why not primary: The static "P×256→fp8" path is not surfaced in the Python API and would require kernel-epilogue changes (medium confidence), whereas the simulator can express the identical numerics in PyTorch fake-quant immediately — yet this direction's findings define exactly what the simulator must implement.

### Alt-4: Workload Capture & Threshold Sweep Harness
- Gist: Build the experiment driver that loops over all 60 real Wan2.1 attention traces (`~/dataset/v-dit/wan21_p{1,2}/layer_{0,10,20,30,39}/timestep_{0,3,6,9,29,49}.pt` — dicts of query/key/value, each `[1, 69120, 40, 128]` bf16; seqlen=69120, 40 heads, head_dim=128), pins to GPU3, sweeps a list of skip thresholds τ, runs the quantized simulation against torch SDPA ground truth, and emits a results table (`layer`, `timestep`, `threshold`, `rmse`, `mse`, `cosine`) into `agent_space/`.
- Objective Evidence:
  - `~/dataset/v-dit/wan21_p{1,2}/layer_*/timestep_*.pt` — 60 files; each a dict `{query, key, value}` of shape `[1, 69120, 40, 128]` bf16.
  - `benchmarks/benchmark_attn.py`, `benchmarks/bench_sm90.py` — parametrization over seqlen/headdim/dtype/causal/nheads; `setup_fa4()`/`setup_standard()` returning `(fwd_fn, bwd_fn)`; `triton.testing.do_bench()` timing; CSV export.
  - `flash_attn/cute/testing.py:325–465` — `attention_ref` with tolerance checks; `tests/cute/test_flash_attn.py` max/mean abs-error assertions.
  - `flash_attn/cute/benchmark_flash_attention_fp8.py` — existing FP8 benchmark; `DescaleTensors` descaling helpers.
  - `CLAUDE.md` — `agent_space/` scratch convention; `CUDA_VISIBLE_DEVICES` GPU selection; `FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED` caching.
- Why not primary: It is the orchestration/deliverable layer that wraps the primary simulator rather than the technical core that determines accuracy, so it is built jointly with — and on top of — the primary.

### Alt-5: Error-Decomposition Methodology
- Gist: Frame the measurement as an ablation that attributes total error to its sources — run each input as {fp32 baseline, fp8-QKV only, fp8-QKV + static-P, fp8-QKV + static-P + skip@τ} and report RMSE/MSE/cosine for each stage vs fp32 — then correlate the per-block discarded softmax probability mass (computable from `row_sum`/LSE) against the resulting output error to justify the largest safe skip threshold.
- Objective Evidence:
  - `flash_attn/cute/softmax.py:127–227` — `online_softmax()` per-row `row_max`/`row_sum`; `finalize()` computes `LSE = log(row_sum) + max_scaled` (the softmax normalizer / total attention mass).
  - `flash_attn/cute/interface.py` — `return_lse` (LSE shape `(batch, num_heads, seqlen_q)`); `FlashAttentionForwardCombine` merges partial LSE across SplitKV, so per-block LSE is tracked.
  - `flash_attn/cute/flash_fwd_sm100.py` — `DescaleTensors` + `_load_effective_descales()` allow toggling quantization states for the ablation.
  - `flash_attn/cute/testing.py:326–465` — `attention_ref` returns LSE (`torch.logsumexp`); `flash_attn/cute/block_sparsity.py` exposes blocks-processed vs blocks-available.
- Why not primary: It is an analysis lens layered on the same measurement runs rather than the implementation itself, so it shapes how the primary's results are reported and thresholded rather than standing alone.

## Synthesis Notes

The primary simulator is deliberately the convergence point for the other five directions, which are best treated as components rather than competitors. The FP8 Numerics Pipeline (Alt-3) supplies the exact fake-quant recipe the simulator must reproduce — e4m3 Q/K/V with per-head max scales, the `MAX_OFFSET=8` → P×256 → e5m2 mapping, and the precise list of tricks to disable (score_mod, softcap, fp8 exp2-emulation) — so the simulator's numerics stay verifiably faithful to the real kernel. The Cheap Block-Skip Predicate (Alt-1) defines the skip statistic and threshold semantics the simulator evaluates per 128×128 block, and is the natural follow-on if results justify porting the skip into the live CuTe kernel for actual speedup. The Space-Time Locality Reordering (Alt-2) plugs in as an optional configurable layout in front of the simulator (using the real Wan2.1 t/h/w factorization once known) to test the hypothesis that spatially-coherent blocks raise the skip rate at a fixed accuracy budget. The Workload Sweep Harness (Alt-4) is the driver that runs the simulator over all 60 wan21 traces, pins GPU3, sweeps τ, and writes the RMSE/MSE/cosine table into `agent_space/`. The Error-Decomposition Methodology (Alt-5) shapes how those numbers are reported — as a {fp32 → fp8 → +static-P → +skip} ablation plus a discarded-mass-vs-error correlation that pinpoints the safe threshold. In short: build Alt-3's numerics and Alt-1's predicate inside the primary simulator, optionally enable Alt-2's layout, drive it with Alt-4, and report it through Alt-5.

--- Original Design Draft End ---
