# FP8 + BLASST Block-Skip Attention — Preliminary Analysis (8-file subset)

> Preliminary: computed on an 8-file stratified subset of the 60 Wan2.1
> workloads (subset-dev workflow). To be regenerated on the full dataset at the
> final run. Source data: `results.json` / `records.csv` in this directory.
> Authored via Codex (`analyze` task routing).

## Final Analysis

This 8-workload Wan2.1 subset should be read as an exploratory
degradation-vs-lambda sweep rather than a final quality bound. The ablation
ladder shows that the dominant accuracy loss is already present before block
skipping: bf16 no-quant matches SDPA closely, while FP8 Q/K/V drops the global
cosine to `0.99835` with `4.70e-2` relative RMSE. Adding static `P*256` FP8
quantization is essentially neutral (`0.99833`, `4.72e-2`), so the lambda sweep
is best interpreted on top of an FP8 floor near `0.99833`.

Under that floor, the safe-lambda read-off is straightforward. A `cos >= 0.999`
target is unattainable in this setup because FP8 alone is below it. For
`cos >= 0.998`, lambda can go to `0.03`, giving about `16%` tile skip. For
`cos >= 0.995`, lambda can go to `0.2`, giving about `31%` skip. For
`cos >= 0.99`, lambda can go to `0.3`, giving about `39%` skip. Beyond that,
degradation accelerates: `lambda=0.5` falls to `0.98570`, and `lambda=0.7` to
`0.97484`.

The dropped-mass statistics are strongly predictive of skip-induced error. The
correlation between `dropped_mass_p95` and skip-only relative RMSE is high in
linear space (`r=0.957`) and even tighter in log-log space (`r=0.979`),
supporting dropped mass as a useful scalar diagnostic for choosing lambda.

The key caveat is coverage. These results are from only 8 stratified files; the
full 60-file run and the pending space-time reorder arm, with true Wan2.1
`t,h,w` grid still unconfirmed, are needed before treating these thresholds as
production recommendations.
