# Multi-Bin UTA Skip Approximation Probe on wan21_p1

- Workloads: 30 unique=30 (`wan21_p1`, all layer/timestep files from prior group shards).
- Mean no-skip FP8/static-P vs SDPA relRMSE: `4.0943e-02`.
- Max peak memory per shard: `38.1 GB`; summed GPU seconds across shards: `4465.2`.
- Fill modes tested: `mean_a1.5`, `uta8_a1.5`, `uta16`, `uta16_a1.5`, `uta16_a2`; lambdas `0.03/0.1/0.2`; scopes `tile`, `row`, `group32`, `group64`.

## Findings

- Multi-bin UTA is clearly better than the previous single-mean fill at the same skip mask. `uta16` is usually best or tied best; `uta8` is already better than `mean_a1.5` but leaves accuracy on the table.
- The best alpha is workload/scope dependent: `a1.5` is safest in the averaged results here; `a2` wins some individual shards/low-lambda cases. Unscaled `uta16` is also competitive at higher lambda, so alpha should be swept before hard-coding.
- Practical point: this does not require kmeans or temporal cache. It only splits a 128-key block into fixed contiguous bins and computes per-bin mean/max plus `sum(V)`; a kernel can precompute per-bin `sum(V)` for each K/V block.
- For tensor-core-friendly tile skip, `tile:uta16_a1.5:lambda=0.2` reaches the same `36.82%` full-tile skip as previous `tile:mean_a1.5:lambda=0.2`, and skip-only relRMSE drops from `4.8171e-02` to `3.9635e-02`.

## Same-Mask Improvement

| same skip mask | mean_a1.5 relRMSE | UTA relRMSE | relative drop |
|---|---:|---:|---:|
| `tile:lambda=0.1` | 2.9577e-02 | `uta16_a1.5` 2.3803e-02 | 19.5% |
| `tile:lambda=0.2` | 4.8171e-02 | `uta16_a1.5` 3.9635e-02 | 17.7% |
| `group64:lambda=0.1` | 3.5686e-02 | `uta16_a1.5` 2.8675e-02 | 19.6% |
| `group32:lambda=0.1` | 4.3975e-02 | `uta16_a1.5` 3.5403e-02 | 19.5% |
| `group32:lambda=0.2` | 7.0085e-02 | `uta16_a1.5` 5.7628e-02 | 17.8% |

## Selected Configs

| config | row skip | full tile skip | partial tile | skip-only relRMSE | SDPA relRMSE |
|---|---:|---:|---:|---:|---:|
| `tile:zero:lam0.03` | 0.1794 | 0.1794 | 0.0000 | 1.9599e-02 | 4.6187e-02 |
| `tile:mean_a1.5:lam0.03` | 0.1794 | 0.1794 | 0.0000 | 1.4605e-02 | 4.3891e-02 |
| `tile:uta8_a1.5:lam0.03` | 0.1794 | 0.1794 | 0.0000 | 1.3047e-02 | 4.3331e-02 |
| `tile:uta16:lam0.03` | 0.1794 | 0.1794 | 0.0000 | 1.2867e-02 | 4.3328e-02 |
| `tile:uta16_a1.5:lam0.03` | 0.1794 | 0.1794 | 0.0000 | 1.1925e-02 | 4.2964e-02 |
| `tile:uta16_a2:lam0.03` | 0.1794 | 0.1794 | 0.0000 | 1.1909e-02 | 4.2975e-02 |
| `group64:zero:lam0.03` | 0.2048 | 0.1794 | 0.0507 | 2.3840e-02 | 4.8404e-02 |
| `group64:mean_a1.5:lam0.03` | 0.2048 | 0.1794 | 0.0507 | 1.7634e-02 | 4.5184e-02 |
| `group64:uta8_a1.5:lam0.03` | 0.2048 | 0.1794 | 0.0507 | 1.5693e-02 | 4.4363e-02 |
| `group64:uta16:lam0.03` | 0.2048 | 0.1794 | 0.0507 | 1.5533e-02 | 4.4387e-02 |
| `group64:uta16_a1.5:lam0.03` | 0.2048 | 0.1794 | 0.0507 | 1.4330e-02 | 4.3837e-02 |
| `group64:uta16_a2:lam0.03` | 0.2048 | 0.1794 | 0.0507 | 1.4299e-02 | 4.3830e-02 |
| `group32:zero:lam0.03` | 0.2431 | 0.1794 | 0.1334 | 2.9948e-02 | 5.2186e-02 |
| `group32:mean_a1.5:lam0.03` | 0.2431 | 0.1794 | 0.1334 | 2.1839e-02 | 4.7355e-02 |
| `group32:uta8_a1.5:lam0.03` | 0.2431 | 0.1794 | 0.1334 | 1.9317e-02 | 4.6092e-02 |
| `group32:uta16:lam0.03` | 0.2431 | 0.1794 | 0.1334 | 1.9208e-02 | 4.6170e-02 |
| `group32:uta16_a1.5:lam0.03` | 0.2431 | 0.1794 | 0.1334 | 1.7607e-02 | 4.5303e-02 |
| `group32:uta16_a2:lam0.03` | 0.2431 | 0.1794 | 0.1334 | 1.7565e-02 | 4.5273e-02 |
| `tile:zero:lam0.1` | 0.2777 | 0.2777 | 0.0000 | 4.4173e-02 | 6.2076e-02 |
| `tile:mean_a1.5:lam0.1` | 0.2777 | 0.2777 | 0.0000 | 2.9577e-02 | 5.1314e-02 |
| `tile:uta8_a1.5:lam0.1` | 0.2777 | 0.2777 | 0.0000 | 2.6072e-02 | 4.9258e-02 |
| `tile:uta16:lam0.1` | 0.2777 | 0.2777 | 0.0000 | 2.4974e-02 | 4.8778e-02 |
| `tile:uta16_a1.5:lam0.1` | 0.2777 | 0.2777 | 0.0000 | 2.3803e-02 | 4.8019e-02 |
| `tile:uta16_a2:lam0.1` | 0.2777 | 0.2777 | 0.0000 | 2.4532e-02 | 4.8480e-02 |
| `group64:zero:lam0.1` | 0.3144 | 0.2777 | 0.0734 | 5.2941e-02 | 6.8924e-02 |
| `group64:mean_a1.5:lam0.1` | 0.3144 | 0.2777 | 0.0734 | 3.5686e-02 | 5.5403e-02 |
| `group64:uta8_a1.5:lam0.1` | 0.3144 | 0.2777 | 0.0734 | 3.1429e-02 | 5.2634e-02 |
| `group64:uta16:lam0.1` | 0.3144 | 0.2777 | 0.0734 | 3.0231e-02 | 5.2090e-02 |
| `group64:uta16_a1.5:lam0.1` | 0.3144 | 0.2777 | 0.0734 | 2.8675e-02 | 5.0955e-02 |
| `group64:uta16_a2:lam0.1` | 0.3144 | 0.2777 | 0.0734 | 2.9428e-02 | 5.1450e-02 |
| `group32:zero:lam0.1` | 0.3661 | 0.2777 | 0.1788 | 6.5593e-02 | 7.9630e-02 |
| `group32:mean_a1.5:lam0.1` | 0.3661 | 0.2777 | 0.1788 | 4.3975e-02 | 6.1588e-02 |
| `group32:uta8_a1.5:lam0.1` | 0.3661 | 0.2777 | 0.1788 | 3.8748e-02 | 5.7808e-02 |
| `group32:uta16:lam0.1` | 0.3661 | 0.2777 | 0.1788 | 3.7434e-02 | 5.7163e-02 |
| `group32:uta16_a1.5:lam0.1` | 0.3661 | 0.2777 | 0.1788 | 3.5403e-02 | 5.5515e-02 |
| `group32:uta16_a2:lam0.1` | 0.3661 | 0.2777 | 0.1788 | 3.6225e-02 | 5.6077e-02 |
| `tile:zero:lam0.2` | 0.3682 | 0.3682 | 0.0000 | 7.5657e-02 | 8.8033e-02 |
| `tile:mean_a1.5:lam0.2` | 0.3682 | 0.3682 | 0.0000 | 4.8171e-02 | 6.3914e-02 |
| `tile:uta8_a1.5:lam0.2` | 0.3682 | 0.3682 | 0.0000 | 4.3001e-02 | 6.0015e-02 |
| `tile:uta16:lam0.2` | 0.3682 | 0.3682 | 0.0000 | 4.0366e-02 | 5.8263e-02 |
| `tile:uta16_a1.5:lam0.2` | 0.3682 | 0.3682 | 0.0000 | 3.9635e-02 | 5.7587e-02 |
| `tile:uta16_a2:lam0.2` | 0.3682 | 0.3682 | 0.0000 | 4.1386e-02 | 5.8914e-02 |
| `group64:zero:lam0.2` | 0.4124 | 0.3682 | 0.0886 | 8.8799e-02 | 9.9677e-02 |
| `group64:mean_a1.5:lam0.2` | 0.4124 | 0.3682 | 0.0886 | 5.7562e-02 | 7.1536e-02 |
| `group64:uta8_a1.5:lam0.2` | 0.4124 | 0.3682 | 0.0886 | 5.1379e-02 | 6.6573e-02 |
| `group64:uta16:lam0.2` | 0.4124 | 0.3682 | 0.0886 | 4.8453e-02 | 6.4533e-02 |
| `group64:uta16_a1.5:lam0.2` | 0.4124 | 0.3682 | 0.0886 | 4.7323e-02 | 6.3442e-02 |
| `group64:uta16_a2:lam0.2` | 0.4124 | 0.3682 | 0.0886 | 4.9169e-02 | 6.4896e-02 |
| `group32:zero:lam0.2` | 0.4714 | 0.3682 | 0.2039 | 1.0682e-01 | 1.1628e-01 |
| `group32:mean_a1.5:lam0.2` | 0.4714 | 0.3682 | 0.2039 | 7.0085e-02 | 8.2301e-02 |
| `group32:uta8_a1.5:lam0.2` | 0.4714 | 0.3682 | 0.2039 | 6.2555e-02 | 7.5897e-02 |
| `group32:uta16:lam0.2` | 0.4714 | 0.3682 | 0.2039 | 5.9188e-02 | 7.3405e-02 |
| `group32:uta16_a1.5:lam0.2` | 0.4714 | 0.3682 | 0.2039 | 5.7628e-02 | 7.1844e-02 |
| `group32:uta16_a2:lam0.2` | 0.4714 | 0.3682 | 0.2039 | 5.9689e-02 | 7.3539e-02 |

## Best By Row-Skip Budget

| skip-only relRMSE budget | best config | row skip | full tile skip | partial tile | skip-only relRMSE |
|---:|---|---:|---:|---:|---:|
| 0.020 | `group32:uta16_a2:lam0.03` | 0.2431 | 0.1794 | 0.1334 | 1.7565e-02 |
| 0.030 | `group64:uta16_a1.5:lam0.1` | 0.3144 | 0.2777 | 0.0734 | 2.8675e-02 |
| 0.050 | `group64:uta16_a1.5:lam0.2` | 0.4124 | 0.3682 | 0.0886 | 4.7323e-02 |
| 0.080 | `row:uta16_a2:lam0.03` | 0.4928 | 0.1794 | 0.5809 | 5.0723e-02 |
| 0.100 | `row:uta16_a1.5:lam0.1` | 0.6399 | 0.2777 | 0.5633 | 9.8185e-02 |

## Best With Partial-Tile Constraint

| budget | max partial tile | best config | row skip | full tile skip | partial tile | skip-only relRMSE |
|---:|---:|---|---:|---:|---:|---:|
| 0.030 | 0.05 | `tile:uta16_a1.5:lam0.1` | 0.2777 | 0.2777 | 0.0000 | 2.3803e-02 |
| 0.030 | 0.10 | `group64:uta16_a1.5:lam0.1` | 0.3144 | 0.2777 | 0.0734 | 2.8675e-02 |
| 0.030 | 0.20 | `group64:uta16_a1.5:lam0.1` | 0.3144 | 0.2777 | 0.0734 | 2.8675e-02 |
| 0.050 | 0.05 | `tile:uta16_a1.5:lam0.2` | 0.3682 | 0.3682 | 0.0000 | 3.9635e-02 |
| 0.050 | 0.10 | `group64:uta16_a1.5:lam0.2` | 0.4124 | 0.3682 | 0.0886 | 4.7323e-02 |
| 0.050 | 0.20 | `group64:uta16_a1.5:lam0.2` | 0.4124 | 0.3682 | 0.0886 | 4.7323e-02 |
| 0.080 | 0.05 | `tile:uta16_a1.5:lam0.2` | 0.3682 | 0.3682 | 0.0000 | 3.9635e-02 |
| 0.080 | 0.10 | `group64:uta16_a1.5:lam0.2` | 0.4124 | 0.3682 | 0.0886 | 4.7323e-02 |
| 0.080 | 0.20 | `group64:uta16_a1.5:lam0.2` | 0.4124 | 0.3682 | 0.0886 | 4.7323e-02 |
