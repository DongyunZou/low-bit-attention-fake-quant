# low-bit-fake-quant

Triton-first fake-quant tools for low-bit attention experiments, mainly for
Wan/video diffusion attention precision studies.

## Supported Settings

The main entry point is `low_bit_fake_quant.fake_quant_attention(q, k, v, cfg)`,
where `cfg` is a `QuantConfig`.

- Q/K quant: `fp8_block`, `mxfp8`
- V quant: `fp8_channel`, `fp8_block`, `mxfp8`
- Q/K smoothing: `off`, `k_only`, `full`
- Q reorder: `q_kmeans_k=None` or a cluster count such as `32`/`64`
- V reorder: `v_kmeans_k=None` or a cluster count; K is co-permuted with V
- V smoothing: `off`, `per_block`
- P requant: `p_requant=False` for Q/K/V-only fake quant, or `True` for Triton
  streaming P-to-FP8 attention
- P quant: `elementwise` (`P * 256 -> FP8`), `mx` (pow2 dynamic scale),
  `dynamic` (FP32 dynamic scale), or `auto`
- Row max: `online` exact row max, or `qm_k` using `max(qm @ K_smooth.T)` as an
  experimental estimate

`auto` picks `mx` for `v_quant="mxfp8"` and `elementwise` otherwise.

## Experiments

- `bench/eval_wan21.py`: full Wan attention workload matrix.
- `bench/sweep_v_quant.py`: focused V quant and V smoothing sweep.
- `bench/sweep_kmeans_k.py`: Q/V k-means cluster count sweep.
- `bench/gen_wan_videos.py`: end-to-end Wan video generation comparison.

See `docs/quant_precision_test_plan.md` for the longer implementation and
evaluation plan.

## Setup

```bash
uv sync --extra dev --extra bench
uv run pytest
```
