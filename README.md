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
- `bench/gen_wan_e2e_pquant.py`: SDPA plus five P-quant end-to-end videos.
- `bench/eval_video_dirs.py`: PSNR/SSIM/LPIPS for generated videos.

## End-to-End Wan Videos

Prepare Wan2.1 and the T2V-14B checkpoint:

```bash
git clone https://github.com/Wan-Video/Wan2.1.git /path/to/Wan2.1
uv sync --extra dev --extra bench

# Hugging Face
uv run hf download Wan-AI/Wan2.1-T2V-14B \
  --local-dir /path/to/Wan2.1-T2V-14B
```

The same checkpoint is also available on ModelScope as
`Wan-AI/Wan2.1-T2V-14B` if Hugging Face is not reachable.

Generate one SDPA reference and five fake-quant videos with the same prompt and
seed:

```bash
uv run python bench/gen_wan_e2e_pquant.py \
  --wan-root /path/to/Wan2.1 \
  --ckpt-dir /path/to/Wan2.1-T2V-14B \
  --out-dir bench/wan_e2e_pquant \
  --prompt "A skateboarding scene in a dynamic street style..." \
  --seed 42 \
  --size 832*480 \
  --frame-num 81 \
  --sample-steps 50 \
  --t5-cpu \
  --offload-model
```

Compare every fake-quant video in the output directory to `sdpa.mp4`:

```bash
uv run python bench/eval_video_dirs.py \
  --pred-dir bench/wan_e2e_pquant \
  --ref-video bench/wan_e2e_pquant/sdpa.mp4 \
  --output-json bench/wan_e2e_pquant/metrics.json
```

For 14B generation, use a mostly free 80GB GPU. If another process is already
using tens of GB of VRAM, model load can OOM before generation starts.

## Wan Layer-0 Accuracy Snapshot

Mean over six real Wan2.1 layer-0 attention workloads in
`/home/dongyun/dataset/v-dit/wan21_p1/layer_0`. Reference is BF16 SDPA.
All rows use Q/K `fp8_block` with FP32 block scales and V `fp8_channel`.

| Stack | P quant / row max | MSE | RMSE | Cosine |
|---|---|---:|---:|---:|
| K smooth only | static `P*256`, online row max | 1.717711e-03 | 4.046466e-02 | 0.996441 |
| K smooth only | dynamic P, estimated row max | N/A | N/A | N/A |
| Q k-means + Q smooth + K smooth | static `P*256`, online row max | 7.194572e-04 | 2.636703e-02 | 0.998528 |
| Q k-means + Q smooth + K smooth | dynamic P, estimated row max | 7.128224e-04 | 2.624454e-02 | 0.998541 |
| Q k-means + Q/V smooth + V k-means + K smooth | static `P*256`, online row max | 6.735246e-04 | 2.523165e-02 | 0.998598 |
| Q k-means + Q/V smooth + V k-means + K smooth | dynamic P, estimated row max | 6.586863e-04 | 2.495268e-02 | 0.998630 |

`dynamic P, estimated row max` uses `rowmax_mode="qm_k"`, so it requires Q
smoothing and is not defined for K-smooth-only runs.

See `docs/quant_precision_test_plan.md` for the longer implementation and
evaluation plan.

## Setup

```bash
uv sync --extra dev --extra bench
uv run pytest
```
