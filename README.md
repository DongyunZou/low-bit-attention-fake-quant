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
  experimental estimate. In dynamic-P mode the Triton path can start from this
  estimate and only fall back to an online-style rowmax update when the current
  tile moves outside a threshold.

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

## Estimated Rowmax Update Trick

The dynamic-P path can use Q smoothing to pre-estimate each row's softmax
coordinate:

```text
rowmax_est ~= max_K(q_m @ K_smooth.T)
```

This avoids running a full online rowmax update for most K tiles. The kernel
still keeps online-softmax state and corrects the estimate when needed:

- upward correction: update if `tile_max - m_i > up_threshold`
- downward correction: update if `m_i - max_seen_i > down_threshold`
- current debug candidate: `up_threshold=16`, `down_threshold=32`

On the saved Wan bad self-attention call
`bench/wan_diag_badcall/call_066.pt` with `q_smooth_block=256`,
`q_kmeans_k=32`, `fp8_block_size=64`, and `block_m=block_n=64`, the Triton
counter gives:

| Method | Rowmax updates | Rescales | Rows with update |
|---|---:|---:|---:|
| FA4-style online, threshold 8 | 2,717,371 | 1,406,651 | 1,310,720 |
| Estimated rowmax, up 16 / down 32 | 677,294 | 595,885 | 153,999 |

So the estimated-rowmax path reduces rowmax updates by about 75% and rescale
events by about 58% on this call. The point is to skip many online-softmax
coordinate changes, which also avoids the associated accumulator rescale and
extra exponent work in the hot K loop. The output is still checked against the
online-rowmax dynamic-P path; saved sensitive calls and Wan layer-0 workloads
show no NaN/Inf with this setting.

See `docs/quant_precision_test_plan.md` for the longer implementation and
evaluation plan.

See `docs/fa4_implementation_roadmap.md` for the FA4 CUTE kernel roadmap from
per-block FP32-scale FP8 quant through K/Q/V smoothing, k-means reorder, and the
estimated-rowmax trick.

## Setup

```bash
uv sync --extra dev --extra bench
uv run pytest
```
