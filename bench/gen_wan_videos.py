"""Generate Wan2.1-T2V-14B videos with different attention strategies and
report MSE/RMSE/PSNR/SSIM against the SDPA reference.

Strategies:
    sdpa            : torch SDPA (reference)
    fp8_block_ksm   : QK fp8_block + K smooth only
    fp8_block_qsm   : QK fp8_block + full smooth + Q kmeans=32
    fp8_block_qvsm  : QK fp8_block + full smooth + Q kmeans=32 + V kmeans=64
    fp8_block_full  : everything + V smooth

V quant fixed: ``fp8_block`` (FP32 scale per (B, S/64, H)).

Generates a video per strategy on the same prompt+seed, writes to
``bench/wan_videos/<strategy>.mp4``, then computes per-frame MSE/RMSE/PSNR/SSIM
in pixel space (uint8 RGB).
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

REPO_ROOT = Path("/home/dongyun/workspace/low-bit-attention-fake-quant")
WAN_ROOT = Path("/home/dongyun/workspace/Wan2.1")
sys.path.insert(0, str(WAN_ROOT))
sys.path.insert(0, str(REPO_ROOT))

import wan  # noqa: E402
from wan.configs import WAN_CONFIGS  # noqa: E402
from wan.utils.utils import cache_video  # noqa: E402

from bench.wan_attention_hook import (  # noqa: E402
    NO_QUANT,
    install_hook,
    set_quant_cfg,
)
from low_bit_fake_quant import QuantConfig  # noqa: E402


# ---------------------------------------------------------------------------
# Quant config matrix (user-specified)
# ---------------------------------------------------------------------------


def make_cfg(name: str) -> Optional[QuantConfig]:
    """All V variants use ``v_quant='fp8_block'``.

    Block sizes chosen to satisfy BOTH:
      (a) be a power of 2 (Triton's tl.arange requires this), and
      (b) divide Wan2.1 480p token length S=10944 (so no LCM padding).
    The largest power of 2 that divides 10944 = 64 × 171 is 64. So we use
    fp8_block_size = q_smooth_block_size = v_smooth_block_size = v_fp8_block_size = 64.
    This is smaller than the typical 128/256, but it avoids the hook padding
    bug that contaminated Q smoothing + Q kmeans (zero Q rows poisoned the
    256-group means, killing the smoothing benefit).
    """
    common = dict(
        qk_quant="fp8_block",
        v_quant="fp8_block",
        q_smooth_block_size=64,
        fp8_block_size=64,
        v_fp8_block_size=64,
        p_requant=True,
        p_requant_block_m=64,
        p_requant_block_n=64,
        p_quant="auto",
    )
    if name == "sdpa":
        return None  # baseline, no fake-quant
    if name == "fp8_block_ksm":
        return QuantConfig(
            **common, smoothing="k_only", q_kmeans_k=None,
            v_smooth_mode="off", v_kmeans_k=None,
        )
    if name == "fp8_block_qsm":
        return QuantConfig(
            **common, smoothing="full", q_kmeans_k=32,
            v_smooth_mode="off", v_kmeans_k=None,
        )
    if name == "fp8_block_qvsm":
        return QuantConfig(
            **common, smoothing="full", q_kmeans_k=32,
            v_smooth_mode="off", v_kmeans_k=64,
        )
    if name == "fp8_block_full":
        return QuantConfig(
            **common, smoothing="full", q_kmeans_k=32,
            v_smooth_mode="per_block", v_smooth_block_size=64, v_kmeans_k=64,
        )
    raise ValueError(f"unknown strategy: {name!r}")


STRATEGIES = ["sdpa", "fp8_block_ksm", "fp8_block_qsm", "fp8_block_qvsm", "fp8_block_full"]


# ---------------------------------------------------------------------------
# Video generation
# ---------------------------------------------------------------------------


_LOADED_MODEL = None


def load_model(ckpt_dir: Path):
    """Load WanT2V once and cache. Saves ~50s per subsequent strategy."""
    global _LOADED_MODEL
    if _LOADED_MODEL is not None:
        return _LOADED_MODEL
    wan_cfg = WAN_CONFIGS["t2v-14B"]
    print(f"[load] WanT2V from {ckpt_dir}")
    t_load = time.time()
    _LOADED_MODEL = wan.WanT2V(
        config=wan_cfg,
        checkpoint_dir=str(ckpt_dir),
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,
    )
    print(f"[load] model loaded in {time.time() - t_load:.1f}s")
    return _LOADED_MODEL


def generate_one(
    strategy: str,
    prompt: str,
    *,
    ckpt_dir: Path,
    out_dir: Path,
    seed: int,
    size: str = "832*480",
    frame_num: int = 81,
    sample_steps: int = 50,
    sample_shift: float = 5.0,
    sample_guide_scale: float = 5.0,
    sample_solver: str = "unipc",
) -> Path:
    """Run a single Wan2.1 T2V generation with the named strategy.

    Reuses a globally cached WanT2V instance — only the quant cfg flips
    between runs, which is a cheap module-level state change.
    """
    out_path = out_dir / f"{strategy}.mp4"
    if out_path.exists():
        print(f"[skip] {out_path} exists")
        return out_path

    cfg = make_cfg(strategy)
    set_quant_cfg(cfg)
    wan_t2v = load_model(ckpt_dir)
    wan_cfg = WAN_CONFIGS["t2v-14B"]

    print(f"[{strategy}] generating size={size} frames={frame_num} steps={sample_steps} "
          f"shift={sample_shift} guide={sample_guide_scale} seed={seed}")
    t_gen = time.time()
    video = wan_t2v.generate(
        input_prompt=prompt,
        size=tuple(int(x) for x in size.split("*")),
        frame_num=frame_num,
        shift=sample_shift,
        sample_solver=sample_solver,
        sampling_steps=sample_steps,
        guide_scale=sample_guide_scale,
        n_prompt="",
        seed=seed,
        offload_model=False,   # keep model on GPU; saves reload between runs
    )
    print(f"[{strategy}] generation took {time.time() - t_gen:.1f}s; saving to {out_path}")

    cache_video(
        tensor=video[None],
        save_file=str(out_path),
        fps=wan_cfg.sample_fps,
        nrow=1,
        normalize=True,
        value_range=(-1, 1),
    )

    del video
    gc.collect()
    torch.cuda.empty_cache()
    return out_path


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def load_video_frames(path: Path) -> np.ndarray:
    """Read all frames of mp4 as uint8 (T, H, W, 3) RGB."""
    import imageio.v3 as iio
    frames = iio.imread(path, plugin="pyav")
    if frames.ndim == 3:
        frames = frames[None]
    return frames  # (T, H, W, 3) uint8


def compute_video_metrics(ref_path: Path, test_path: Path) -> dict:
    """Per-frame MSE/RMSE/PSNR/SSIM, then averaged over T."""
    from skimage.metrics import structural_similarity as ssim

    ref = load_video_frames(ref_path).astype(np.float64)  # (T, H, W, 3) 0..255
    tst = load_video_frames(test_path).astype(np.float64)
    if ref.shape != tst.shape:
        T = min(ref.shape[0], tst.shape[0])
        ref = ref[:T]
        tst = tst[:T]

    diff = ref - tst
    mse = float((diff ** 2).mean())
    rmse = float(math.sqrt(mse))
    # PSNR (over uint8 range 255).
    psnr = float(10.0 * math.log10(255.0 ** 2 / max(mse, 1e-12)))
    # SSIM averaged per-frame, mean over channels.
    ssims = []
    for t in range(ref.shape[0]):
        s = ssim(ref[t].astype(np.uint8), tst[t].astype(np.uint8), channel_axis=-1, data_range=255)
        ssims.append(s)
    return {
        "mse_pixel": mse,
        "rmse_pixel": rmse,
        "psnr_db": psnr,
        "ssim": float(np.mean(ssims)),
        "ssim_min": float(np.min(ssims)),
        "T": int(ref.shape[0]),
        "H": int(ref.shape[1]),
        "W": int(ref.shape[2]),
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", type=Path, default=Path("/home/dongyun/models/Wan2.1-T2V-14B"))
    ap.add_argument("--out-dir", type=Path, default=Path("bench/wan_videos"))
    ap.add_argument("--prompt-file", type=Path, default=Path("/tmp/picked_prompt.txt"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--size", default="832*480")
    ap.add_argument("--frame-num", type=int, default=81)
    ap.add_argument("--sample-steps", type=int, default=50)
    ap.add_argument("--strategies", nargs="*", default=STRATEGIES)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Recover the picked prompt from the file (last paragraph).
    with args.prompt_file.open() as f:
        lines = [l.strip() for l in f if l.strip()]
    # Skip the leading meta lines (Total / Picked index / blank), take the last block.
    prompt = lines[-1]
    print(f"Prompt ({len(prompt)} chars): {prompt[:200]}...")

    install_hook()

    for strategy in args.strategies:
        print(f"\n========== strategy={strategy} ==========")
        generate_one(
            strategy, prompt,
            ckpt_dir=args.ckpt_dir,
            out_dir=args.out_dir,
            seed=args.seed,
            size=args.size,
            frame_num=args.frame_num,
            sample_steps=args.sample_steps,
        )

    # Metrics
    print("\n========== metrics ==========")
    ref_path = args.out_dir / "sdpa.mp4"
    if not ref_path.exists():
        print(f"missing reference {ref_path}; skip metrics")
        return
    results = {}
    for strategy in args.strategies:
        if strategy == "sdpa":
            continue
        test_path = args.out_dir / f"{strategy}.mp4"
        if not test_path.exists():
            print(f"missing {test_path}; skip")
            continue
        m = compute_video_metrics(ref_path, test_path)
        results[strategy] = m
        print(f"{strategy:<22} MSE={m['mse_pixel']:>8.2f} RMSE={m['rmse_pixel']:>6.2f} "
              f"PSNR={m['psnr_db']:>6.2f}dB SSIM={m['ssim']:.4f} (min {m['ssim_min']:.4f})")

    out_json = args.out_dir / "metrics.json"
    with out_json.open("w") as f:
        json.dump({"prompt": prompt, "seed": args.seed, "size": args.size,
                   "frame_num": args.frame_num, "sample_steps": args.sample_steps,
                   "results": results}, f, indent=2)
    print(f"\nSaved metrics to {out_json}")


if __name__ == "__main__":
    main()
