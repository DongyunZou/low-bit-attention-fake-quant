"""Generate Wan2.1 videos for end-to-end P-quant experiments.

This script writes one SDPA reference video plus five fake-quant variants that
match the attention-level accuracy snapshot in the README:

  sdpa
  k_smooth_static
  q_smooth_static
  q_smooth_dynamic
  full_static
  full_dynamic

The output videos are intended to be evaluated with ``bench/eval_video_dirs.py``.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Optional

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from bench.wan_attention_hook import install_hook, set_quant_cfg  # noqa: E402
from low_bit_fake_quant import QuantConfig  # noqa: E402


STRATEGIES = [
    "sdpa",
    "k_smooth_static",
    "q_smooth_static",
    "q_smooth_dynamic",
    "full_static",
    "full_dynamic",
]


def make_cfg(name: str) -> Optional[QuantConfig]:
    """Return the quant config for one end-to-end video strategy."""
    common = dict(
        qk_quant="fp8_block",
        v_quant="fp8_channel",
        q_smooth_block_size=64,
        fp8_block_size=64,
        p_requant=True,
        p_requant_block_m=64,
        p_requant_block_n=64,
    )
    if name == "sdpa":
        return None
    if name == "k_smooth_static":
        return QuantConfig(
            **common,
            smoothing="k_only",
            q_kmeans_k=None,
            v_smooth_mode="off",
            v_kmeans_k=None,
            p_quant="elementwise",
            rowmax_mode="online",
        )
    if name == "q_smooth_static":
        return QuantConfig(
            **common,
            smoothing="full",
            q_kmeans_k=32,
            v_smooth_mode="off",
            v_kmeans_k=None,
            p_quant="elementwise",
            rowmax_mode="online",
        )
    if name == "q_smooth_dynamic":
        return QuantConfig(
            **common,
            smoothing="full",
            q_kmeans_k=32,
            v_smooth_mode="off",
            v_kmeans_k=None,
            p_quant="dynamic",
            rowmax_mode="qm_k",
        )
    if name == "full_static":
        return QuantConfig(
            **common,
            smoothing="full",
            q_kmeans_k=32,
            v_smooth_mode="per_block",
            v_smooth_block_size=64,
            v_kmeans_k=64,
            p_quant="elementwise",
            rowmax_mode="online",
        )
    if name == "full_dynamic":
        return QuantConfig(
            **common,
            smoothing="full",
            q_kmeans_k=32,
            v_smooth_mode="per_block",
            v_smooth_block_size=64,
            v_kmeans_k=64,
            p_quant="dynamic",
            rowmax_mode="qm_k",
        )
    raise ValueError(f"unknown strategy: {name!r}")


def read_prompt(prompt_file: Path, prompt: str | None) -> str:
    if prompt is not None:
        return prompt
    with prompt_file.open() as f:
        lines = [line.strip() for line in f if line.strip()]
    if not lines:
        raise RuntimeError(f"empty prompt file: {prompt_file}")
    return lines[-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wan-root", type=Path, required=True, help="Path to the Wan2.1 repo")
    ap.add_argument("--ckpt-dir", type=Path, required=True, help="Path to Wan2.1-T2V-14B")
    ap.add_argument("--out-dir", type=Path, default=Path("bench/wan_e2e_pquant"))
    ap.add_argument("--prompt-file", type=Path, default=Path("/tmp/picked_prompt.txt"))
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--size", default="832*480")
    ap.add_argument("--frame-num", type=int, default=81)
    ap.add_argument("--sample-steps", type=int, default=50)
    ap.add_argument("--sample-shift", type=float, default=5.0)
    ap.add_argument("--sample-guide-scale", type=float, default=5.0)
    ap.add_argument("--strategies", nargs="*", default=STRATEGIES)
    ap.add_argument("--offload-model", action="store_true")
    ap.add_argument("--t5-cpu", action="store_true")
    args = ap.parse_args()

    sys.path.insert(0, str(args.wan_root.resolve()))
    import wan  # noqa: PLC0415
    from wan.configs import WAN_CONFIGS  # noqa: PLC0415
    from wan.utils.utils import cache_video  # noqa: PLC0415

    prompt = read_prompt(args.prompt_file, args.prompt)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    run_config = {
        "prompt": prompt,
        "seed": args.seed,
        "size": args.size,
        "frame_num": args.frame_num,
        "sample_steps": args.sample_steps,
        "sample_shift": args.sample_shift,
        "sample_guide_scale": args.sample_guide_scale,
        "strategies": args.strategies,
        "configs": {
            name: None if make_cfg(name) is None else make_cfg(name).__dict__
            for name in args.strategies
        },
    }
    with (args.out_dir / "run_config.json").open("w") as f:
        json.dump(run_config, f, indent=2, default=str)

    install_hook()
    wan_cfg = WAN_CONFIGS["t2v-14B"]
    print(f"[load] WanT2V from {args.ckpt_dir}")
    start = time.time()
    wan_t2v = wan.WanT2V(
        config=wan_cfg,
        checkpoint_dir=str(args.ckpt_dir),
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=args.t5_cpu,
    )
    print(f"[load] done in {time.time() - start:.1f}s")

    for strategy in args.strategies:
        out_path = args.out_dir / f"{strategy}.mp4"
        if out_path.exists():
            print(f"[skip] {out_path}")
            continue
        cfg = make_cfg(strategy)
        set_quant_cfg(cfg)
        print(f"\n[{strategy}] generating {args.size}, {args.frame_num} frames")
        start = time.time()
        video = wan_t2v.generate(
            input_prompt=prompt,
            size=tuple(int(x) for x in args.size.split("*")),
            frame_num=args.frame_num,
            shift=args.sample_shift,
            sample_solver="unipc",
            sampling_steps=args.sample_steps,
            guide_scale=args.sample_guide_scale,
            n_prompt="",
            seed=args.seed,
            offload_model=args.offload_model,
        )
        print(f"[{strategy}] generated in {time.time() - start:.1f}s -> {out_path}")
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


if __name__ == "__main__":
    main()
