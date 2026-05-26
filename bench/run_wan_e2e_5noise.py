"""Run Wan2.1 end-to-end video metrics over five independent noise seeds.

This is a thin launcher around ``bench/gen_wan_e2e_pquant.py``.  It runs one
seed per GPU, then evaluates each seed directory against its own SDPA video and
aggregates metrics across seeds.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_PROMPT = (
    "A skateboarding scene in a dynamic street style, cinematic camera motion, "
    "natural lighting, high detail, smooth motion, realistic urban background."
)

STRATEGIES = [
    "sdpa",
    "k_smooth_static",
    "q_smooth_static",
    "full_static",
    "full_dynamic",
]

DISPLAY_NAMES = {
    "sdpa": "SDPA baseline",
    "k_smooth_static": "K smooth only",
    "q_smooth_static": "QK smooth",
    "full_static": "QKV smooth",
    "full_dynamic": "QKV smooth + fixed rowmax + dynamic P scale",
}


def _run_generation(args: argparse.Namespace, seed: int, gpu: str, log_path: Path) -> subprocess.Popen:
    seed_dir = args.out_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(REPO_ROOT / "bench" / "gen_wan_e2e_pquant.py"),
        "--wan-root",
        str(args.wan_root),
        "--ckpt-dir",
        str(args.ckpt_dir),
        "--out-dir",
        str(seed_dir),
        "--prompt",
        args.prompt,
        "--seed",
        str(seed),
        "--size",
        args.size,
        "--frame-num",
        str(args.frame_num),
        "--sample-steps",
        str(args.sample_steps),
        "--sample-shift",
        str(args.sample_shift),
        "--sample-guide-scale",
        str(args.sample_guide_scale),
        "--strategies",
        *STRATEGIES,
    ]
    if args.t5_cpu:
        cmd.append("--t5-cpu")
    if args.offload_model:
        cmd.append("--offload-model")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    log_f = log_path.open("w", encoding="utf-8")
    print(f"[launch] seed={seed} gpu={gpu} log={log_path}")
    return subprocess.Popen(
        cmd,
        cwd=REPO_ROOT,
        env=env,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        text=True,
    )


def _run_eval(args: argparse.Namespace, seed: int, gpu: str) -> Path:
    seed_dir = args.out_dir / f"seed_{seed}"
    metrics_path = seed_dir / "metrics.json"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "bench" / "eval_video_dirs.py"),
        "--pred-dir",
        str(seed_dir),
        "--ref-video",
        str(seed_dir / "sdpa.mp4"),
        "--output-json",
        str(metrics_path),
        "--match-mode",
        "stem",
        "--device",
        "cuda",
    ]
    if args.disable_lpips:
        cmd.append("--disable-lpips")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    print(f"[eval] seed={seed} gpu={gpu}")
    subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)
    return metrics_path


def _aggregate(out_dir: Path, seeds: list[int]) -> dict:
    by_strategy: dict[str, list[dict]] = {}
    seed_reports = {}
    for seed in seeds:
        path = out_dir / f"seed_{seed}" / "metrics.json"
        with path.open(encoding="utf-8") as f:
            report = json.load(f)
        seed_reports[str(seed)] = str(path)
        for item in report["per_video"]:
            by_strategy.setdefault(item["key"], []).append(item)

    summary = {
        "seeds": seeds,
        "seed_reports": seed_reports,
        "strategies": {k: DISPLAY_NAMES.get(k, k) for k in STRATEGIES},
        "mean_by_strategy": {},
    }
    for strategy in STRATEGIES:
        if strategy == "sdpa":
            continue
        rows = by_strategy.get(strategy, [])
        if not rows:
            continue
        metrics = {}
        for metric in ("psnr", "ssim", "lpips"):
            values = [r[metric] for r in rows if r.get(metric) is not None]
            metrics[metric] = sum(values) / len(values) if values else None
        summary["mean_by_strategy"][strategy] = {
            "display_name": DISPLAY_NAMES.get(strategy, strategy),
            "num_seeds": len(rows),
            **metrics,
        }
    return summary


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wan-root", type=Path, required=True)
    ap.add_argument("--ckpt-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("bench/wan_e2e_5noise"))
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    ap.add_argument("--gpus", default="1,2,3,4,6")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--size", default="832*480")
    ap.add_argument("--frame-num", type=int, default=81)
    ap.add_argument("--sample-steps", type=int, default=50)
    ap.add_argument("--sample-shift", type=float, default=5.0)
    ap.add_argument("--sample-guide-scale", type=float, default=5.0)
    ap.add_argument("--t5-cpu", action="store_true", default=True)
    ap.add_argument("--no-t5-cpu", dest="t5_cpu", action="store_false")
    ap.add_argument("--offload-model", action="store_true", default=True)
    ap.add_argument("--no-offload-model", dest="offload_model", action="store_false")
    ap.add_argument("--disable-lpips", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    gpus = [x.strip() for x in args.gpus.split(",") if x.strip()]
    if len(gpus) < len(args.seeds):
        raise RuntimeError(f"Need at least one GPU per seed; got {gpus} for seeds {args.seeds}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = args.out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "launcher_config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "wan_root": str(args.wan_root),
                "ckpt_dir": str(args.ckpt_dir),
                "seeds": args.seeds,
                "gpus": gpus[: len(args.seeds)],
                "prompt": args.prompt,
                "strategies": STRATEGIES,
                "display_names": DISPLAY_NAMES,
                "size": args.size,
                "frame_num": args.frame_num,
                "sample_steps": args.sample_steps,
                "sample_shift": args.sample_shift,
                "sample_guide_scale": args.sample_guide_scale,
                "t5_cpu": args.t5_cpu,
                "offload_model": args.offload_model,
            },
            f,
            indent=2,
        )

    procs = []
    for seed, gpu in zip(args.seeds, gpus, strict=False):
        procs.append((seed, gpu, _run_generation(args, seed, gpu, log_dir / f"seed_{seed}.log")))

    failed = []
    for seed, gpu, proc in procs:
        code = proc.wait()
        if code != 0:
            failed.append((seed, gpu, code))
        print(f"[done] seed={seed} gpu={gpu} returncode={code}")
    if failed:
        raise RuntimeError(f"Generation failed: {failed}. See logs in {log_dir}")

    eval_gpu = gpus[0]
    for seed in args.seeds:
        _run_eval(args, seed, eval_gpu)

    summary = _aggregate(args.out_dir, args.seeds)
    summary_path = args.out_dir / "summary_metrics.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary["mean_by_strategy"], indent=2))
    print(f"[summary] {summary_path.resolve()}")


if __name__ == "__main__":
    main()
