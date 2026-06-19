"""Emit full per-lambda degradation tables (markdown) from a sweep result dir.

Reads ``results.json`` (summary, aggregated globally) and ``records.csv`` (for
the skip-only error, L4 vs L3) and prints one markdown table per run with, for
each lambda: skip-rate, RMSE, MSE, cosine, relRMSE, dropped-mass p95, and
skip-only relRMSE. Also emits the no-skip ablation ladder. Used to build the
final analysis.

Run:
    uv run python bench/make_analysis_tables.py \
        --native agent_space/blasst_skip --reordered agent_space/blasst_skip_reordered
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

LADDER_NOSKIP = ["bf16_ref", "fp8_qkv", "fp8_static_p"]


def skip_only_relrmse(out_dir: Path):
    """Mean skip-only relRMSE (L4 vs no-skip fp8) per lambda from records.csv."""
    rows = list(csv.DictReader((out_dir / "records.csv").open()))
    by_lambda = defaultdict(list)
    for r in rows:
        if r["level"] == "fp8_static_p_skip" and r["vs"] == "no_skip_fp8":
            by_lambda[float(r["lambda"])].append(float(r["rel_rmse"]))
    return {lam: statistics.mean(v) for lam, v in by_lambda.items()}


def ladder_table(res: dict) -> str:
    s = res["summary"]
    lines = ["| rung | cosine | relRMSE | RMSE | MSE | worst-workload cos |",
             "|---|---|---|---|---|---|"]
    for lv in LADDER_NOSKIP:
        g = s[lv]["global"]
        lines.append(
            f"| {lv} | {g['cosine_global']['mean']:.5f} | {g['rel_rmse']['mean']:.3e} | "
            f"{g['rmse']['mean']:.3e} | {g['mse']['mean']:.3e} | {g['cosine_global']['worst_min']:.5f} |"
        )
    return "\n".join(lines)


def skip_table(res: dict, skip_only: dict) -> str:
    s = res["summary"]["fp8_static_p_skip"]
    lines = ["| λ | skip-rate | cosine | relRMSE | RMSE | MSE | dropped-mass p95 | skip-only relRMSE |",
             "|---|---|---|---|---|---|---|---|"]
    for th, v in s.items():
        acc = v["accuracy"]["global"]
        lam = float(th)
        so = skip_only.get(lam, float("nan"))
        lines.append(
            f"| {lam:g} | {v['skip_rate']['mean']:.3f} | {acc['cosine_global']['mean']:.5f} | "
            f"{acc['rel_rmse']['mean']:.3e} | {acc['rmse']['mean']:.3e} | {acc['mse']['mean']:.3e} | "
            f"{v['dropped_mass_p95']['mean']:.3e} | {so:.3e} |"
        )
    return "\n".join(lines)


def emit(out_dir: Path, title: str) -> str:
    res = json.load((out_dir / "results.json").open())
    so = skip_only_relrmse(out_dir)
    man = res["manifest"]
    head = f"### {title}\n"
    if man.get("reorder_thw"):
        head += (f"\nReorder grid `{man['reorder_thw']}` block `{man['reorder_block']}` "
                 f"order `{man.get('reorder_native_axis_order')}`; {man['num_workloads']} workloads.\n")
    else:
        head += f"\nNative order; {man['num_workloads']} workloads.\n"
    return (head + "\n**Ablation ladder (no skip), mean vs SDPA:**\n\n" + ladder_table(res)
            + "\n\n**Block-skip ladder (mean over workloads):**\n\n" + skip_table(res, so) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--native", type=Path, default=Path("agent_space/blasst_skip"))
    ap.add_argument("--reordered", type=Path, default=Path("agent_space/blasst_skip_reordered"))
    ap.add_argument("--out", type=Path, default=None, help="write markdown here instead of stdout")
    args = ap.parse_args()
    parts = [emit(args.native, "Native order")]
    if args.reordered.exists():
        parts.append(emit(args.reordered, "Space-time reordered"))
    md = "\n\n".join(parts)
    if args.out:
        args.out.write_text(md)
        print(f"wrote {args.out}")
    else:
        print(md)


if __name__ == "__main__":
    main()
