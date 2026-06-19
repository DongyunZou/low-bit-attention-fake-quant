"""Sweep FP8 block-skip attention accuracy over the real Wan2.1 workloads.

For every workload (``layer_<L>/timestep_<T>.pt`` under the wan21 dataset) and
every skip threshold in the grid, this runs the ablation ladder

    L0  torch SDPA (bf16)                              -- numerical ground truth
    L1  no-quant / no-skip tiled simulator
    L2  fp8 Q/K/V, online P (fp32), no skip
    L3  fp8 Q/K/V + static P*256, no skip
    L4  fp8 Q/K/V + static P*256 + BLASST skip @ lambda

and records RMSE / MSE / cosine / relative-RMSE for every rung vs L0, the skip
error (L4 vs L3), the quant error (L3 vs L0), plus skip-rate, dropped-mass,
force-keep, and Q/K/V/P saturation / underflow diagnostics. Results (per-workload
JSON + flat CSV + stratified summary + env manifest) are written under
``agent_space/``.

Run (pinned to GPU3 per the study constraint):

    CUDA_VISIBLE_DEVICES=3 uv run python bench/eval_blasst_skip.py \
        --data-roots ~/dataset/v-dit/wan21_p1 ~/dataset/v-dit/wan21_p2 \
        --out-dir agent_space/blasst_skip
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import gc
import json
import os
import time
from pathlib import Path
from typing import Iterable, Optional

import torch

from low_bit_fake_quant.blasst_skip import (
    LADDER,
    LEVEL_FP8_QKV,
    LEVEL_FP8_STATIC_P,
    LEVEL_REFERENCE,
    apply_token_permutation,
    choose_local_block,
    invert_permutation,
    sdpa_ground_truth,
    simulate_workload,
    space_time_reorder_index,
)
from low_bit_fake_quant.skip_metrics import aggregate, compute_output_metrics

# Default geometric lambda grid: 0.0 (no skip) through an aggressive region.
DEFAULT_LAMBDAS = [0.0, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.2, 0.3, 0.5, 0.7]

# Pre-registered thresholds for the reference (L1) vs SDPA (L0) sanity gate.
L1_COS_MIN = 0.999
L1_REL_RMSE_MAX = 2e-2


@dataclasses.dataclass
class Workload:
    part: str
    layer: str
    timestep: str
    path: Path


def iter_workloads(roots: list[Path]) -> Iterable[Workload]:
    for root in roots:
        part = root.name
        for layer_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            for ts_file in sorted(layer_dir.glob("timestep_*.pt")):
                yield Workload(part=part, layer=layer_dir.name,
                               timestep=ts_file.stem, path=ts_file)


def stratified_sample(workloads: list, every: int) -> list:
    """Take every ``every``-th workload after sorting by (layer, timestep, part).

    This spreads a subset across layers and timesteps (rather than the first-N
    which all fall in layer_0), for representative development runs.
    """
    if every <= 1:
        return workloads
    ordered = sorted(workloads, key=lambda w: (w.layer, w.timestep, w.part))
    return ordered[::every]


def load_qkv(path: Path, device: torch.device):
    data = torch.load(path, map_location="cpu", weights_only=False)
    q = data["query"].to(torch.bfloat16)
    k = data["key"].to(torch.bfloat16)
    v = data["value"].to(torch.bfloat16)
    return (q.to(device, non_blocking=True),
            k.to(device, non_blocking=True),
            v.to(device, non_blocking=True))


def env_manifest(device: torch.device) -> dict:
    return {
        "gpu_name": torch.cuda.get_device_name(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "sdpa_backends": ["FLASH_ATTENTION", "EFFICIENT_ATTENTION"],
        "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "allow_tf32_cudnn": torch.backends.cudnn.allow_tf32,
    }


def run(
    roots: list[Path],
    *,
    out_dir: Path,
    lambdas: list[float],
    device: torch.device,
    matmul_dtype: torch.dtype,
    limit: Optional[int],
    sample_every: int = 1,
    reorder_thw: Optional[tuple] = None,
    reorder_order: tuple = ("t", "h", "w"),
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    workloads = list(iter_workloads(roots))
    if sample_every > 1:
        workloads = stratified_sample(workloads, sample_every)
    if limit is not None:
        workloads = workloads[:limit]
    if not workloads:
        raise RuntimeError(f"no workloads found under {roots}")

    manifest = env_manifest(device)
    manifest.update({
        "data_roots": [str(r) for r in roots],
        "lambdas": lambdas,
        "matmul_dtype": str(matmul_dtype),
        "num_workloads": len(workloads),
        "ablation_ladder": list(LADDER),
        "reorder_thw": list(reorder_thw) if reorder_thw else None,
        "reorder_block": list(choose_local_block(*reorder_thw)) if reorder_thw else None,
        "reorder_native_axis_order": list(reorder_order) if reorder_thw else None,
    })

    results = {"manifest": manifest, "workloads": []}
    flat_records: list[dict] = []      # one row per (workload, level/lambda) for CSV + agg

    for wi, wl in enumerate(workloads):
        t0 = time.time()
        q, k, v = load_qkv(wl.path, device)
        b, s, h, d = q.shape
        torch.cuda.synchronize()

        # ground truth is ALWAYS computed in native token order.
        ref = sdpa_ground_truth(q, k, v)
        torch.cuda.synchronize()

        # optional space-time reorder arm: permute Q/K/V identically, simulate
        # in reordered order, then inverse-permute every simulator output back
        # to native order so metrics are computed against the native reference.
        # This is a pure reindexing for the no-skip rungs; skipping differs
        # because the 128-token tiles now hold coherent space-time blocks.
        inv = None
        if reorder_thw is not None:
            t_, h_, w_ = reorder_thw
            if t_ * h_ * w_ != s:
                raise ValueError(f"reorder t*h*w={t_*h_*w_} != seqlen {s}")
            block = choose_local_block(t_, h_, w_)
            perm = space_time_reorder_index(t_, h_, w_, block=block,
                                            native_axis_order=reorder_order, device=device)
            inv = invert_permutation(perm)
            qs = apply_token_permutation(q, perm)
            ks = apply_token_permutation(k, perm)
            vs = apply_token_permutation(v, perm)
        else:
            qs, ks, vs = q, k, v

        res = simulate_workload(
            qs, ks, vs, skip_thresholds=lambdas, levels=LADDER,
            matmul_dtype=matmul_dtype,
        )
        torch.cuda.synchronize()

        if inv is not None:
            for lv in list(res.outputs):
                res.outputs[lv] = apply_token_permutation(res.outputs[lv], inv)
            for th in list(res.skip_outputs):
                res.skip_outputs[th] = apply_token_permutation(res.skip_outputs[th], inv)

        rec = {
            "part": wl.part, "layer": wl.layer, "timestep": wl.timestep,
            "path": str(wl.path), "shape": [b, s, h, d],
            "peak_memory_bytes": res.peak_memory_bytes,
            "quant_stats": {k2: dataclasses.asdict(v2) for k2, v2 in res.quant_stats.items()},
            "levels": {},
            "skip": {},
        }

        # ladder rungs vs ground truth
        level_metrics = {}
        for lv in (LEVEL_REFERENCE, LEVEL_FP8_QKV, LEVEL_FP8_STATIC_P):
            m = compute_output_metrics(res.outputs[lv], ref)
            level_metrics[lv] = m
            rec["levels"][lv] = m.asdict()
            flat_records.append({
                "part": wl.part, "layer": wl.layer, "timestep": wl.timestep,
                "level": lv, "lambda": None, "vs": "ground_truth", **m.asdict(),
            })

        # reference (L1) vs SDPA (L0) sanity gate (recorded, exploratory -> never aborts)
        l1 = level_metrics[LEVEL_REFERENCE]
        rec["reference_check"] = {
            "cosine_global": l1.cosine_global,
            "rel_rmse": l1.rel_rmse,
            "passes_cos": l1.cosine_global >= L1_COS_MIN,
            "passes_rel_rmse": l1.rel_rmse <= L1_REL_RMSE_MAX,
        }

        # total fp8-quant error: L3 vs L0 (already above), isolate via L3 ref
        l3_ref = res.outputs[LEVEL_FP8_STATIC_P]

        # skip rungs: L4(lambda) vs ground truth AND vs L3 (skip-induced only)
        for th in lambdas:
            out4 = res.skip_outputs[th]
            m_gt = compute_output_metrics(out4, ref)
            m_skip = compute_output_metrics(out4, l3_ref)
            diag = res.skip_diagnostics[th]
            rec["skip"][str(th)] = {
                "vs_ground_truth": m_gt.asdict(),
                "vs_no_skip_fp8": m_skip.asdict(),
                "diagnostics": dataclasses.asdict(diag),
            }
            flat_records.append({
                "part": wl.part, "layer": wl.layer, "timestep": wl.timestep,
                "level": "fp8_static_p_skip", "lambda": th, "vs": "ground_truth",
                "skip_rate": diag.skip_rate, "force_keep_count": diag.force_keep_count,
                "dropped_mass_mean": diag.dropped_mass_mean,
                "dropped_mass_p95": diag.dropped_mass_p95,
                "dropped_mass_max": diag.dropped_mass_max,
                **m_gt.asdict(),
            })
            flat_records.append({
                "part": wl.part, "layer": wl.layer, "timestep": wl.timestep,
                "level": "fp8_static_p_skip", "lambda": th, "vs": "no_skip_fp8",
                "skip_rate": diag.skip_rate, **m_skip.asdict(),
            })

        rec["seconds"] = time.time() - t0
        results["workloads"].append(rec)
        print(
            f"[{wi+1}/{len(workloads)}] {wl.part}/{wl.layer}/{wl.timestep}: "
            f"L1 cos={l1.cosine_global:.5f} relRMSE={l1.rel_rmse:.3e} "
            f"| L3 cos={level_metrics[LEVEL_FP8_STATIC_P].cosine_global:.5f} "
            f"| peak={res.peak_memory_bytes/1e9:.1f}GB ({rec['seconds']:.1f}s)",
            flush=True,
        )

        # persist incrementally
        with (out_dir / "results.json").open("w") as f:
            json.dump(results, f, indent=2, default=str)

        del q, k, v, ref, res, l3_ref
        gc.collect()
        torch.cuda.empty_cache()

    # stratified aggregation over ground-truth comparisons
    gt_records = [r for r in flat_records if r.get("vs") == "ground_truth"]
    metric_keys = ["mse", "rmse", "rel_rmse", "cosine_global", "cosine_row_mean",
                   "cosine_row_p05", "cosine_row_min"]
    summary = {}
    for lv in LADDER:
        subset = [r for r in gt_records if r["level"] == lv]
        if lv == "fp8_static_p_skip":
            by_lambda = {}
            for th in lambdas:
                ls = [r for r in subset if r["lambda"] == th]
                by_lambda[str(th)] = {
                    "accuracy": aggregate(ls, metric_keys),
                    "skip_rate": aggregate(ls, ["skip_rate"])["global"].get("skip_rate"),
                    "dropped_mass_p95": aggregate(ls, ["dropped_mass_p95"])["global"].get("dropped_mass_p95"),
                }
            summary[lv] = by_lambda
        else:
            summary[lv] = aggregate(subset, metric_keys)
    results["summary"] = summary

    with (out_dir / "results.json").open("w") as f:
        json.dump(results, f, indent=2, default=str)
    with (out_dir / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2, default=str)

    # flat CSV
    if flat_records:
        all_keys: list[str] = []
        for r in flat_records:
            for kk in r:
                if kk not in all_keys:
                    all_keys.append(kk)
        with (out_dir / "records.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=all_keys)
            w.writeheader()
            w.writerows(flat_records)

    print(f"\nwrote results to {out_dir}/ (results.json, records.csv, manifest.json)")
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    home = Path(os.path.expanduser("~"))
    ap.add_argument("--data-roots", nargs="+", type=Path,
                    default=[home / "dataset/v-dit/wan21_p1",
                             home / "dataset/v-dit/wan21_p2"])
    ap.add_argument("--out-dir", type=Path, default=Path("agent_space/blasst_skip"))
    ap.add_argument("--lambdas", nargs="*", type=float, default=DEFAULT_LAMBDAS)
    ap.add_argument("--limit", type=int, default=None, help="cap number of workloads")
    ap.add_argument("--sample-every", type=int, default=1,
                    help="take every Nth workload (stratified by layer/timestep) for subset dev runs")
    ap.add_argument("--matmul-dtype", choices=["bf16", "fp32"], default="bf16")
    ap.add_argument("--reorder", nargs=3, type=int, metavar=("T", "H", "W"), default=None,
                    help="space-time reorder arm: (t,h,w) latent grid, product must equal seqlen "
                         "(the true Wan2.1 grid is unconfirmed; pick via rank_reorder_layouts.py)")
    ap.add_argument("--reorder-order", nargs=3, choices=["t", "h", "w"], default=["t", "h", "w"],
                    metavar=("A", "B", "C"), help="native flatten order of the latent grid")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA device required")
    device = torch.device("cuda")
    dtype = torch.bfloat16 if args.matmul_dtype == "bf16" else torch.float32

    roots = [Path(os.path.expanduser(str(r))) for r in args.data_roots]
    run(roots, out_dir=args.out_dir, lambdas=args.lambdas, device=device,
        matmul_dtype=dtype, limit=args.limit, sample_every=args.sample_every,
        reorder_thw=tuple(args.reorder) if args.reorder else None,
        reorder_order=tuple(args.reorder_order))


if __name__ == "__main__":
    main()
