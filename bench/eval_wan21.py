"""Evaluate fake-quant attention vs torch SDPA on real wan21 workloads.

Dataset layout:
    /home/dongyun/dataset/v-dit/wan21_p1/layer_<L>/timestep_<T>.pt
    each file: dict[str, torch.Tensor] with keys {query, key, value}
    each tensor: (B, S, H, D) BF16, default S=69120, H=40, D=128.

For every quant configuration in the matrix we run:
    fake_q_out = fake_quant_attention(Q, K, V, cfg)
    ref_out    = reference_attention(Q, K, V)         # torch SDPA on bf16
and report per-element MSE / RMSE / cosine of the flattened tensors.

Run:
    uv run python bench/eval_wan21.py \\
        --data-root /home/dongyun/dataset/v-dit/wan21_p1 \\
        --output bench/results_wan21.json

Pass ``--configs fp8_block_full_kmeans32`` etc. to restrict the matrix.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import math
import os
import time
from pathlib import Path
from typing import Iterable, Optional

import torch

from low_bit_fake_quant import (
    PreprocessCache,
    QuantConfig,
    fake_quant_attention,
    prepare_for_attention,
    reference_attention,
)


# ---------------------------------------------------------------------------
# Quant config matrix
# ---------------------------------------------------------------------------


def build_matrix() -> dict[str, QuantConfig]:
    """All quant configurations covered by the goal + V smoothing extension.

    QK quant     : fp8_block, mxfp8
    V  quant     : fp8_channel, fp8_block, mxfp8
    Smoothing    : off, k_only, full   (Q/K SageAttention smoothing)
    Q kmeans     : off, 32, 64
    V smoothing  : off, per_block(64)
    P requant    : off, on
    """
    matrix: dict[str, QuantConfig] = {}
    for qk in ("fp8_block", "mxfp8"):
        for vq in ("fp8_channel", "fp8_block", "mxfp8"):
            for sm in ("off", "k_only", "full"):
                for kk in (None, 32, 64):
                    for vs in ("off", "per_block"):
                        for pr in (False, True):
                            name = (
                                f"{qk}__{vq}__{sm}"
                                f"__kmeans_{'off' if kk is None else kk}"
                                f"__vsmooth_{vs}"
                                f"__P_{'on' if pr else 'off'}"
                            )
                            matrix[name] = QuantConfig(
                                qk_quant=qk,
                                v_quant=vq,
                                smoothing=sm,
                                q_kmeans_k=kk,
                                q_smooth_block_size=256,
                                fp8_block_size=128,
                                mxfp8_block_size=32,
                                v_fp8_block_size=64,
                                v_mxfp8_block_size=64,
                                v_smooth_mode=vs,
                                v_smooth_block_size=64,
                                p_requant=pr,
                                p_requant_q_chunk=512,
                                p_requant_block_m=64,
                                p_requant_block_n=64,
                            )
    return matrix


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Metrics:
    mse: float
    rmse: float
    cosine: float
    rel_l2: float
    max_abs_err: float

    def asdict(self) -> dict[str, float]:
        return dataclasses.asdict(self)


def compute_metrics(o_quant: torch.Tensor, o_ref: torch.Tensor) -> Metrics:
    """Compute MSE/RMSE/Cosine + relative L2 + max abs error in FP32."""
    a = o_quant.float().flatten()
    b = o_ref.float().flatten()
    diff = a - b
    mse = float(diff.pow(2).mean().item())
    rmse = math.sqrt(mse)
    # cosine over the full flattened vector to capture global alignment.
    eps = 1e-12
    cos = float(
        torch.dot(a, b).item() / max(eps, math.sqrt(float(a.pow(2).sum().item()))) /
        max(eps, math.sqrt(float(b.pow(2).sum().item())))
    )
    ref_norm = math.sqrt(float(b.pow(2).sum().item()))
    rel_l2 = float(torch.linalg.vector_norm(diff).item() / max(eps, ref_norm))
    max_abs = float(diff.abs().max().item())
    return Metrics(mse=mse, rmse=rmse, cosine=cos, rel_l2=rel_l2, max_abs_err=max_abs)


# ---------------------------------------------------------------------------
# Workload iteration
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Workload:
    layer: str
    timestep: str
    path: Path


def iter_workloads(root: Path) -> Iterable[Workload]:
    for layer_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for ts_file in sorted(layer_dir.glob("timestep_*.pt")):
            yield Workload(layer=layer_dir.name, timestep=ts_file.stem, path=ts_file)


def load_qkv(path: Path, device: torch.device, dtype: Optional[torch.dtype]):
    """Load a single timestep .pt as (q,k,v) on ``device``."""
    data = torch.load(path, map_location="cpu", weights_only=False)
    q = data["query"]
    k = data["key"]
    v = data["value"]
    if dtype is not None:
        q = q.to(dtype)
        k = k.to(dtype)
        v = v.to(dtype)
    return (
        q.to(device, non_blocking=True),
        k.to(device, non_blocking=True),
        v.to(device, non_blocking=True),
    )


# ---------------------------------------------------------------------------
# Eval driver
# ---------------------------------------------------------------------------


def run_eval(
    data_root: Path,
    configs: dict[str, QuantConfig],
    *,
    output: Path,
    limit_workloads: Optional[int],
    device: torch.device,
    config_filter: Optional[set[str]],
) -> dict:
    results: dict = {
        "dataset": str(data_root),
        "device": str(device),
        "torch_version": torch.__version__,
        "configs": {name: dataclasses.asdict(cfg) for name, cfg in configs.items()},
        "workloads": [],
    }
    workloads = list(iter_workloads(data_root))
    if limit_workloads is not None:
        workloads = workloads[:limit_workloads]
    if not workloads:
        raise RuntimeError(f"no workloads found under {data_root}")

    selected_configs = (
        {n: c for n, c in configs.items() if n in config_filter}
        if config_filter
        else configs
    )

    for w_idx, wl in enumerate(workloads):
        t_load = time.time()
        q, k, v = load_qkv(wl.path, device=device, dtype=torch.bfloat16)
        b, s, h, d = q.shape
        load_s = time.time() - t_load

        torch.cuda.synchronize()
        t_ref = time.time()
        ref_out = reference_attention(q, k, v)
        torch.cuda.synchronize()
        ref_s = time.time() - t_ref

        # Build preprocess cache keyed by (smoothing, q_kmeans_k,
        # q_smooth_block_size). All configs that share these reuse the
        # kmeans + smooth_k + group_mean_q outputs.
        preprocess_cache: dict[tuple, PreprocessCache] = {}
        t_pre = time.time()
        for cfg in selected_configs.values():
            key = (cfg.smoothing, cfg.q_kmeans_k, cfg.q_smooth_block_size,
                   cfg.q_kmeans_iters, cfg.q_kmeans_seed,
                   cfg.v_smooth_mode, cfg.v_smooth_block_size)
            if key in preprocess_cache:
                continue
            preprocess_cache[key] = prepare_for_attention(q, k, v, cfg)
        torch.cuda.synchronize()
        pre_s = time.time() - t_pre

        wl_record = {
            "layer": wl.layer,
            "timestep": wl.timestep,
            "path": str(wl.path),
            "shape": [b, s, h, d],
            "load_seconds": load_s,
            "reference_seconds": ref_s,
            "preprocess_seconds": pre_s,
            "preprocess_cache_entries": len(preprocess_cache),
            "configs": {},
        }
        print(
            f"[{w_idx+1}/{len(workloads)}] {wl.layer}/{wl.timestep}: "
            f"load={load_s:.1f}s ref={ref_s:.2f}s preproc={pre_s:.2f}s "
            f"({len(preprocess_cache)} unique preproc states)"
        )

        for cfg_name, cfg in selected_configs.items():
            key = (cfg.smoothing, cfg.q_kmeans_k, cfg.q_smooth_block_size,
                   cfg.q_kmeans_iters, cfg.q_kmeans_seed,
                   cfg.v_smooth_mode, cfg.v_smooth_block_size)
            torch.cuda.synchronize()
            t_q = time.time()
            try:
                quant_out = fake_quant_attention(
                    q, k, v, cfg, preprocess_cache=preprocess_cache[key]
                )
            except Exception as e:  # noqa: BLE001
                wl_record["configs"][cfg_name] = {"error": repr(e)}
                continue
            torch.cuda.synchronize()
            cfg_s = time.time() - t_q

            metrics = compute_metrics(quant_out, ref_out)
            wl_record["configs"][cfg_name] = {
                "metrics": metrics.asdict(),
                "seconds": cfg_s,
            }
            del quant_out
            torch.cuda.empty_cache()

            print(
                f"[{w_idx+1}/{len(workloads)}] {wl.layer}/{wl.timestep} "
                f"{cfg_name}: MSE={metrics.mse:.3e} RMSE={metrics.rmse:.3e} "
                f"cos={metrics.cosine:.6f} rel_l2={metrics.rel_l2:.3e} "
                f"max_abs={metrics.max_abs_err:.3e}  ({cfg_s:.2f}s)"
            )

        results["workloads"].append(wl_record)
        # Persist incrementally so a long run isn't lost.
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w") as f:
            json.dump(results, f, indent=2, default=str)

        del q, k, v, ref_out
        gc.collect()
        torch.cuda.empty_cache()

    return results


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def summarize(results: dict) -> dict:
    """Aggregate per-config metrics across workloads."""
    per_config: dict[str, dict[str, list[float]]] = {}
    for wl in results["workloads"]:
        for cfg_name, payload in wl["configs"].items():
            if "metrics" not in payload:
                continue
            slot = per_config.setdefault(cfg_name, {"mse": [], "rmse": [], "cosine": [], "rel_l2": [], "max_abs_err": []})
            for k, v in payload["metrics"].items():
                slot[k].append(v)

    summary: dict[str, dict[str, float]] = {}
    for cfg_name, slot in per_config.items():
        agg = {}
        for k, vals in slot.items():
            if not vals:
                continue
            agg[f"{k}_mean"] = sum(vals) / len(vals)
            agg[f"{k}_min"] = min(vals)
            agg[f"{k}_max"] = max(vals)
        summary[cfg_name] = agg
    return summary


def print_summary(summary: dict) -> None:
    header = f"{'config':<48} {'cos_mean':>10} {'cos_min':>10} {'mse_mean':>12} {'rmse_mean':>12} {'rel_l2_mean':>12}"
    print(header)
    print("-" * len(header))
    rows = []
    for cfg_name, agg in summary.items():
        if "cosine_mean" not in agg:
            continue
        rows.append((cfg_name, agg))
    rows.sort(key=lambda r: -r[1].get("cosine_mean", 0.0))
    for cfg_name, agg in rows:
        print(
            f"{cfg_name:<48} "
            f"{agg.get('cosine_mean', float('nan')):>10.6f} "
            f"{agg.get('cosine_min', float('nan')):>10.6f} "
            f"{agg.get('mse_mean', float('nan')):>12.3e} "
            f"{agg.get('rmse_mean', float('nan')):>12.3e} "
            f"{agg.get('rel_l2_mean', float('nan')):>12.3e} "
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path("/home/dongyun/dataset/v-dit/wan21_p1"))
    ap.add_argument("--output", type=Path, default=Path("bench/results_wan21.json"))
    ap.add_argument("--limit-workloads", type=int, default=None)
    ap.add_argument("--configs", nargs="*", default=None, help="Restrict to these config names")
    ap.add_argument("--list-configs", action="store_true")
    args = ap.parse_args()

    configs = build_matrix()
    if args.list_configs:
        for name in sorted(configs):
            print(name)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("CUDA device required for evaluation")

    cfg_filter = set(args.configs) if args.configs else None

    print(f"running eval on {args.data_root}, output -> {args.output}")
    print(f"configurations: {len(cfg_filter or configs)}")
    results = run_eval(
        data_root=args.data_root,
        configs=configs,
        output=args.output,
        limit_workloads=args.limit_workloads,
        device=device,
        config_filter=cfg_filter,
    )
    summary = summarize(results)
    results["summary"] = summary
    with args.output.open("w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\n==== Summary (sorted by cosine_mean, higher is better) ====\n")
    print_summary(summary)


if __name__ == "__main__":
    main()
