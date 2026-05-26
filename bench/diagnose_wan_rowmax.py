"""Diagnose dynamic-P fixed-rowmax behavior on a Wan QKV workload."""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from bench.eval_wan21 import compute_metrics, load_qkv
from low_bit_fake_quant import QuantConfig, fake_quant_attention, reference_attention
from low_bit_fake_quant.attention import (
    _dequant_qk,
    _estimate_rowmax_from_qm_k,
    _preprocess,
    _quant_qk,
)


def _cfg(*, block: int, rowmax_mode: str, p_quant: str) -> QuantConfig:
    return QuantConfig(
        qk_quant="fp8_block",
        v_quant="fp8_channel",
        smoothing="full",
        q_kmeans_k=32,
        q_smooth_block_size=block,
        fp8_block_size=block,
        v_smooth_mode="per_block",
        v_smooth_block_size=64,
        v_kmeans_k=64,
        p_requant=True,
        p_requant_block_m=64,
        p_requant_block_n=64,
        p_quant=p_quant,
        rowmax_mode=rowmax_mode,
    )


def _sample_rowmax_error(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cfg: QuantConfig,
    *,
    heads: int,
    groups: int,
    rows_per_group: int,
    chunk_n: int,
) -> dict:
    sm_scale = 1.0 / math.sqrt(q.shape[-1])
    q_work, k_work, _v_work, qm, _v_alpha, _q_kmeans, _v_kmeans = _preprocess(q, k, v, cfg)
    if qm is None:
        raise RuntimeError("expected full smoothing to produce qm")

    q_fp8, q_scale, q_meta = _quant_qk(q_work, cfg)
    k_fp8, k_scale, k_meta = _quant_qk(k_work, cfg)
    q_deq = _dequant_qk(q_fp8, q_scale, q_meta, torch.bfloat16)
    k_deq = _dequant_qk(k_fp8, k_scale, k_meta, torch.bfloat16)
    rowmax_est = _estimate_rowmax_from_qm_k(
        qm, k_work, block_q=cfg.q_smooth_block_size, sm_scale=sm_scale, chunk_n=chunk_n
    )

    b, s, h, d = q.shape
    block = cfg.q_smooth_block_size
    n_groups = s // block
    head_ids = torch.linspace(0, h - 1, min(heads, h), device=q.device).round().long().unique()
    group_ids = torch.linspace(0, n_groups - 1, min(groups, n_groups), device=q.device).round().long().unique()
    row_offsets = torch.linspace(0, block - 1, min(rows_per_group, block), device=q.device).round().long().unique()

    errors = []
    true_vals = []
    est_vals = []
    over_80 = 0
    over_20 = 0
    under_neg20 = 0
    checked_rows = 0
    for hi in head_ids.tolist():
        k_h = k_deq[0, :, hi, :].float()
        ks_h = k_work[0, :, hi, :].float()
        for gi in group_ids.tolist():
            corr = (qm[0, gi, hi, :].float() @ ks_h.T) * sm_scale
            for off in row_offsets.tolist():
                row = gi * block + off
                q_row = q_deq[0, row, hi, :].float()
                max_score = -float("inf")
                for n0 in range(0, s, chunk_n):
                    n1 = min(n0 + chunk_n, s)
                    scores = (q_row @ k_h[n0:n1].T) * sm_scale + corr[n0:n1]
                    max_score = max(max_score, float(scores.max().item()))
                est = float(rowmax_est[0, hi, row].item())
                err = max_score - est
                errors.append(err)
                true_vals.append(max_score)
                est_vals.append(est)
                over_80 += int(err > 80.0)
                over_20 += int(err > 20.0)
                under_neg20 += int(err < -20.0)
                checked_rows += 1

    e = torch.tensor(errors)
    return {
        "sampled_rows": checked_rows,
        "heads": head_ids.cpu().tolist(),
        "groups": group_ids.cpu().tolist(),
        "rows_per_group": row_offsets.cpu().tolist(),
        "true_rowmax": {
            "min": min(true_vals),
            "max": max(true_vals),
        },
        "estimated_rowmax": {
            "min": min(est_vals),
            "max": max(est_vals),
        },
        "true_minus_est": {
            "min": float(e.min().item()),
            "p50": float(e.quantile(0.50).item()),
            "p90": float(e.quantile(0.90).item()),
            "p99": float(e.quantile(0.99).item()),
            "max": float(e.max().item()),
        },
        "count_true_minus_est_gt_20": over_20,
        "count_true_minus_est_gt_80": over_80,
        "count_true_minus_est_lt_neg20": under_neg20,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--block", type=int, default=256)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--groups", type=int, default=8)
    parser.add_argument("--rows-per-group", type=int, default=4)
    parser.add_argument("--chunk-n", type=int, default=4096)
    parser.add_argument("--skip-output-metrics", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    q, k, v = load_qkv(args.workload, device=device, dtype=torch.bfloat16)
    report = {
        "workload": str(args.workload),
        "shape": list(q.shape),
        "block": args.block,
        "rowmax_error_sample": _sample_rowmax_error(
            q,
            k,
            v,
            _cfg(block=args.block, rowmax_mode="qm_k", p_quant="dynamic"),
            heads=args.heads,
            groups=args.groups,
            rows_per_group=args.rows_per_group,
            chunk_n=args.chunk_n,
        ),
    }

    if not args.skip_output_metrics:
        ref = reference_attention(q, k, v)
        outputs = {}
        for name, cfg in {
            "static_online": _cfg(block=args.block, rowmax_mode="online", p_quant="elementwise"),
            "dynamic_online": _cfg(block=args.block, rowmax_mode="online", p_quant="dynamic"),
            "dynamic_qm_k": _cfg(block=args.block, rowmax_mode="qm_k", p_quant="dynamic"),
        }.items():
            out = fake_quant_attention(q, k, v, cfg)
            outputs[name] = compute_metrics(out, ref).asdict()
            del out
            torch.cuda.empty_cache()
        report["output_metrics_vs_sdpa"] = outputs

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
