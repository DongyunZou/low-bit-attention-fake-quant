"""Verify estimated-rowmax dynamic P against online-rowmax dynamic P.

The script accepts either dumped Wan hook calls with keys
``query/key/value`` or v-dit workload files understood by ``bench.eval_wan21``.
It reports output metrics for ``rowmax_mode=qm_k`` relative to
``rowmax_mode=online`` using the same Q/K/V quantization and smoothing config.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from bench.eval_wan21 import compute_metrics, load_qkv  # noqa: E402
from low_bit_fake_quant import QuantConfig, fake_quant_attention  # noqa: E402
from low_bit_fake_quant.attention import prepare_for_attention  # noqa: E402


def _load(path: Path, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and {"query", "key", "value"}.issubset(obj):
        return (
            obj["query"].to(device=device, dtype=torch.bfloat16),
            obj["key"].to(device=device, dtype=torch.bfloat16),
            obj["value"].to(device=device, dtype=torch.bfloat16),
        )
    return load_qkv(path, device=device, dtype=torch.bfloat16)


def _pad_to_block(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    pad = (block - q.shape[1] % block) % block
    if pad == 0:
        return q, k, v, 0
    q_pad = q.new_zeros(q.shape[0], pad, q.shape[2], q.shape[3])
    k_pad = k[:, -1:, :, :].expand(q.shape[0], pad, k.shape[2], k.shape[3])
    v_pad = v[:, -1:, :, :].expand(q.shape[0], pad, v.shape[2], v.shape[3])
    return torch.cat([q, q_pad], dim=1), torch.cat([k, k_pad], dim=1), torch.cat([v, v_pad], dim=1), pad


def _cfg(args: argparse.Namespace, rowmax_mode: str) -> QuantConfig:
    return QuantConfig(
        qk_quant=args.qk_quant,
        v_quant=args.v_quant,
        smoothing="full",
        q_kmeans_k=args.q_kmeans_k,
        q_smooth_block_size=args.block,
        fp8_block_size=args.block,
        v_smooth_mode=args.v_smooth_mode,
        v_smooth_block_size=args.v_smooth_block_size,
        v_kmeans_k=args.v_kmeans_k if args.v_kmeans_k > 0 else None,
        p_quant="dynamic",
        rowmax_mode=rowmax_mode,
        p_requant=True,
        p_requant_block_m=args.p_block_m,
        p_requant_block_n=args.p_block_n,
    )


def verify_one(path: Path, args: argparse.Namespace, device: torch.device) -> dict:
    q, k, v = _load(path, device)
    q_in, k_in, v_in, pad = _pad_to_block(q, k, v, math.lcm(args.block, args.p_block_m, args.p_block_n))
    sm_scale = 1.0 / math.sqrt(q.shape[-1])
    online_cfg = _cfg(args, "online")
    estimated_cfg = _cfg(args, "qm_k")
    cache = prepare_for_attention(q_in, k_in, v_in, online_cfg)
    online = fake_quant_attention(
        q_in, k_in, v_in, online_cfg, sm_scale=sm_scale, preprocess_cache=cache
    )[:, : q.shape[1]]
    estimated = fake_quant_attention(
        q_in, k_in, v_in, estimated_cfg, sm_scale=sm_scale, preprocess_cache=cache
    )[:, : q.shape[1]]
    m = compute_metrics(estimated, online).asdict()
    return {
        "path": str(path),
        "shape": list(q.shape),
        "pad": pad,
        "metrics_qm_k_vs_online": m,
        "estimated_has_nan": bool(torch.isnan(estimated).any().item()),
        "estimated_has_inf": bool(torch.isinf(estimated).any().item()),
        "online_has_nan": bool(torch.isnan(online).any().item()),
        "online_has_inf": bool(torch.isinf(online).any().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--block", type=int, default=256)
    parser.add_argument("--p-block-m", type=int, default=64)
    parser.add_argument("--p-block-n", type=int, default=64)
    parser.add_argument("--q-kmeans-k", type=int, default=32)
    parser.add_argument("--qk-quant", default="fp8_block", choices=["fp8_block", "mxfp8"])
    parser.add_argument("--v-quant", default="fp8_channel", choices=["fp8_channel", "fp8_block", "mxfp8"])
    parser.add_argument("--v-smooth-mode", default="per_block", choices=["off", "per_block"])
    parser.add_argument("--v-smooth-block-size", type=int, default=64)
    parser.add_argument("--v-kmeans-k", type=int, default=64)
    args = parser.parse_args()

    device = torch.device(args.device)
    reports = [verify_one(path, args, device) for path in args.paths]
    summary = {
        "num_workloads": len(reports),
        "max_mse": max(r["metrics_qm_k_vs_online"]["mse"] for r in reports),
        "min_cosine": min(r["metrics_qm_k_vs_online"]["cosine"] for r in reports),
        "max_rel_l2": max(r["metrics_qm_k_vs_online"]["rel_l2"] for r in reports),
        "max_abs_err": max(r["metrics_qm_k_vs_online"]["max_abs_err"] for r in reports),
        "bad_outputs": sum(
            r["estimated_has_nan"] or r["estimated_has_inf"] or r["online_has_nan"] or r["online_has_inf"]
            for r in reports
        ),
    }
    out = {"summary": summary, "reports": reports}
    print(json.dumps(out, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
