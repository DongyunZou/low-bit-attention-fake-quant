"""Evaluate Q/K Hadamard rotation with all smoothing tricks disabled.

This script is intentionally narrow: it compares baseline Q/K quantization with
Hadamard-rotated Q/K quantization while forcing

    smoothing="off", q_kmeans_k=None, v_smooth_mode="off", v_kmeans_k=None

so the reported gain belongs to the rotation, not smoothing or reorder tricks.
It reports both sampled QK-logit quantization error and full attention-output
error against raw BF16 SDPA.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import math
import time
from pathlib import Path
from typing import Iterable

import torch

from low_bit_fake_quant import (
    QuantConfig,
    fake_quant_attention,
    fast_hadamard_available,
    reference_attention,
)
from low_bit_fake_quant.attention import _dequant_qk, _qk_inputs_for_quant, _quant_qk


@dataclasses.dataclass(frozen=True)
class Workload:
    layer: str
    timestep: str
    path: Path


@dataclasses.dataclass
class Metrics:
    mse: float
    rmse: float
    cosine: float
    max_abs_err: float


def make_cfg(*, qk_quant: str, hadamard: bool, p_requant: bool) -> QuantConfig:
    return QuantConfig(
        qk_quant=qk_quant,
        v_quant="fp8_channel",
        smoothing="off",
        q_kmeans_k=None,
        qk_hadamard=hadamard,
        qk_hadamard_random_sign=True,
        qk_hadamard_seed=0,
        fp8_block_size=128,
        mxfp8_block_size=32,
        v_smooth_mode="off",
        v_kmeans_k=None,
        p_quant="elementwise",
        rowmax_mode="online",
        p_requant=p_requant,
        p_requant_block_m=64,
        p_requant_block_n=64,
    )


CONFIGS: dict[str, QuantConfig] = {
    "fp8_block_base": make_cfg(qk_quant="fp8_block", hadamard=False, p_requant=False),
    "fp8_block_hadamard": make_cfg(qk_quant="fp8_block", hadamard=True, p_requant=False),
    "mxfp8_base": make_cfg(qk_quant="mxfp8", hadamard=False, p_requant=False),
    "mxfp8_hadamard": make_cfg(qk_quant="mxfp8", hadamard=True, p_requant=False),
}


def iter_workloads(root: Path) -> Iterable[Workload]:
    for layer_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for ts_file in sorted(layer_dir.glob("timestep_*.pt")):
            yield Workload(layer=layer_dir.name, timestep=ts_file.stem, path=ts_file)


def load_qkv(path: Path, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    q = obj["query"].to(device=device, dtype=torch.bfloat16, non_blocking=True).contiguous()
    k = obj["key"].to(device=device, dtype=torch.bfloat16, non_blocking=True).contiguous()
    v = obj["value"].to(device=device, dtype=torch.bfloat16, non_blocking=True).contiguous()
    return q, k, v


def compute_metrics(out: torch.Tensor, ref: torch.Tensor) -> Metrics:
    a = out.float().flatten()
    b = ref.float().flatten()
    diff = a - b
    mse = float(diff.square().mean().item())
    rmse = math.sqrt(mse)
    eps = 1.0e-12
    cosine = float(
        torch.dot(a, b).item()
        / max(eps, math.sqrt(float(a.square().sum().item())))
        / max(eps, math.sqrt(float(b.square().sum().item())))
    )
    return Metrics(mse=mse, rmse=rmse, cosine=cosine, max_abs_err=float(diff.abs().max().item()))


def _score_sample(q: torch.Tensor, k: torch.Tensor, max_tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
    s = min(q.shape[1], max_tokens)
    # Keep fp8_block configs valid.
    s = max(128, (s // 128) * 128)
    return q[:, :s].contiguous(), k[:, :s].contiguous()


def qk_score_metrics(
    q: torch.Tensor,
    k: torch.Tensor,
    cfg: QuantConfig,
    *,
    max_tokens: int,
) -> Metrics:
    q_sample, k_sample = _score_sample(q, k, max_tokens)
    q_quant_in, k_quant_in = _qk_inputs_for_quant(q_sample, k_sample, cfg)
    q_fp8, q_scale, q_meta = _quant_qk(q_quant_in, cfg)
    k_fp8, k_scale, k_meta = _quant_qk(k_quant_in, cfg)
    q_deq = _dequant_qk(q_fp8, q_scale, q_meta, torch.bfloat16)
    k_deq = _dequant_qk(k_fp8, k_scale, k_meta, torch.bfloat16)

    q_ref = q_sample.permute(0, 2, 1, 3).float()
    k_ref = k_sample.permute(0, 2, 1, 3).float()
    q_quant = q_deq.permute(0, 2, 1, 3).float()
    k_quant = k_deq.permute(0, 2, 1, 3).float()
    ref_scores = torch.matmul(q_ref, k_ref.transpose(-2, -1))
    quant_scores = torch.matmul(q_quant, k_quant.transpose(-2, -1))
    return compute_metrics(quant_scores, ref_scores)


def summarize(results: dict) -> dict:
    summary: dict[str, dict[str, float]] = {}
    for name in results["configs"]:
        attn_mse: list[float] = []
        attn_cos: list[float] = []
        qk_mse: list[float] = []
        qk_cos: list[float] = []
        secs: list[float] = []
        for wl in results["workloads"]:
            rec = wl["configs"].get(name, {})
            if "error" in rec:
                continue
            attn_mse.append(rec["attention_metrics"]["mse"])
            attn_cos.append(rec["attention_metrics"]["cosine"])
            qk_mse.append(rec["qk_score_metrics"]["mse"])
            qk_cos.append(rec["qk_score_metrics"]["cosine"])
            secs.append(rec["seconds"])
        summary[name] = {
            "count": len(attn_mse),
            "attention_mse_mean": sum(attn_mse) / len(attn_mse) if attn_mse else float("nan"),
            "attention_cosine_mean": sum(attn_cos) / len(attn_cos) if attn_cos else float("nan"),
            "qk_score_mse_mean": sum(qk_mse) / len(qk_mse) if qk_mse else float("nan"),
            "qk_score_cosine_mean": sum(qk_cos) / len(qk_cos) if qk_cos else float("nan"),
            "seconds_mean": sum(secs) / len(secs) if secs else float("nan"),
        }

    comparisons: dict[str, dict[str, float]] = {}
    for qk_quant in ("fp8_block", "mxfp8"):
        base = summary[f"{qk_quant}_base"]
        had = summary[f"{qk_quant}_hadamard"]
        comparisons[qk_quant] = {
            "qk_score_mse_rel_delta": (
                (had["qk_score_mse_mean"] - base["qk_score_mse_mean"])
                / base["qk_score_mse_mean"]
            ),
            "attention_mse_rel_delta": (
                (had["attention_mse_mean"] - base["attention_mse_mean"])
                / base["attention_mse_mean"]
            ),
            "qk_score_cosine_delta": had["qk_score_cosine_mean"] - base["qk_score_cosine_mean"],
            "attention_cosine_delta": had["attention_cosine_mean"] - base["attention_cosine_mean"],
        }
    summary["comparisons"] = comparisons
    return summary


def run(args: argparse.Namespace) -> dict:
    device = torch.device("cuda")
    workloads = list(iter_workloads(args.data_root))
    if args.limit_workloads is not None:
        workloads = workloads[: args.limit_workloads]
    if not workloads:
        raise RuntimeError(f"no timestep_*.pt files under {args.data_root}")
    configs = {k: v for k, v in CONFIGS.items() if not args.configs or k in set(args.configs)}
    results = {
        "dataset": str(args.data_root),
        "torch_version": torch.__version__,
        "device": torch.cuda.get_device_name(device),
        "fast_hadamard_available": fast_hadamard_available(),
        "max_score_tokens": args.max_score_tokens,
        "smooth_controls": {
            "smoothing": "off",
            "q_kmeans_k": None,
            "v_smooth_mode": "off",
            "v_kmeans_k": None,
        },
        "configs": {name: dataclasses.asdict(cfg) for name, cfg in configs.items()},
        "workloads": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for i, wl in enumerate(workloads, start=1):
        q, k, v = load_qkv(wl.path, device)
        torch.cuda.synchronize()
        t0 = time.time()
        ref = reference_attention(q, k, v)
        torch.cuda.synchronize()
        ref_s = time.time() - t0
        rec = {
            "layer": wl.layer,
            "timestep": wl.timestep,
            "path": str(wl.path),
            "shape": list(q.shape),
            "reference_seconds": ref_s,
            "configs": {},
        }
        print(f"[{i}/{len(workloads)}] {wl.layer}/{wl.timestep} ref={ref_s:.2f}s shape={tuple(q.shape)}")
        for name, cfg in configs.items():
            torch.cuda.synchronize()
            t0 = time.time()
            try:
                qk_metrics = qk_score_metrics(q, k, cfg, max_tokens=args.max_score_tokens)
                out = fake_quant_attention(q, k, v, cfg)
                torch.cuda.synchronize()
                seconds = time.time() - t0
                attn_metrics = compute_metrics(out, ref)
                rec["configs"][name] = {
                    "qk_score_metrics": dataclasses.asdict(qk_metrics),
                    "attention_metrics": dataclasses.asdict(attn_metrics),
                    "seconds": seconds,
                }
                print(
                    f"  {name:<20} QK_MSE={qk_metrics.mse:.6e} "
                    f"ATTN_MSE={attn_metrics.mse:.6e} "
                    f"ATTN_Cos={attn_metrics.cosine:.9f} seconds={seconds:.2f}"
                )
                del out
            except Exception as exc:  # noqa: BLE001
                torch.cuda.synchronize()
                rec["configs"][name] = {"error": repr(exc)}
                print(f"  {name:<20} ERROR {exc!r}")
            torch.cuda.empty_cache()
        results["workloads"].append(rec)
        results["summary"] = summarize(results)
        args.output.write_text(json.dumps(results, indent=2, default=str))
        del q, k, v, ref
        gc.collect()
        torch.cuda.empty_cache()
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/home/dongyunz/dataset/v-dit/wan21_p1"),
    )
    parser.add_argument("--output", type=Path, default=Path("bench/results_hadamard_qk.json"))
    parser.add_argument("--limit-workloads", type=int, default=None)
    parser.add_argument("--max-score-tokens", type=int, default=1024)
    parser.add_argument("--configs", nargs="*", default=None, choices=sorted(CONFIGS))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
