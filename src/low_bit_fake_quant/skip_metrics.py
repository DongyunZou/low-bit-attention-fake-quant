"""Accuracy metrics for the FP8 block-skip attention study.

Metrics are computed in FP32. Cosine is reported two ways: as a single global
flattened value (overall alignment) and as a distribution over query rows (each
``(query, head)`` D-vector), so a per-row collapse cannot hide behind a healthy
global mean. Aggregation helpers stratify per layer and per timestep with
mean / median / p95 / worst-case, never a single global mean alone.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

import torch


@dataclass
class OutputMetrics:
    """Error of one output tensor vs the ground-truth output."""

    mse: float
    rmse: float
    rel_rmse: float          # ||pred - ref|| / ||ref||
    cosine_global: float     # cosine of the flattened tensors
    cosine_row_mean: float   # mean over per-(query, head) cosines
    cosine_row_median: float
    cosine_row_p05: float    # 5th percentile (worst rows)
    cosine_row_min: float
    max_abs_err: float

    def asdict(self) -> dict:
        return dataclasses.asdict(self)


def compute_output_metrics(pred: torch.Tensor, ref: torch.Tensor) -> OutputMetrics:
    """Compute error metrics for ``pred`` against ``ref`` (any matching shape).

    The last dim is treated as the per-row feature vector for the per-row cosine
    distribution; all leading dims are flattened into the row index.
    """
    a = pred.detach().float()
    b = ref.detach().float()
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}")
    af = a.reshape(-1)
    bf = b.reshape(-1)
    diff = af - bf
    mse = float(diff.pow(2).mean())
    rmse = math.sqrt(mse)
    eps = 1e-12
    ref_norm = float(bf.norm())
    rel_rmse = float(diff.norm() / max(eps, ref_norm))
    cos_global = float(
        torch.dot(af, bf) / max(eps, af.norm().item()) / max(eps, bf.norm().item())
    )
    # per-row cosine over the feature (last) dim
    d = a.shape[-1]
    ar = a.reshape(-1, d)
    br = b.reshape(-1, d)
    row_cos = torch.nn.functional.cosine_similarity(ar, br, dim=-1, eps=eps)
    rc = row_cos.float()
    return OutputMetrics(
        mse=mse,
        rmse=rmse,
        rel_rmse=rel_rmse,
        cosine_global=cos_global,
        cosine_row_mean=float(rc.mean()),
        cosine_row_median=float(rc.median()),
        cosine_row_p05=float(torch.quantile(rc, 0.05)),
        cosine_row_min=float(rc.min()),
        max_abs_err=float(diff.abs().max()),
    )


def _agg(values: list[float]) -> dict:
    if not values:
        return {}
    t = torch.tensor(values, dtype=torch.float64)
    return {
        "mean": float(t.mean()),
        "median": float(t.median()),
        "p95": float(torch.quantile(t, 0.95)),
        "worst_max": float(t.max()),
        "worst_min": float(t.min()),
        "count": len(values),
    }


def aggregate(records: list[dict], metric_keys, group_keys=("layer", "timestep")) -> dict:
    """Aggregate flat metric records globally and stratified by group keys.

    ``records`` is a list of dicts each carrying the ``group_keys`` plus the
    ``metric_keys`` numeric values. Returns ``{"global": {...}, "by_layer":
    {...}, "by_timestep": {...}}`` where each leaf is the mean/median/p95/worst
    aggregation of one metric.
    """
    def collect(subset):
        return {mk: _agg([r[mk] for r in subset if mk in r and r[mk] is not None])
                for mk in metric_keys}

    out = {"global": collect(records)}
    for gk in group_keys:
        groups: dict[str, list[dict]] = {}
        for r in records:
            groups.setdefault(str(r.get(gk)), []).append(r)
        out[f"by_{gk}"] = {g: collect(sub) for g, sub in sorted(groups.items())}
    return out
