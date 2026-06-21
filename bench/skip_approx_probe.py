"""Probe softmax-free approximations for BLASST-skipped blocks on Wan workloads.

This is an exploratory accuracy harness for three ideas:

1. When a BLASST skip condition fires, do not drop the block to zero. Instead
   fill the skipped rows with an approximate uniform P row and add
   ``P_approx @ V`` without computing a full softmax for that block.
2. Use a fine-grained row-level skip mask: compute exact softmax/PV only for
   rows in a 128x128 tile that fail the skip condition; for skipped rows either
   fill with zero or an approximate uniform P row.
3. Use fixed-bin UTA-style pseudo-tokens for skipped blocks. A fill mode such
   as ``uta16_a1.5`` splits each 128-key block into 16 contiguous bins, estimates
   one mass per bin, and adds both the approximate ``P @ V`` contribution and
   the skipped mass back to the softmax denominator.

The harness intentionally reuses the same FP8 Q/K/V + static P*256 numerics as
``blasst_skip.py``. It computes full scores in PyTorch so every approximation
can be compared against the no-skip FP8 baseline and SDPA; this is a measurement
tool, not a deployable kernel.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import re
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from low_bit_fake_quant.blasst_skip import (  # noqa: E402
    DENOM_FLOOR,
    KEY_BLOCK,
    QUERY_TILE,
    fake_quant_per_head,
    sdpa_ground_truth,
    static_p_quant,
)


DEFAULT_LAMBDAS = [0.01, 0.03, 0.1, 0.2, 0.3]


@dataclasses.dataclass
class StreamingMetrics:
    """Streaming output metrics with per-row cosine retained on CPU."""

    collect_row_cos: bool = True
    sqerr: float = 0.0
    count: int = 0
    dot: float = 0.0
    pred_norm2: float = 0.0
    ref_norm2: float = 0.0
    max_abs: float = 0.0
    row_cos: list[torch.Tensor] = dataclasses.field(default_factory=list)

    def update(self, pred: torch.Tensor, ref: torch.Tensor) -> None:
        p = pred.detach().float()
        r = ref.detach().float()
        diff = p - r
        self.sqerr += float(diff.pow(2).sum())
        self.count += diff.numel()
        self.dot += float((p * r).sum())
        self.pred_norm2 += float(p.pow(2).sum())
        self.ref_norm2 += float(r.pow(2).sum())
        self.max_abs = max(self.max_abs, float(diff.abs().max()))
        if not self.collect_row_cos:
            return
        pc = p.reshape(-1, p.shape[-1])
        rc = r.reshape(-1, r.shape[-1])
        cos = torch.nn.functional.cosine_similarity(pc, rc, dim=-1, eps=1e-12)
        self.row_cos.append(cos.float().cpu())

    def asdict(self) -> dict:
        mse = self.sqerr / max(1, self.count)
        rmse = math.sqrt(mse)
        ref_norm = math.sqrt(max(self.ref_norm2, 1e-30))
        pred_norm = math.sqrt(max(self.pred_norm2, 1e-30))
        cos_global = self.dot / max(1e-30, pred_norm * ref_norm)
        rows = torch.cat(self.row_cos).float() if self.row_cos else torch.empty(0)
        if rows.numel():
            row_sorted = rows.sort().values
            n = row_sorted.numel()
            row_p05 = float(row_sorted[min(n - 1, int(0.05 * (n - 1)))])
            row_min = float(row_sorted[0])
            row_median = float(row_sorted[(n - 1) // 2])
            row_mean = float(rows.mean())
        else:
            row_p05 = row_min = row_median = row_mean = float("nan")
        return {
            "mse": mse,
            "rmse": rmse,
            "rel_rmse": math.sqrt(self.sqerr) / max(1e-30, ref_norm),
            "cosine_global": cos_global,
            "cosine_row_mean": row_mean,
            "cosine_row_median": row_median,
            "cosine_row_p05": row_p05,
            "cosine_row_min": row_min,
            "max_abs_err": self.max_abs,
        }


@dataclasses.dataclass(frozen=True)
class Config:
    scope: str     # "tile", "row", or "group<N>"
    fill: str      # "zero", "oracle_pq", "uta4_a1.25", "mean_a*", ...
    lam: float

    @property
    def name(self) -> str:
        return f"{self.scope}:{self.fill}:lam{self.lam:g}"


@dataclasses.dataclass
class SkipStats:
    skipped_rows: int = 0
    total_rows: int = 0
    skipped_tiles: int = 0
    full_tiles: int = 0
    partial_tiles: int = 0
    total_tiles: int = 0

    def update_tile(self, skip_tile: torch.Tensor, rows_per_tile: int) -> None:
        skipped = int(skip_tile.sum())
        total = skip_tile.numel()
        self.skipped_tiles += skipped
        self.full_tiles += skipped
        self.total_tiles += total
        self.skipped_rows += skipped * rows_per_tile
        self.total_rows += total * rows_per_tile

    def update_row(self, skip_row: torch.Tensor) -> None:
        # skip_row: (H, m, nb), one row-block decision per query row.
        skipped = int(skip_row.sum())
        total = skip_row.numel()
        per_tile = skip_row
        full = per_tile.all(dim=1)
        any_ = per_tile.any(dim=1)
        self.skipped_rows += skipped
        self.total_rows += total
        self.full_tiles += int(full.sum())
        self.partial_tiles += int((any_ & ~full).sum())
        self.skipped_tiles += int(full.sum())
        self.total_tiles += full.numel()

    def update_group(self, skip_group: torch.Tensor, group_size: int) -> None:
        # skip_group: (H, n_groups, nb), one decision per contiguous query-row group.
        skipped = int(skip_group.sum())
        total = skip_group.numel()
        full = skip_group.all(dim=1)
        any_ = skip_group.any(dim=1)
        self.skipped_rows += skipped * group_size
        self.total_rows += total * group_size
        self.full_tiles += int(full.sum())
        self.partial_tiles += int((any_ & ~full).sum())
        self.skipped_tiles += int(full.sum())
        self.total_tiles += full.numel()

    def asdict(self) -> dict:
        return {
            "row_skip_rate": self.skipped_rows / max(1, self.total_rows),
            "full_tile_skip_rate": self.full_tiles / max(1, self.total_tiles),
            "partial_tile_rate": self.partial_tiles / max(1, self.total_tiles),
            "tile_skip_rate": self.skipped_tiles / max(1, self.total_tiles),
        }


def parse_workload(path: Path) -> dict:
    m = re.search(r"(wan21_p\d+)/layer_(\d+)/timestep_(\d+)\.pt$", str(path))
    if not m:
        return {"part": "", "layer": "", "timestep": ""}
    return {"part": m.group(1), "layer": f"layer_{m.group(2)}", "timestep": f"timestep_{m.group(3)}"}


def iter_workloads(data_root: Path) -> list[Path]:
    return sorted(data_root.glob("layer_*/timestep_*.pt"))


def configs(lambdas: list[float], fill_modes: list[str], group_sizes: list[int]) -> list[Config]:
    out = []
    for lam in lambdas:
        out.append(Config("tile", "zero", lam))
        out.append(Config("row", "zero", lam))
        for group_size in group_sizes:
            out.append(Config(f"group{group_size}", "zero", lam))
        for fill in fill_modes:
            out.append(Config("tile", fill, lam))
            out.append(Config("row", fill, lam))
            for group_size in group_sizes:
                out.append(Config(f"group{group_size}", fill, lam))
    return out


def load_qkv(path: Path, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    return (
        data["query"].to(torch.bfloat16)[0].to(device),
        data["key"].to(torch.bfloat16)[0].to(device),
        data["value"].to(torch.bfloat16)[0].to(device),
    )


def normalize(acc: torch.Tensor, denom: torch.Tensor) -> torch.Tensor:
    return (acc / denom.clamp_min(DENOM_FLOOR)[..., None]).permute(1, 0, 2).contiguous()


def uniform_pv(
    mass: torch.Tensor,
    v_sum: torch.Tensor,
    *,
    quantize_p: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(pv, denom_mass)`` for a uniform P row over a key block.

    ``mass`` is the estimated unquantized row/block mass in the final-rowmax
    frame, shape ``(H, m, nb)``. Numerator uses static-P quantization when
    ``quantize_p`` is true, matching the FP8 P path.
    """
    p_mean = mass / KEY_BLOCK
    p_num = static_p_quant(p_mean) if quantize_p else p_mean
    pv = p_num[..., None] * v_sum[:, None, :, :]
    return pv.float(), mass.float()


def parse_uta_mode(mode: str) -> tuple[int, float] | None:
    """Parse multi-bin UTA modes such as ``uta4`` or ``uta8_a1.25``."""
    m = re.fullmatch(r"uta(\d+)(?:_a([0-9]+(?:\.[0-9]+)?))?", mode)
    if not m:
        return None
    bins = int(m.group(1))
    alpha = 1.0 if m.group(2) is None else float(m.group(2))
    return bins, alpha


def multi_bin_uta_pv(
    mode: str,
    *,
    rowmax: torch.Tensor,
    sb: torch.Tensor,
    v_sum_bins: dict[int, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(pv, denom_mass)`` for fixed-bin UTA skipped-block fill.

    Each 128-key block is split into ``N`` contiguous bins. A skipped bin is
    represented by one uniform pseudo-token with mass
    ``alpha * bin_size * exp(mean(score_bin) - rowmax)``, capped by the local
    max bound for that bin.
    """
    parsed = parse_uta_mode(mode)
    if parsed is None:
        raise ValueError(f"unknown UTA fill mode {mode}")
    bins, alpha = parsed
    if KEY_BLOCK % bins:
        raise ValueError(f"UTA bin count {bins} must divide key block {KEY_BLOCK}")
    bin_size = KEY_BLOCK // bins

    sbb = sb.view(*sb.shape[:-1], bins, bin_size)
    mean = sbb.mean(dim=-1)
    max_ = sbb.amax(dim=-1)
    max_bound = bin_size * torch.exp(max_ - rowmax[..., None, None])
    mass = (alpha * bin_size * torch.exp(mean - rowmax[..., None, None])).clamp_max(max_bound)
    p_num = static_p_quant(mass / bin_size)
    pv = torch.einsum("hmjb,hjbd->hmjd", p_num.float(), v_sum_bins[bins])
    return pv.float(), mass.sum(dim=-1).float()


def mass_estimate(
    mode: str,
    *,
    block_mass: torch.Tensor,
    block_max: torch.Tensor,
    running: torch.Tensor,
    rowmax: torch.Tensor,
    sb: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """Softmax-free row/block mass estimates plus oracle upper bounds."""
    if mode == "oracle_pq" or mode == "oracle_unq":
        return block_mass

    # Upper bound from each block's local max: sum exp(score-M) <= K exp(max-M).
    max_bound = KEY_BLOCK * torch.exp(block_max - rowmax[..., None])

    if mode.startswith("max_a"):
        alpha = float(mode[len("max_a"):])
        return max_bound * alpha

    if mode == "mean" or mode.startswith("mean_a"):
        alpha = 1.0 if mode == "mean" else float(mode[len("mean_a"):])
        mean = sb.mean(dim=-1)
        return (alpha * KEY_BLOCK * torch.exp(mean - rowmax[..., None])).clamp_max(max_bound)

    if mode == "logn":
        mean = sb.mean(dim=-1)
        var = sb.var(dim=-1, unbiased=False)
        return (KEY_BLOCK * torch.exp(mean + 0.5 * var - rowmax[..., None])).clamp_max(max_bound)

    if mode == "sample8" or mode.startswith("sample8_a"):
        alpha = 1.0 if mode == "sample8" else float(mode[len("sample8_a"):])
        sample = sb[..., :: max(1, KEY_BLOCK // 8)]
        return (
            alpha * KEY_BLOCK * torch.exp(sample - rowmax[..., None, None]).mean(dim=-1)
        ).clamp_max(max_bound)

    if mode.startswith("thr_a"):
        alpha = float(mode[len("thr_a"):])
        # Uses only the running max and lambda threshold, not the block score distribution.
        return KEY_BLOCK * lam * torch.exp(running - rowmax[..., None]) * alpha

    raise ValueError(f"unknown fill mode {mode}")


def run_workload(
    path: Path,
    device: torch.device,
    *,
    cfgs: list[Config],
    matmul_dtype: torch.dtype,
    collect_row_cos: bool,
) -> dict:
    t0 = time.time()
    q, k, v = load_qkv(path, device)
    ref = sdpa_ground_truth(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0))[0]

    qf, _, _ = fake_quant_per_head(q.float())
    kf, _, _ = fake_quant_per_head(k.float())
    vf, _, _ = fake_quant_per_head(v.float())
    S, H, D = qf.shape
    if S % QUERY_TILE:
        raise ValueError(f"seqlen {S} is not divisible by {QUERY_TILE}")
    nb = S // KEY_BLOCK
    sm = 1.0 / math.sqrt(D)

    qh = qf.permute(1, 0, 2).contiguous().to(matmul_dtype)
    kt = kf.permute(1, 0, 2).transpose(1, 2).contiguous().to(matmul_dtype)
    vh = vf.permute(1, 0, 2).contiguous().to(matmul_dtype)
    vb = vh.view(H, nb, KEY_BLOCK, D)
    v_sum = vb.float().sum(dim=2)  # (H, nb, D), reusable for uniform fill.
    uta_bins = sorted({parse_uta_mode(cfg.fill)[0] for cfg in cfgs if parse_uta_mode(cfg.fill)})
    v_sum_bins = {}
    for bins in uta_bins:
        if KEY_BLOCK % bins:
            raise ValueError(f"UTA bin count {bins} must divide key block {KEY_BLOCK}")
        bin_size = KEY_BLOCK // bins
        v_sum_bins[bins] = vb.view(H, nb, bins, bin_size, D).float().sum(dim=3)

    metrics_sdpa = {cfg.name: StreamingMetrics(collect_row_cos=collect_row_cos) for cfg in cfgs}
    metrics_fp8 = {cfg.name: StreamingMetrics(collect_row_cos=collect_row_cos) for cfg in cfgs}
    stats = {cfg.name: SkipStats() for cfg in cfgs}
    baseline = StreamingMetrics(collect_row_cos=collect_row_cos)

    torch.cuda.reset_peak_memory_stats(device)
    total_tiles = 0

    for ti in range(nb):
        rows = slice(ti * QUERY_TILE, (ti + 1) * QUERY_TILE)
        ref_tile = ref[rows]
        scores = torch.matmul(qh[:, rows, :].float(), kt.float()) * sm  # (H,m,S)
        m = scores.shape[1]
        sb = scores.view(H, m, nb, KEY_BLOCK)
        block_max = sb.amax(dim=-1)
        rowmax = block_max.amax(dim=-1)
        running = torch.cummax(block_max, dim=-1).values
        online = torch.exp(sb - rowmax[..., None, None])
        block_mass = online.sum(dim=-1)  # (H,m,nb), unquantized denominator pieces.

        p_q = static_p_quant(online)
        pv = torch.einsum("hmjk,hjkd->hmjd", p_q.to(matmul_dtype), vb).float()
        no_skip = normalize(pv.sum(dim=2), block_mass.sum(dim=-1))
        baseline.update(no_skip, ref_tile)

        # Shared skip predicates.
        margin = block_max - running
        tile_margin = margin.amax(dim=1)

        approx_cache: dict[tuple[str, float], tuple[torch.Tensor, torch.Tensor]] = {}

        def get_approx(fill: str, lam: float) -> tuple[torch.Tensor, torch.Tensor]:
            key = (fill, lam)
            if key not in approx_cache:
                if parse_uta_mode(fill):
                    approx_cache[key] = multi_bin_uta_pv(fill, rowmax=rowmax, sb=sb, v_sum_bins=v_sum_bins)
                else:
                    mass = mass_estimate(
                        fill, block_mass=block_mass, block_max=block_max,
                        running=running, rowmax=rowmax, sb=sb, lam=lam,
                    )
                    quantize = fill != "oracle_unq"
                    approx_cache[key] = uniform_pv(mass, v_sum, quantize_p=quantize)
            return approx_cache[key]

        for cfg in cfgs:
            logt = -math.inf if cfg.lam <= 0 else math.log(cfg.lam)
            if cfg.scope == "tile":
                skip_tile = tile_margin < logt                         # (H,nb)
                keep_tile = (~skip_tile).float()
                acc = torch.einsum("hmjd,hj->hmd", pv, keep_tile)
                denom = torch.einsum("hmj,hj->hm", block_mass, keep_tile)
                stats[cfg.name].update_tile(skip_tile, m)
                if cfg.fill != "zero":
                    apv, aden = get_approx(cfg.fill, cfg.lam)
                    skipf = skip_tile[:, None, :].float()
                    acc = acc + (apv * skipf[..., None]).sum(dim=2)
                    denom = denom + (aden * skipf).sum(dim=2)
            elif cfg.scope == "row":
                skip_row = margin < logt                               # (H,m,nb)
                keep_row = (~skip_row).float()
                acc = (pv * keep_row[..., None]).sum(dim=2)
                denom = (block_mass * keep_row).sum(dim=-1)
                stats[cfg.name].update_row(skip_row)
                if cfg.fill != "zero":
                    apv, aden = get_approx(cfg.fill, cfg.lam)
                    skipf = skip_row.float()
                    acc = acc + (apv * skipf[..., None]).sum(dim=2)
                    denom = denom + (aden * skipf).sum(dim=-1)
            elif cfg.scope.startswith("group"):
                group_size = int(cfg.scope[len("group"):])
                if m % group_size:
                    raise ValueError(f"group size {group_size} must divide query tile {m}")
                margin_group = margin.view(H, m // group_size, group_size, nb).amax(dim=2)
                skip_group = margin_group < logt                         # (H,ng,nb)
                skip_row = skip_group.repeat_interleave(group_size, dim=1)
                keep_row = (~skip_row).float()
                acc = (pv * keep_row[..., None]).sum(dim=2)
                denom = (block_mass * keep_row).sum(dim=-1)
                stats[cfg.name].update_group(skip_group, group_size)
                if cfg.fill != "zero":
                    apv, aden = get_approx(cfg.fill, cfg.lam)
                    skipf = skip_row.float()
                    acc = acc + (apv * skipf[..., None]).sum(dim=2)
                    denom = denom + (aden * skipf).sum(dim=-1)
            else:
                raise ValueError(cfg.scope)

            pred = normalize(acc, denom)
            metrics_sdpa[cfg.name].update(pred, ref_tile)
            metrics_fp8[cfg.name].update(pred, no_skip)

        total_tiles += H * nb
        del scores, sb, block_max, rowmax, running, online, block_mass, p_q, pv
        del no_skip, margin, tile_margin, approx_cache

    peak = int(torch.cuda.max_memory_allocated(device))
    results = []
    for cfg in cfgs:
        rec = {
            "config": cfg.name,
            "scope": cfg.scope,
            "fill": cfg.fill,
            "lambda": cfg.lam,
            "vs_sdpa": metrics_sdpa[cfg.name].asdict(),
            "vs_no_skip_fp8": metrics_fp8[cfg.name].asdict(),
            "skip": stats[cfg.name].asdict(),
        }
        results.append(rec)

    out = {
        **parse_workload(path),
        "path": str(path),
        "shape": [1, S, H, D],
        "num_query_tiles": nb,
        "total_head_key_tiles_per_workload": total_tiles,
        "seconds": time.time() - t0,
        "peak_memory_bytes": peak,
        "baseline_fp8_static_p_vs_sdpa": baseline.asdict(),
        "results": results,
    }

    del q, k, v, ref, qf, kf, vf, qh, kt, vh, vb, v_sum, v_sum_bins
    torch.cuda.empty_cache()
    return out


def aggregate_reports(reports: list[dict]) -> dict:
    grouped = defaultdict(list)
    for rep in reports:
        for rec in rep["results"]:
            grouped[rec["config"]].append(rec)

    summary = {}
    for name, rows in grouped.items():
        def mean_at(path: tuple[str, ...]) -> float:
            vals = []
            for row in rows:
                cur = row
                for p in path:
                    cur = cur[p]
                vals.append(float(cur))
            return statistics.mean(vals)

        first = rows[0]
        summary[name] = {
            "scope": first["scope"],
            "fill": first["fill"],
            "lambda": first["lambda"],
            "skip": {
                "row_skip_rate": mean_at(("skip", "row_skip_rate")),
                "full_tile_skip_rate": mean_at(("skip", "full_tile_skip_rate")),
                "partial_tile_rate": mean_at(("skip", "partial_tile_rate")),
            },
            "vs_sdpa": {
                "rel_rmse": mean_at(("vs_sdpa", "rel_rmse")),
                "cosine_global": mean_at(("vs_sdpa", "cosine_global")),
                "cosine_row_p05": mean_at(("vs_sdpa", "cosine_row_p05")),
            },
            "vs_no_skip_fp8": {
                "rel_rmse": mean_at(("vs_no_skip_fp8", "rel_rmse")),
                "cosine_global": mean_at(("vs_no_skip_fp8", "cosine_global")),
                "cosine_row_p05": mean_at(("vs_no_skip_fp8", "cosine_row_p05")),
            },
        }

    return dict(sorted(summary.items(), key=lambda kv: (kv[1]["lambda"], kv[1]["scope"], kv[1]["fill"])))


def print_top(summary: dict, *, limit: int = 30) -> None:
    rows = list(summary.items())
    rows.sort(key=lambda kv: (kv[1]["vs_no_skip_fp8"]["rel_rmse"], -kv[1]["skip"]["row_skip_rate"]))
    print("\nlowest skip-only relRMSE:")
    print("| config | row skip | full tile skip | partial tile | skip-only relRMSE | SDPA relRMSE |")
    print("|---|---:|---:|---:|---:|---:|")
    for name, rec in rows[:limit]:
        print(
            f"| {name} | {rec['skip']['row_skip_rate']:.4f} | {rec['skip']['full_tile_skip_rate']:.4f} "
            f"| {rec['skip']['partial_tile_rate']:.4f} | {rec['vs_no_skip_fp8']['rel_rmse']:.4e} "
            f"| {rec['vs_sdpa']['rel_rmse']:.4e} |"
        )

    print("\nbest row skip under skip-only relRMSE budgets:")
    print("| budget | config | row skip | full tile skip | partial tile | skip-only relRMSE |")
    print("|---:|---|---:|---:|---:|---:|")
    for budget in [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1]:
        feasible = [(n, r) for n, r in rows if r["vs_no_skip_fp8"]["rel_rmse"] <= budget]
        if not feasible:
            print(f"| {budget:.3f} | - | - | - | - | - |")
            continue
        best = max(feasible, key=lambda kv: kv[1]["skip"]["row_skip_rate"])
        n, r = best
        print(
            f"| {budget:.3f} | {n} | {r['skip']['row_skip_rate']:.4f} | "
            f"{r['skip']['full_tile_skip_rate']:.4f} | {r['skip']['partial_tile_rate']:.4f} | "
            f"{r['vs_no_skip_fp8']['rel_rmse']:.4e} |"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    home = Path(os.path.expanduser("~"))
    ap.add_argument("--data-root", type=Path, default=home / "dataset/v-dit/wan21_p1")
    ap.add_argument("--workloads", nargs="*", type=Path, default=None)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--lambdas", nargs="*", type=float, default=DEFAULT_LAMBDAS)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--matmul-dtype", choices=["bf16", "fp32"], default="bf16")
    ap.add_argument("--no-row-cos", action="store_true",
                    help="skip per-row cosine collection for wide sweeps")
    ap.add_argument("--group-sizes", nargs="*", type=int, default=[],
                    help="add group<N> row-group skip scopes; each N must divide 128")
    ap.add_argument(
        "--fill-modes",
        nargs="*",
        default=["oracle_pq", "oracle_unq", "max_a1", "max_a0.25", "max_a0.0625", "mean", "logn", "sample8", "thr_a0.25"],
    )
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA device required")
    device = torch.device("cuda")
    dtype = torch.bfloat16 if args.matmul_dtype == "bf16" else torch.float32

    workloads = args.workloads if args.workloads else iter_workloads(Path(os.path.expanduser(str(args.data_root))))
    workloads = [Path(os.path.expanduser(str(w))) for w in workloads]
    if args.limit is not None:
        workloads = workloads[: args.limit]
    if not workloads:
        raise SystemExit("no workloads selected")

    bad_groups = [g for g in args.group_sizes if QUERY_TILE % g != 0]
    if bad_groups:
        raise SystemExit(f"group sizes must divide {QUERY_TILE}: {bad_groups}")

    cfgs = configs(args.lambdas, args.fill_modes, args.group_sizes)
    manifest = {
        "data_root": str(args.data_root),
        "workloads": [str(w) for w in workloads],
        "num_workloads": len(workloads),
        "lambdas": args.lambdas,
        "fill_modes": args.fill_modes,
        "group_sizes": args.group_sizes,
        "configs": [c.name for c in cfgs],
        "matmul_dtype": str(dtype),
        "gpu_name": torch.cuda.get_device_name(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "collect_row_cos": not args.no_row_cos,
    }

    reports = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for idx, wl in enumerate(workloads):
        rep = run_workload(
            wl, device, cfgs=cfgs, matmul_dtype=dtype, collect_row_cos=not args.no_row_cos
        )
        reports.append(rep)
        partial = {"manifest": manifest, "reports": reports, "summary": aggregate_reports(reports)}
        with args.output.open("w") as f:
            json.dump(partial, f, indent=2)
        print(
            f"[{idx + 1}/{len(workloads)}] {wl}: "
            f"baseline relRMSE={rep['baseline_fp8_static_p_vs_sdpa']['rel_rmse']:.4e} "
            f"peak={rep['peak_memory_bytes']/1e9:.1f}GB time={rep['seconds']:.1f}s",
            flush=True,
        )

    summary = aggregate_reports(reports)
    with args.output.open("w") as f:
        json.dump({"manifest": manifest, "reports": reports, "summary": summary}, f, indent=2)
    print(f"wrote {args.output}")
    print_top(summary)


if __name__ == "__main__":
    main()
