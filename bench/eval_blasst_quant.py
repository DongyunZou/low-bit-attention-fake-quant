"""Evaluate BLASST-style block skipping combined with FP8 fake quant on Wan QKV dumps.

This benchmark intentionally disables all existing numerical tricks except the
BLASST-style change under test:

* no K/Q smoothing
* no Q/V k-means
* no Hadamard rotation
* no V smoothing

The quantized path uses Q/K/V fp8_block fake quant and the Triton P-requant
attention kernel. Torch SDPA on the raw BF16 Q/K/V is the numerical ground
truth. ``block_m = block_n = 128`` is fixed by default for the requested QK
tile size.
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
import torch.nn.functional as F

from low_bit_fake_quant.attention_triton import fake_quant_attention_triton
from low_bit_fake_quant.quant_triton import fp8_block_dequant, fp8_block_quant


@dataclasses.dataclass
class Workload:
    layer: str
    timestep: str
    path: Path


@dataclasses.dataclass
class Metrics:
    mse: float
    rmse: float
    cosine: float
    rel_l2: float
    max_abs_err: float


def iter_workloads(root: Path) -> Iterable[Workload]:
    for layer_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for ts_file in sorted(layer_dir.glob("timestep_*.pt")):
            yield Workload(layer=layer_dir.name, timestep=ts_file.stem, path=ts_file)


def load_qkv(path: Path, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    return (
        data["query"].to(device, non_blocking=True),
        data["key"].to(device, non_blocking=True),
        data["value"].to(device, non_blocking=True),
    )


def reference_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    scale = 1.0 / math.sqrt(q.shape[-1])
    out = F.scaled_dot_product_attention(
        q.permute(0, 2, 1, 3).contiguous(),
        k.permute(0, 2, 1, 3).contiguous(),
        v.permute(0, 2, 1, 3).contiguous(),
        scale=scale,
    )
    return out.permute(0, 2, 1, 3).contiguous()


def compute_metrics(out: torch.Tensor, ref: torch.Tensor) -> Metrics:
    a = out.float().flatten()
    b = ref.float().flatten()
    diff = a - b
    mse = float(diff.square().mean().item())
    rmse = math.sqrt(mse)
    denom = torch.clamp(torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b), min=1e-12)
    cosine = float((torch.dot(a, b) / denom).item())
    rel_l2 = float((torch.linalg.vector_norm(diff) / torch.clamp(torch.linalg.vector_norm(b), min=1e-12)).item())
    max_abs = float(diff.abs().max().item())
    return Metrics(mse=mse, rmse=rmse, cosine=cosine, rel_l2=rel_l2, max_abs_err=max_abs)


def infer_grid_s(q: torch.Tensor, hw: tuple[int, int]) -> tuple[int, int, int, int]:
    s = q.shape[1]
    h, w = hw
    hw_tokens = h * w
    t = s // hw_tokens
    grid_s = t * hw_tokens
    if t <= 0:
        raise ValueError(f"S={s} is smaller than hw={hw}")
    return t, h, w, grid_s


def block_shuffle_order(
    *,
    t: int,
    h: int,
    w: int,
    ts: int,
    hs: int,
    ws: int,
    tail: int,
    device: torch.device,
) -> torch.Tensor:
    """Return token order that groups nearby 3D patches into contiguous blocks.

    Original layout is assumed to be row-major ``[t, h, w]``. The shuffled order
    is block-major: ``[t_block, h_block, w_block, inner_t, inner_h, inner_w]``.
    If dimensions are not divisible by the tile size, invalid padded positions
    are dropped and any non-grid tail tokens are appended unchanged.
    """

    ids = torch.arange(t * h * w, device=device, dtype=torch.long).reshape(t, h, w)
    nt = math.ceil(t / ts)
    nh = math.ceil(h / hs)
    nw = math.ceil(w / ws)
    pad_t = nt * ts - t
    pad_h = nh * hs - h
    pad_w = nw * ws - w
    padded = F.pad(ids, (0, pad_w, 0, pad_h, 0, pad_t), value=-1)
    blocks = padded.reshape(nt, ts, nh, hs, nw, ws).permute(0, 2, 4, 1, 3, 5)
    order = blocks.reshape(-1)
    order = order[order >= 0]
    if tail > 0:
        tail_ids = torch.arange(t * h * w, t * h * w + tail, device=device, dtype=torch.long)
        order = torch.cat([order, tail_ids])
    return order.contiguous()


def apply_order(x: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
    return x.index_select(1, order).contiguous()


def invert_order(order: torch.Tensor) -> torch.Tensor:
    inv = torch.empty_like(order)
    inv.scatter_(0, order, torch.arange(order.numel(), device=order.device))
    return inv


def diagonal_rowmax_init(q_bhsd: torch.Tensor, k_bhsd: torch.Tensor, sm_scale: float) -> torch.Tensor:
    return (q_bhsd.float() * k_bhsd.float()).sum(dim=-1) * float(sm_scale)


def quantized_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    block_size: int,
    blasst_lambda: float | None,
    blasst_fill: str,
    blasst_fill_alpha: float,
    blasst_uta_bins: int,
    rowmax_init: bool,
) -> tuple[torch.Tensor, dict[str, float] | None]:
    b, s, h, d = q.shape
    sm_scale = 1.0 / math.sqrt(d)

    q_fp8, q_scale = fp8_block_quant(q, block_s=block_size)
    k_fp8, k_scale = fp8_block_quant(k, block_s=block_size)
    v_fp8, v_scale = fp8_block_quant(v, block_s=block_size)
    q_deq = fp8_block_dequant(q_fp8, q_scale, block_s=block_size, dtype=torch.bfloat16)
    k_deq = fp8_block_dequant(k_fp8, k_scale, block_s=block_size, dtype=torch.bfloat16)
    # Keep V in raw FP8 value units; the per-block scale is applied inside the K loop.
    v_bf16 = v_fp8.to(torch.bfloat16)
    v_scale_bhd = torch.ones((b, h, d), dtype=torch.float32, device=q.device)

    q_bhsd = q_deq.permute(0, 2, 1, 3).contiguous()
    k_bhsd = k_deq.permute(0, 2, 1, 3).contiguous()
    v_bhsd = v_bf16.permute(0, 2, 1, 3).contiguous()
    v_block_scale = v_scale.to(torch.float32).contiguous()
    rowmax_init_bhs = diagonal_rowmax_init(q_bhsd, k_bhsd, sm_scale) if rowmax_init else None

    if blasst_lambda is None:
        out_bhsd = fake_quant_attention_triton(
            q_bhsd,
            k_bhsd,
            v_bhsd,
            v_scale_bhd,
            sm_scale=sm_scale,
            p_max_offset=8,
            block_m=block_size,
            block_n=block_size,
            v_block_scale_bsh=v_block_scale,
            v_block_size=block_size,
            p_quant_mode="elementwise",
        )
        skip_stats = None
    else:
        out_bhsd, stats = fake_quant_attention_triton(
            q_bhsd,
            k_bhsd,
            v_bhsd,
            v_scale_bhd,
            sm_scale=sm_scale,
            p_max_offset=8,
            block_m=block_size,
            block_n=block_size,
            v_block_scale_bsh=v_block_scale,
            v_block_size=block_size,
            p_quant_mode="elementwise",
            rowmax_init_bhs=rowmax_init_bhs,
            blasst_lambda=blasst_lambda,
            blasst_fill_mode=blasst_fill,
            blasst_fill_alpha=blasst_fill_alpha,
            blasst_uta_bins=blasst_uta_bins,
            return_blasst_stats=True,
        )
        skipped = float(stats[..., 0].sum().item())
        total = float(stats[..., 1].sum().item())
        tile_total = float(stats[..., 1].gt(0).sum().item())
        tile_skipped = float((stats[..., 0] == stats[..., 1]).logical_and(stats[..., 1].gt(0)).sum().item())
        skip_stats = {
            "row_skip_ratio": skipped / total if total > 0 else 0.0,
            "tile_skip_ratio": tile_skipped / tile_total if tile_total > 0 else 0.0,
            "skipped_row_blocks": skipped,
            "total_row_blocks": total,
            "skipped_full_tiles": tile_skipped,
            "total_tiles": tile_total,
        }
    return out_bhsd.permute(0, 2, 1, 3).contiguous().to(q.dtype), skip_stats


def run_one_variant(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    ref: torch.Tensor,
    *,
    order: torch.Tensor | None,
    block_size: int,
    lambdas: list[float],
    blasst_fill: str,
    blasst_fill_alpha: float,
    blasst_uta_bins: int,
    rowmax_init: bool,
) -> list[dict]:
    if order is not None:
        inv = invert_order(order)
        q_work = apply_order(q, order)
        k_work = apply_order(k, order)
        v_work = apply_order(v, order)
    else:
        inv = None
        q_work, k_work, v_work = q, k, v

    records: list[dict] = []

    out_base, _ = quantized_attention(
        q_work,
        k_work,
        v_work,
        block_size=block_size,
        blasst_lambda=None,
        blasst_fill="zero",
        blasst_fill_alpha=1.0,
        blasst_uta_bins=1,
        rowmax_init=False,
    )
    if inv is not None:
        out_base = apply_order(out_base, inv)
    base_metrics = compute_metrics(out_base, ref)
    records.append(
        {
            "lambda": None,
            "skip_ratio": None,
            "metrics": dataclasses.asdict(base_metrics),
        }
    )
    del out_base

    for lam in lambdas:
        out, skip_stats = quantized_attention(
            q_work,
            k_work,
            v_work,
            block_size=block_size,
            blasst_lambda=lam,
            blasst_fill=blasst_fill,
            blasst_fill_alpha=blasst_fill_alpha,
            blasst_uta_bins=blasst_uta_bins,
            rowmax_init=rowmax_init,
        )
        if inv is not None:
            out = apply_order(out, inv)
        metrics = compute_metrics(out, ref)
        records.append(
            {
                "lambda": lam,
                "skip_ratio": None if skip_stats is None else skip_stats["row_skip_ratio"],
                "skip_stats": skip_stats,
                "metrics": dataclasses.asdict(metrics),
            }
        )
        del out
    return records


def aggregate(results: dict) -> dict:
    slots: dict[str, dict[str, dict[str, list[float]]]] = {}
    for wl in results["workloads"]:
        for variant_name, records in wl["variants"].items():
            for rec in records:
                key = "dense" if rec["lambda"] is None else str(rec["lambda"])
                slot = slots.setdefault(variant_name, {}).setdefault(key, {})
                if rec["skip_ratio"] is not None:
                    slot.setdefault("skip_ratio", []).append(rec["skip_ratio"])
                if rec.get("skip_stats") is not None:
                    for stat_name, stat_value in rec["skip_stats"].items():
                        slot.setdefault(stat_name, []).append(stat_value)
                for metric, value in rec["metrics"].items():
                    slot.setdefault(metric, []).append(value)

    summary: dict[str, dict[str, dict[str, float]]] = {}
    for variant_name, by_lambda in slots.items():
        out_by_lambda = {}
        for lam, values in by_lambda.items():
            agg = {}
            for key, vals in values.items():
                if vals:
                    agg[f"{key}_mean"] = sum(vals) / len(vals)
                    agg[f"{key}_min"] = min(vals)
                    agg[f"{key}_max"] = max(vals)
            out_by_lambda[lam] = agg
        summary[variant_name] = out_by_lambda
    return summary


def plot_summary(summary: dict, out_dir: Path) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_paths: list[str] = []
    for metric, ylabel, filename in [
        ("skip_ratio_mean", "Skipped row-block decisions", "blasst_skip_ratio.png"),
        ("rmse_mean", "RMSE vs torch SDPA", "blasst_rmse.png"),
        ("cosine_mean", "Cosine vs torch SDPA", "blasst_cosine.png"),
    ]:
        series = []
        for variant, by_lambda in summary.items():
            xs = []
            ys = []
            for lam, vals in by_lambda.items():
                if lam == "dense" or metric not in vals:
                    continue
                xs.append(float(lam))
                ys.append(vals[metric])
            if not xs:
                continue
            pairs = sorted(zip(xs, ys, strict=True))
            series.append((variant, pairs))
        if not series:
            continue
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(7, 4.5))
            for variant, pairs in series:
                ax.plot([p[0] for p in pairs], [p[1] for p in pairs], marker="o", label=variant)
            ax.set_xscale("log")
            ax.set_xlabel("lambda")
            ax.set_ylabel(ylabel)
            ax.grid(True, which="both", alpha=0.25)
            ax.legend()
            fig.tight_layout()
            path = out_dir / filename
            fig.savefig(path, dpi=180)
            plt.close(fig)
        except ModuleNotFoundError:
            path = out_dir / filename.replace(".png", ".svg")
            write_svg_plot(series, ylabel=ylabel, path=path)
        plot_paths.append(str(path))
    return plot_paths


def write_svg_plot(series: list[tuple[str, list[tuple[float, float]]]], *, ylabel: str, path: Path) -> None:
    width, height = 760, 460
    left, right, top, bottom = 80, 24, 24, 72
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]
    xs = [math.log10(x) for _, points in series for x, _ in points]
    ys = [y for _, points in series for _, y in points]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if ymin == ymax:
        pad = abs(ymin) * 0.05 + 1e-9
        ymin -= pad
        ymax += pad
    if xmin == xmax:
        xmin -= 1.0
        xmax += 1.0

    def sx(x: float) -> float:
        return left + (math.log10(x) - xmin) / (xmax - xmin) * (width - left - right)

    def sy(y: float) -> float:
        return top + (ymax - y) / (ymax - ymin) * (height - top - bottom)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#333"/>',
        f'<text x="{width/2}" y="{height-24}" text-anchor="middle" font-family="sans-serif" font-size="14">lambda (log scale)</text>',
        f'<text x="18" y="{height/2}" text-anchor="middle" font-family="sans-serif" font-size="14" transform="rotate(-90 18 {height/2})">{ylabel}</text>',
    ]
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y = ymin + frac * (ymax - ymin)
        py = sy(y)
        lines.append(f'<line x1="{left}" y1="{py:.2f}" x2="{width-right}" y2="{py:.2f}" stroke="#ddd"/>')
        lines.append(f'<text x="{left-8}" y="{py+4:.2f}" text-anchor="end" font-family="sans-serif" font-size="11">{y:.4g}</text>')
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        xlog = xmin + frac * (xmax - xmin)
        xval = 10**xlog
        px = left + frac * (width - left - right)
        lines.append(f'<line x1="{px:.2f}" y1="{top}" x2="{px:.2f}" y2="{height-bottom}" stroke="#eee"/>')
        lines.append(f'<text x="{px:.2f}" y="{height-bottom+18}" text-anchor="middle" font-family="sans-serif" font-size="11">{xval:.1e}</text>')
    legend_y = top + 14
    for idx, (variant, points) in enumerate(series):
        color = colors[idx % len(colors)]
        pts = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in points)
        lines.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{pts}"/>')
        for x, y in points:
            lines.append(f'<circle cx="{sx(x):.2f}" cy="{sy(y):.2f}" r="3" fill="{color}"/>')
        lx = left + 12 + (idx % 2) * 260
        ly = legend_y + (idx // 2) * 20
        lines.append(f'<line x1="{lx}" y1="{ly}" x2="{lx+22}" y2="{ly}" stroke="{color}" stroke-width="2"/>')
        lines.append(f'<text x="{lx+28}" y="{ly+4}" font-family="sans-serif" font-size="12">{variant}</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("/home/dongyunz/dataset/v-dit/wan21_p1"))
    parser.add_argument("--output", type=Path, default=Path("bench/results_blasst_quant.json"))
    parser.add_argument("--plots-dir", type=Path, default=Path("bench/blasst_quant_plots"))
    parser.add_argument("--limit-workloads", type=int, default=None)
    parser.add_argument("--hw", nargs=2, type=int, default=(45, 80), metavar=("H", "W"))
    parser.add_argument("--tile", nargs=3, type=int, default=(1, 8, 16), metavar=("TS", "HS", "WS"))
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument(
        "--blasst-fill",
        default="zero",
        help=(
            "Fill mode for skipped BLASST row-blocks: zero, max, mean, logn, "
            "sample8, thr, uta, or probe spellings like mean_a1.5/uta16_a1.5"
        ),
    )
    parser.add_argument("--blasst-fill-alpha", type=float, default=1.0)
    parser.add_argument("--blasst-uta-bins", type=int, default=16)
    parser.add_argument(
        "--lambdas",
        nargs="+",
        type=float,
        default=[1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5],
    )
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    if args.block_size != 128:
        raise SystemExit("This benchmark is intended to keep QK block size at 128.")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    device = torch.device("cuda")
    workloads = list(iter_workloads(args.data_root))
    if args.limit_workloads is not None:
        workloads = workloads[: args.limit_workloads]
    if not workloads:
        raise SystemExit(f"no workloads found under {args.data_root}")

    results = {
        "data_root": str(args.data_root),
        "hw": list(args.hw),
        "tile": list(args.tile),
        "block_size": args.block_size,
        "lambdas": args.lambdas,
        "blasst_fill": args.blasst_fill,
        "blasst_fill_alpha": args.blasst_fill_alpha,
        "blasst_uta_bins": args.blasst_uta_bins,
        "notes": {
            "ground_truth": "torch SDPA on raw BF16 q/k/v",
            "quant_path": "fp8_block Q/K/V + elementwise P FP8 requant Triton kernel",
            "disabled_tricks": ["smoothing", "q_kmeans", "v_kmeans", "v_smoothing", "qk_hadamard"],
            "tail_policy": "3D shuffle covers floor(S / (H*W)) * H*W tokens; tail tokens are appended unchanged",
        },
        "workloads": [],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)

    for idx, wl in enumerate(workloads):
        print(f"[{idx + 1}/{len(workloads)}] loading {wl.path}")
        q, k, v = load_qkv(wl.path, device=device)
        b, s, h_heads, d = q.shape
        t_grid, h_grid, w_grid, grid_s = infer_grid_s(q, tuple(args.hw))
        tail = s - grid_s
        if s % args.block_size != 0:
            raise RuntimeError(f"S={s} must be divisible by block_size={args.block_size}")

        torch.cuda.synchronize()
        t0 = time.time()
        ref = reference_attention(q, k, v)
        torch.cuda.synchronize()
        ref_seconds = time.time() - t0
        print(
            f"[{idx + 1}/{len(workloads)}] shape={tuple(q.shape)} grid={(t_grid, h_grid, w_grid)} "
            f"tail={tail} ref={ref_seconds:.2f}s"
        )

        ts, hs, ws = args.tile
        shuffle_order = block_shuffle_order(
            t=t_grid,
            h=h_grid,
            w=w_grid,
            ts=ts,
            hs=hs,
            ws=ws,
            tail=tail,
            device=device,
        )
        variants = {
            "sequential": (None, False),
            "sequential_diag_init": (None, True),
            "shuffle": (shuffle_order, False),
            "shuffle_diag_init": (shuffle_order, True),
        }
        wl_record = {
            "layer": wl.layer,
            "timestep": wl.timestep,
            "path": str(wl.path),
            "shape": [b, s, h_heads, d],
            "grid": [t_grid, h_grid, w_grid],
            "grid_tokens": grid_s,
            "tail_tokens": tail,
            "reference_seconds": ref_seconds,
            "variants": {},
        }

        for name, (order, rowmax_init) in variants.items():
            torch.cuda.synchronize()
            t1 = time.time()
            records = run_one_variant(
                q,
                k,
                v,
                ref,
                order=order,
                block_size=args.block_size,
                lambdas=args.lambdas,
                blasst_fill=args.blasst_fill,
                blasst_fill_alpha=args.blasst_fill_alpha,
                blasst_uta_bins=args.blasst_uta_bins,
                rowmax_init=rowmax_init,
            )
            torch.cuda.synchronize()
            seconds = time.time() - t1
            wl_record["variants"][name] = records
            print(f"[{idx + 1}/{len(workloads)}] {name} done in {seconds:.2f}s")
            for rec in records:
                if rec["lambda"] is None:
                    print(
                        f"  dense: rmse={rec['metrics']['rmse']:.4e} "
                        f"cos={rec['metrics']['cosine']:.6f}"
                    )
                else:
                    skip_stats = rec.get("skip_stats") or {}
                    print(
                        f"  lambda={rec['lambda']:.1e} row_skip={rec['skip_ratio']:.3%} "
                        f"tile_skip={skip_stats.get('tile_skip_ratio', float('nan')):.3%} "
                        f"rmse={rec['metrics']['rmse']:.4e} cos={rec['metrics']['cosine']:.6f}"
                    )

        results["workloads"].append(wl_record)
        results["summary"] = aggregate(results)
        results["plots"] = plot_summary(results["summary"], args.plots_dir)
        with args.output.open("w") as f:
            json.dump(results, f, indent=2)

        del q, k, v, ref, shuffle_order
        gc.collect()
        torch.cuda.empty_cache()

    results["summary"] = aggregate(results)
    results["plots"] = plot_summary(results["summary"], args.plots_dir)
    with args.output.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {args.output}")
    for path in results["plots"]:
        print(f"plot {path}")


if __name__ == "__main__":
    main()
