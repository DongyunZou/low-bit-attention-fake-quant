"""Rank candidate space-time (t,h,w) reorder layouts by attention-mass locality.

The Wan2.1 traces carry no (t,h,w) metadata (DEC-4). This empirically picks the
factorization of the sequence length whose 128-token spatial-temporal blocks
concentrate the most attention mass on the block diagonal — i.e. the layout that
makes whole 128x128 tiles most skippable. It permutes Q/K by each candidate and
measures the mean fraction of softmax mass that falls in a query tile's own
128-block, on a small stratified sample of real workloads.

Run:
    CUDA_VISIBLE_DEVICES=3 uv run python bench/rank_reorder_layouts.py \
        --out agent_space/blasst_skip/layout_ranking.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path

import torch

from low_bit_fake_quant.blasst_skip import (
    QUERY_TILE,
    block_diagonal_mass,
    choose_local_block,
    space_time_reorder_index,
)


def divisors(n: int):
    ds = []
    i = 1
    while i * i <= n:
        if n % i == 0:
            ds.append(i)
            if i != n // i:
                ds.append(n // i)
        i += 1
    return sorted(ds)


def candidate_grids(seqlen: int, *, t_range=(4, 48), spatial_range=(16, 200), limit=None):
    """Video-plausible (t,h,w) factorizations of ``seqlen`` that admit a 128-block.

    Implausible aspect ratios (e.g. w=1920) are excluded. By default there is no
    cap (``limit=None``) so every plausible grid is scored.
    """
    grids = []
    for t in divisors(seqlen):
        if not (t_range[0] <= t <= t_range[1]):
            continue
        hw = seqlen // t
        for h in divisors(hw):
            w = hw // h
            if not (spatial_range[0] <= h <= spatial_range[1]):
                continue
            if not (spatial_range[0] <= w <= spatial_range[1]):
                continue
            try:
                choose_local_block(t, h, w)
            except ValueError:
                continue
            grids.append((t, h, w))
    default = (20, 48, 72)
    ordered = ([default] if default in grids else []) + [g for g in grids if g != default]
    seen, out = set(), []
    for g in ordered:
        if g not in seen:
            seen.add(g)
            out.append(g)
        if limit is not None and len(out) >= limit:
            break
    return out


def select_layout(results: list) -> tuple:
    """Pick the overall-best layout (including native) and the best reorder.

    ``results`` is a list of ``{"grid", "diagonal_mass", ...}`` rows where the
    native row has ``grid is None``. Returns ``(best, best_reorder)`` where
    ``best`` may be the native row (recommend no reorder if no candidate beats
    native locality) and ``best_reorder`` is the top non-native candidate.
    """
    ranked = sorted(results, key=lambda r: -r["diagonal_mass"])
    best = ranked[0]
    best_reorder = next((r for r in ranked if r["grid"] is not None), None)
    return best, best_reorder


def load_sample(path: Path, device):
    data = torch.load(path, map_location="cpu", weights_only=False)
    q = data["query"].to(torch.bfloat16).to(device)
    k = data["key"].to(torch.bfloat16).to(device)
    v = data["value"].to(torch.bfloat16).to(device)
    return q, k, v


def main() -> None:
    home = Path(os.path.expanduser("~"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", nargs="+", type=Path, default=[
        home / "dataset/v-dit/wan21_p1/layer_0/timestep_0.pt",
        home / "dataset/v-dit/wan21_p1/layer_20/timestep_3.pt",
        home / "dataset/v-dit/wan21_p1/layer_39/timestep_6.pt",
    ])
    ap.add_argument("--out", type=Path, default=Path("agent_space/blasst_skip/layout_ranking.json"))
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--n-query-tiles", type=int, default=8)
    args = ap.parse_args()

    device = torch.device("cuda")
    samples = [load_sample(p, device) for p in args.samples if p.exists()]
    if not samples:
        raise SystemExit("no sample workloads found")
    seqlen = samples[0][0].shape[1]
    grids = candidate_grids(seqlen)
    axis_orders = list(itertools.permutations(("t", "h", "w")))   # all 6 flatten orders

    def mean_mass(perm):
        return sum(
            block_diagonal_mass(q, k, perm, n_heads=args.n_heads, n_query_tiles=args.n_query_tiles)
            for q, k, _ in samples
        ) / len(samples)

    # native order baseline (identity permutation)
    native_score = mean_mass(torch.arange(seqlen, device=device))
    results = [{"grid": None, "block": None, "native_axis_order": None,
                "label": "native", "diagonal_mass": native_score}]

    scored = 0
    for g in grids:
        block = choose_local_block(*g)
        for order in axis_orders:
            perm = space_time_reorder_index(*g, block=block, native_axis_order=order, device=device)
            score = mean_mass(perm)
            scored += 1
            results.append({"grid": list(g), "block": list(block),
                            "native_axis_order": list(order),
                            "label": f"{g}/{''.join(order)}", "diagonal_mass": score})

    results.sort(key=lambda r: -r["diagonal_mass"])
    # Select the overall best, INCLUDING native: if no reorder beats native
    # locality, recommend no reorder (selected_grid = None) rather than a
    # candidate that is worse than doing nothing.
    best, best_reorder = select_layout(results)

    out = {
        "seqlen": seqlen,
        "samples": [str(p) for p in args.samples if p.exists()],
        "metric": "mean fraction of softmax mass in the query tile's own 128-token block",
        "block_tile": QUERY_TILE,
        "candidate_count_total": len(grids),
        "candidate_count_scored": scored,
        "axis_orders_per_grid": len(axis_orders),
        "native_diagonal_mass": native_score,
        "native_is_best": best["grid"] is None,
        "selected_grid": best["grid"],
        "selected_block": best["block"],
        "selected_native_axis_order": best["native_axis_order"],
        "best_reorder_grid": best_reorder["grid"] if best_reorder else None,
        "best_reorder_block": best_reorder["block"] if best_reorder else None,
        "best_reorder_native_axis_order": best_reorder["native_axis_order"] if best_reorder else None,
        "best_reorder_diagonal_mass": best_reorder["diagonal_mass"] if best_reorder else None,
        "ranking": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(out, f, indent=2)

    print(f"native diagonal-mass = {native_score:.4f} | grids={len(grids)} scored={scored}")
    print("rank  grid/order                  block        diagonal_mass")
    for i, r in enumerate(results[:20]):
        print(f"{i:>3}  {r['label']:<26} {str(r['block']):<12} {r['diagonal_mass']:.4f}")
    if best["grid"] is None:
        print(f"\nSELECTED: native order (no reorder beats native locality) -> {args.out}")
    else:
        print(f"\nSELECTED grid={best['grid']} block={best['block']} "
              f"order={best['native_axis_order']} -> {args.out}")


if __name__ == "__main__":
    main()
