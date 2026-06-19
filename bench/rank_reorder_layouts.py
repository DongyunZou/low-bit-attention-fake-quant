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


def candidate_grids(seqlen: int, *, t_range=(4, 48), spatial_range=(16, 200), limit: int = 16):
    """Video-plausible (t,h,w) factorizations of ``seqlen`` that admit a *genuine*
    (non-identity) 128-token space-time block.

    Implausible aspect ratios (e.g. w=1920) and identity-equivalent layouts
    (a pure contiguous ``(1,1,128)`` block, which reproduces native order) are
    excluded so the ranking compares real reorders.
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
                block = choose_local_block(t, h, w)
            except ValueError:
                continue
            # genuine reorder only: the block must span more than one axis, and
            # the permutation must differ from native order.
            perm = space_time_reorder_index(t, h, w, block=block)
            if torch.equal(perm, torch.arange(seqlen)):
                continue
            grids.append((t, h, w))
    default = (20, 48, 72)
    ordered = ([default] if default in grids else []) + [g for g in grids if g != default]
    seen, out = set(), []
    for g in ordered:
        if g not in seen:
            seen.add(g)
            out.append(g)
        if len(out) >= limit:
            break
    return out


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

    # native order baseline (identity)
    native_perm = torch.arange(seqlen, device=device)
    results = []
    native_score = sum(
        block_diagonal_mass(q, k, native_perm, n_heads=args.n_heads, n_query_tiles=args.n_query_tiles)
        for q, k, _ in samples
    ) / len(samples)
    results.append({"grid": None, "block": None, "label": "native", "diagonal_mass": native_score})

    for g in grids:
        block = choose_local_block(*g)
        perm = space_time_reorder_index(*g, block=block, device=device)
        score = sum(
            block_diagonal_mass(q, k, perm, n_heads=args.n_heads, n_query_tiles=args.n_query_tiles)
            for q, k, _ in samples
        ) / len(samples)
        results.append({"grid": list(g), "block": list(block), "label": f"{g}", "diagonal_mass": score})

    results.sort(key=lambda r: -r["diagonal_mass"])
    best = next(r for r in results if r["grid"] is not None)

    out = {
        "seqlen": seqlen,
        "samples": [str(p) for p in args.samples if p.exists()],
        "metric": "mean fraction of softmax mass in the query tile's own 128-token block",
        "block_tile": QUERY_TILE,
        "native_diagonal_mass": native_score,
        "ranking": results,
        "selected_grid": best["grid"],
        "selected_block": best["block"],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(out, f, indent=2)

    print(f"native diagonal-mass = {native_score:.4f}")
    print("rank  grid                block        diagonal_mass")
    for i, r in enumerate(results):
        print(f"{i:>3}  {str(r['grid']):<18} {str(r['block']):<12} {r['diagonal_mass']:.4f}")
    print(f"\nSELECTED grid={best['grid']} block={best['block']} -> {args.out}")


if __name__ == "__main__":
    main()
