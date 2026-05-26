"""Count rowmax update/rescale frequency in a Triton diagnostic kernel."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from low_bit_fake_quant import QuantConfig  # noqa: E402
from low_bit_fake_quant.attention import (  # noqa: E402
    _dequant_qk,
    _estimate_rowmax_from_qm_k,
    _quant_qk,
    prepare_for_attention,
)


@triton.jit
def _count_kernel(
    Q,
    K,
    K_SMOOTH,
    QM,
    ROWMAX_EST,
    COUNTERS,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_ksz,
    stride_ksh,
    stride_ksn,
    stride_ksk,
    stride_qmz,
    stride_qmg,
    stride_qmh,
    stride_qmd,
    stride_rmz,
    stride_rmh,
    stride_rmm,
    H: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    sm_scale,
    UP_THRESHOLD_LOG2: tl.constexpr,
    DOWN_THRESHOLD_LOG2: tl.constexpr,
    MASS_FLOOR: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    D: tl.constexpr,
    Q_SMOOTH_BLOCK: tl.constexpr,
    USE_ESTIMATE: tl.constexpr,
    DOWN_ALWAYS: tl.constexpr,
):
    LOG2E: tl.constexpr = 1.4426950408889634
    up_threshold: tl.constexpr = UP_THRESHOLD_LOG2 / LOG2E
    down_threshold: tl.constexpr = DOWN_THRESHOLD_LOG2 / LOG2E

    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    q_off = off_z * stride_qz + off_h * stride_qh
    k_off = off_z * stride_kz + off_h * stride_kh
    ks_off = off_z * stride_ksz + off_h * stride_ksh
    qm_off = off_z * stride_qmz + off_h * stride_qmh
    rm_off = off_z * stride_rmz + off_h * stride_rmh

    q = tl.load(
        Q + q_off + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk,
        mask=offs_m[:, None] < M,
        other=0.0,
    )
    qm_group = (start_m * BLOCK_M) // Q_SMOOTH_BLOCK
    qm_vec = tl.load(QM + qm_off + qm_group * stride_qmg + offs_d * stride_qmd)

    if USE_ESTIMATE:
        m_i = tl.load(
            ROWMAX_EST + rm_off + offs_m * stride_rmm,
            mask=offs_m < M,
            other=0.0,
        ).to(tl.float32)
    else:
        m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    max_seen_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    update_count = tl.zeros([BLOCK_M], dtype=tl.int32)

    updates = tl.full((), 0, dtype=tl.int64)
    rescales = tl.full((), 0, dtype=tl.int64)
    upward = tl.full((), 0, dtype=tl.int64)
    downward = tl.full((), 0, dtype=tl.int64)

    for start_n in range(0, N, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        col_mask = offs_n < N
        k = tl.load(
            K + k_off + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk,
            mask=col_mask[:, None],
            other=0.0,
        )
        s_ij = tl.dot(q, tl.trans(k)).to(tl.float32) * sm_scale
        k_smooth = tl.load(
            K_SMOOTH + ks_off + offs_n[:, None] * stride_ksn + offs_d[None, :] * stride_ksk,
            mask=col_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        corr_n = tl.sum(qm_vec[None, :] * k_smooth, axis=1) * sm_scale
        s_ij = s_ij + corr_n[None, :]
        s_ij = tl.where((offs_m[:, None] < M) & col_mask[None, :], s_ij, float("-inf"))
        block_max = tl.max(s_ij, axis=1)

        if USE_ESTIMATE:
            max_seen_ij = tl.maximum(max_seen_i, block_max)
            under = (block_max - m_i) > up_threshold
            over = (m_i - max_seen_ij) > down_threshold
            if not DOWN_ALWAYS:
                over = over & (l_i < MASS_FLOOR)
            adjust = under | over
            down_m_bounded = tl.maximum(max_seen_ij + down_threshold, m_i - 80.0)
            down_m = tl.where(l_i == 0.0, max_seen_ij, down_m_bounded)
            adjusted_m = tl.where(under, block_max, down_m)
            m_ij = tl.where(adjust, adjusted_m, m_i)
            alpha_update = tl.exp2((m_i - m_ij) * LOG2E)
            alpha_update = tl.where(l_i == 0.0, 0.0, alpha_update)
            alpha = tl.where(adjust, alpha_update, 1.0)
            max_seen_i = max_seen_ij
            rescale = adjust & (l_i != 0.0)
            upward += tl.sum(tl.where(adjust & under, 1, 0), axis=0)
            downward += tl.sum(tl.where(adjust & (~under), 1, 0), axis=0)
        else:
            first = m_i == float("-inf")
            adjust = first | ((block_max - m_i) > up_threshold)
            m_ij = tl.where(adjust, block_max, m_i)
            alpha = tl.exp2((m_i - m_ij) * LOG2E)
            alpha = tl.where(first, 0.0, alpha)
            rescale = adjust & (~first)
            upward += tl.sum(tl.where(adjust, 1, 0), axis=0)

        p_sum = tl.sum(tl.exp2((s_ij - m_ij[:, None]) * LOG2E), axis=1)
        l_i = l_i * alpha + p_sum
        m_i = m_ij
        updates += tl.sum(tl.where(adjust, 1, 0), axis=0)
        rescales += tl.sum(tl.where(rescale, 1, 0), axis=0)
        update_count += tl.where(adjust, 1, 0)

    valid = offs_m < M
    rows_with_update = tl.sum(tl.where((update_count > 0) & valid, 1, 0), axis=0)
    max_updates = tl.max(tl.where(valid, update_count, 0), axis=0)
    tl.atomic_add(COUNTERS + 0, updates)
    tl.atomic_add(COUNTERS + 1, rescales)
    tl.atomic_add(COUNTERS + 2, upward)
    tl.atomic_add(COUNTERS + 3, downward)
    tl.atomic_add(COUNTERS + 4, rows_with_update)
    tl.atomic_max(COUNTERS + 5, max_updates)


def _load_qkv(path: Path, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    obj = torch.load(path, map_location="cpu")
    return (
        obj["query"].to(device=device, dtype=torch.bfloat16),
        obj["key"].to(device=device, dtype=torch.bfloat16),
        obj["value"].to(device=device, dtype=torch.bfloat16),
    )


def _pad(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, block: int):
    pad = (block - q.shape[1] % block) % block
    if pad == 0:
        return q, k, v, 0
    return (
        torch.cat([q, q.new_zeros(q.shape[0], pad, q.shape[2], q.shape[3])], dim=1),
        torch.cat([k, k[:, -1:].expand(k.shape[0], pad, k.shape[2], k.shape[3])], dim=1),
        torch.cat([v, v[:, -1:].expand(v.shape[0], pad, v.shape[2], v.shape[3])], dim=1),
        pad,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--q-smooth-block", type=int, default=256)
    ap.add_argument("--q-kmeans-k", type=int, default=32)
    ap.add_argument("--fp8-block-size", type=int, default=64)
    ap.add_argument("--block-m", type=int, default=64)
    ap.add_argument("--block-n", type=int, default=64)
    ap.add_argument("--thresholds", nargs="+", type=float, default=[8.0, 16.0])
    ap.add_argument("--up-thresholds", nargs="+", type=float, default=None)
    ap.add_argument("--down-thresholds", nargs="+", type=float, default=None)
    ap.add_argument("--down-always", action="store_true")
    args = ap.parse_args()

    device = torch.device(args.device)
    q, k, v = _load_qkv(args.workload, device)
    orig_s = q.shape[1]
    q, k, v, pad = _pad(q, k, v, args.q_smooth_block)
    cfg = QuantConfig(
        qk_quant="fp8_block",
        v_quant="fp8_channel",
        smoothing="full",
        q_kmeans_k=args.q_kmeans_k,
        q_smooth_block_size=args.q_smooth_block,
        fp8_block_size=args.fp8_block_size,
        v_smooth_mode="off",
        v_kmeans_k=None,
        p_quant="dynamic",
        rowmax_mode="qm_k",
        p_requant=True,
        p_requant_block_m=args.block_m,
        p_requant_block_n=args.block_n,
    )
    cache = prepare_for_attention(q, k, v, cfg)
    q_fp8, q_scale, q_meta = _quant_qk(cache.q_work, cfg)
    k_fp8, k_scale, k_meta = _quant_qk(cache.k_work, cfg)
    q_deq = _dequant_qk(q_fp8, q_scale, q_meta, torch.bfloat16).permute(0, 2, 1, 3).contiguous()
    k_deq = _dequant_qk(k_fp8, k_scale, k_meta, torch.bfloat16).permute(0, 2, 1, 3).contiguous()
    k_smooth = cache.k_work.to(torch.bfloat16).permute(0, 2, 1, 3).contiguous()
    qm = cache.qm.to(torch.float32).contiguous()
    rowmax = _estimate_rowmax_from_qm_k(
        cache.qm,
        cache.k_work,
        block_q=args.q_smooth_block,
        sm_scale=1.0 / math.sqrt(q.shape[-1]),
    ).reshape(q.shape[0], q.shape[2], q.shape[1]).contiguous()

    b, h, s, d = q_deq.shape
    grid = (triton.cdiv(s, args.block_m), b * h)
    results = {}
    pairs: list[tuple[float, float]]
    if args.up_thresholds is not None or args.down_thresholds is not None:
        ups = args.up_thresholds if args.up_thresholds is not None else args.thresholds
        downs = args.down_thresholds if args.down_thresholds is not None else args.thresholds
        pairs = [(float(up), float(down)) for up in ups for down in downs]
    else:
        pairs = [(float(threshold), float(threshold)) for threshold in args.thresholds]

    for up_threshold, down_threshold in pairs:
        key = f"up{up_threshold}_down{down_threshold}"
        results[key] = {}
        for name, use_est in (("fa4_online", False), ("estimated", True)):
            counters = torch.zeros(6, device=device, dtype=torch.int64)
            _count_kernel[grid](
                q_deq,
                k_deq,
                k_smooth,
                qm,
                rowmax,
                counters,
                q_deq.stride(0),
                q_deq.stride(1),
                q_deq.stride(2),
                q_deq.stride(3),
                k_deq.stride(0),
                k_deq.stride(1),
                k_deq.stride(2),
                k_deq.stride(3),
                k_smooth.stride(0),
                k_smooth.stride(1),
                k_smooth.stride(2),
                k_smooth.stride(3),
                qm.stride(0),
                qm.stride(1),
                qm.stride(2),
                qm.stride(3),
                rowmax.stride(0),
                rowmax.stride(1),
                rowmax.stride(2),
                H=h,
                M=s,
                N=s,
                sm_scale=1.0 / math.sqrt(d),
                UP_THRESHOLD_LOG2=float(up_threshold),
                DOWN_THRESHOLD_LOG2=float(down_threshold),
                MASS_FLOOR=1.0e-6,
                BLOCK_M=args.block_m,
                BLOCK_N=args.block_n,
                D=d,
                Q_SMOOTH_BLOCK=args.q_smooth_block,
                USE_ESTIMATE=use_est,
                DOWN_ALWAYS=args.down_always,
                num_warps=4,
                num_stages=2,
            )
            torch.cuda.synchronize()
            vals = counters.cpu().tolist()
            rows = b * h * s
            row_tiles = rows * triton.cdiv(s, args.block_n)
            results[key][name] = {
                "updates": vals[0],
                "rescales": vals[1],
                "upward_updates": vals[2],
                "downward_updates": vals[3],
                "rows_with_update": vals[4],
                "max_updates_per_row": vals[5],
                "updates_per_row_avg": vals[0] / rows,
                "rescales_per_row_avg": vals[1] / rows,
                "update_rate_per_row_tile": vals[0] / row_tiles,
                "rescale_rate_per_row_tile": vals[1] / row_tiles,
            }

    report = {
        "workload": str(args.workload),
        "shape": [b, s, h, d],
        "orig_s": orig_s,
        "pad": pad,
        "q_smooth_block": args.q_smooth_block,
        "q_kmeans_k": args.q_kmeans_k,
        "fp8_block_size": args.fp8_block_size,
        "block_m": args.block_m,
        "block_n": args.block_n,
        "results": results,
        "down_always": args.down_always,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
