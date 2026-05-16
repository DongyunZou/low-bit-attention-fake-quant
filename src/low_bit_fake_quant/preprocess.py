from __future__ import annotations

from typing import Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def _smooth_k_kernel(
    k, k_out, k_mean,
    stride_kb, stride_ks, stride_kh, stride_kd,
    stride_ob, stride_os, stride_oh, stride_od,
    stride_mb, stride_mh, stride_md,
    S: tl.constexpr, BLK_S: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_d = tl.program_id(2)
    base = k + pid_b * stride_kb + pid_h * stride_kh + pid_d * stride_kd
    acc = tl.zeros((), dtype=tl.float32)
    for s0 in range(0, S, BLK_S):
        offs_s = s0 + tl.arange(0, BLK_S)
        mask = offs_s < S
        vals = tl.load(base + offs_s * stride_ks, mask=mask, other=0.0).to(tl.float32)
        acc += tl.sum(vals, axis=0)
    mean = acc / S
    tl.store(k_mean + pid_b * stride_mb + pid_h * stride_mh + pid_d * stride_md, mean)
    out_base = k_out + pid_b * stride_ob + pid_h * stride_oh + pid_d * stride_od
    for s0 in range(0, S, BLK_S):
        offs_s = s0 + tl.arange(0, BLK_S)
        mask = offs_s < S
        vals = tl.load(base + offs_s * stride_ks, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_base + offs_s * stride_os, vals - mean, mask=mask)


def smooth_k(k: torch.Tensor, block_s: int = 256) -> Tuple[torch.Tensor, torch.Tensor]:
    """Triton K-mean subtraction over the S axis.

    Returns ``(k_smoothed, k_mean)``. ``k_mean`` is FP32 with shape
    ``(B, H, D)``; ``k_smoothed`` has the same dtype as ``k``.
    """
    if k.ndim != 4 or not k.is_cuda:
        raise ValueError(f"k must be 4-D NHD on CUDA; got {tuple(k.shape)} on {k.device}")
    b, s, h, d = k.shape
    k = k.contiguous()
    out = torch.empty_like(k)
    mean = torch.empty((b, h, d), dtype=torch.float32, device=k.device)
    _smooth_k_kernel[(b, h, d)](
        k, out, mean,
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        mean.stride(0), mean.stride(1), mean.stride(2),
        S=s, BLK_S=block_s,
    )
    return out, mean


@triton.jit
def _group_mean_q_kernel(
    q, q_out, qm,
    stride_qb, stride_qs, stride_qh, stride_qd,
    stride_ob, stride_os, stride_oh, stride_od,
    stride_mb, stride_mn, stride_mh, stride_md,
    D: tl.constexpr, BLK_Q: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_blk = tl.program_id(1)
    pid_h = tl.program_id(2)
    offs_s = pid_blk * BLK_Q + tl.arange(0, BLK_Q)
    offs_d = tl.arange(0, D)
    q_ptrs = (
        q
        + pid_b * stride_qb
        + offs_s[:, None] * stride_qs
        + pid_h * stride_qh
        + offs_d[None, :] * stride_qd
    )
    x = tl.load(q_ptrs).to(tl.float32)
    mean = tl.sum(x, axis=0) / BLK_Q
    o_ptrs = (
        q_out
        + pid_b * stride_ob
        + offs_s[:, None] * stride_os
        + pid_h * stride_oh
        + offs_d[None, :] * stride_od
    )
    tl.store(o_ptrs, x - mean[None, :])
    m_ptrs = qm + pid_b * stride_mb + pid_blk * stride_mn + pid_h * stride_mh + offs_d * stride_md
    tl.store(m_ptrs, mean)


def group_mean_q(q: torch.Tensor, block_q: int = 256) -> Tuple[torch.Tensor, torch.Tensor]:
    """Triton Q centering by groups along S.

    Returns ``(q_centered, qm)`` where ``qm`` is FP32 with shape
    ``(B, S / block_q, H, D)``.
    """
    if q.ndim != 4 or not q.is_cuda:
        raise ValueError(f"q must be 4-D NHD on CUDA; got {tuple(q.shape)} on {q.device}")
    b, s, h, d = q.shape
    if s % block_q != 0:
        raise ValueError(f"S={s} must be divisible by block_q={block_q}")
    if d not in (64, 128):
        raise ValueError(f"D must be 64 or 128; got {d}")
    q = q.contiguous()
    n = s // block_q
    out = torch.empty_like(q)
    qm = torch.empty((b, n, h, d), dtype=torch.float32, device=q.device)
    _group_mean_q_kernel[(b, n, h)](
        q, out, qm,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        qm.stride(0), qm.stride(1), qm.stride(2), qm.stride(3),
        D=d, BLK_Q=block_q,
    )
    return out, qm
