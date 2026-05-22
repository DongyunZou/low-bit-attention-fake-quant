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


@triton.jit
def _smooth_v_per_block_kernel(
    v, v_out, v_alpha,
    stride_vb, stride_vs, stride_vh, stride_vd,
    stride_ob, stride_os, stride_oh, stride_od,
    stride_ab, stride_an, stride_ah, stride_ad,
    D: tl.constexpr, BLK_S: tl.constexpr,
):
    # Each program handles one (B, S-block, H) tile of size (BLK_S, D).
    pid_b = tl.program_id(0)
    pid_blk = tl.program_id(1)
    pid_h = tl.program_id(2)
    offs_s = pid_blk * BLK_S + tl.arange(0, BLK_S)
    offs_d = tl.arange(0, D)
    v_ptrs = (
        v
        + pid_b * stride_vb
        + offs_s[:, None] * stride_vs
        + pid_h * stride_vh
        + offs_d[None, :] * stride_vd
    )
    x = tl.load(v_ptrs).to(tl.float32)
    alpha = tl.sum(x, axis=0) / BLK_S  # (D,)
    o_ptrs = (
        v_out
        + pid_b * stride_ob
        + offs_s[:, None] * stride_os
        + pid_h * stride_oh
        + offs_d[None, :] * stride_od
    )
    tl.store(o_ptrs, (x - alpha[None, :]).to(v_out.dtype.element_ty))
    a_ptrs = (
        v_alpha
        + pid_b * stride_ab
        + pid_blk * stride_an
        + pid_h * stride_ah
        + offs_d * stride_ad
    )
    tl.store(a_ptrs, alpha)


def smooth_v_per_block(v: torch.Tensor, block_s: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-S-block V mean smoothing.

    Splits ``v`` along S into blocks of size ``block_s`` and subtracts each
    block's per-channel mean (a D-vector). Returns ``(v_centered, v_alpha)``
    where ``v_alpha`` is FP32 with shape ``(B, S/block_s, H, D)``.

    Inspired by Algorithm 2 in the SVG / SageAttention3 paper, this reduces
    V's local dynamic range so FP8 quantization is more accurate. The
    per-block mean is later folded back into the attention output via the C
    accumulator (Triton kernel path) or by inline reconstitution (SDPA
    path).
    """
    if v.ndim != 4 or not v.is_cuda:
        raise ValueError(f"v must be 4-D NHD on CUDA; got {tuple(v.shape)} on {v.device}")
    b, s, h, d = v.shape
    if s % block_s != 0:
        raise ValueError(f"S={s} must be divisible by block_s={block_s}")
    if d not in (64, 128):
        raise ValueError(f"D must be 64 or 128; got {d}")
    v = v.contiguous()
    n = s // block_s
    out = torch.empty_like(v)
    alpha = torch.empty((b, n, h, d), dtype=torch.float32, device=v.device)
    _smooth_v_per_block_kernel[(b, n, h)](
        v, out, alpha,
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        alpha.stride(0), alpha.stride(1), alpha.stride(2), alpha.stride(3),
        D=d, BLK_S=block_s,
    )
    return out, alpha


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
