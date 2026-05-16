from __future__ import annotations

from typing import Tuple

import torch
import triton
import triton.language as tl

_FP8_E4M3_MAX = 448.0
_AMAX_FLOOR = 1e-4
_UE8M0_EXP_MIN = -127.0
_UE8M0_EXP_MAX = 127.0


def _check_nhd(t: torch.Tensor, *, name: str = "tensor") -> tuple[int, int, int, int]:
    if t.ndim != 4:
        raise ValueError(f"{name} must be 4-D NHD (B, S, H, D); got {tuple(t.shape)}")
    if not t.is_cuda:
        raise ValueError(f"{name} must live on CUDA; got {t.device}")
    b, s, h, d = t.shape
    if d not in (64, 128):
        raise ValueError(f"{name}.shape[-1] must be 64 or 128; got {d}")
    return b, s, h, d


@triton.jit
def _fp8_block_quant_kernel(
    src, dst, scale,
    stride_sb, stride_ss, stride_sh, stride_sd,
    stride_db, stride_ds, stride_dh, stride_dd,
    stride_scb, stride_scs, stride_sch,
    D: tl.constexpr, BLK: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_blk = tl.program_id(1)
    pid_h = tl.program_id(2)
    offs_s = pid_blk * BLK + tl.arange(0, BLK)
    offs_d = tl.arange(0, D)
    src_ptrs = (
        src
        + pid_b * stride_sb
        + offs_s[:, None] * stride_ss
        + pid_h * stride_sh
        + offs_d[None, :] * stride_sd
    )
    x = tl.load(src_ptrs).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(x)), _AMAX_FLOOR)
    sc = amax / _FP8_E4M3_MAX
    xq = tl.minimum(tl.maximum(x / sc, -_FP8_E4M3_MAX), _FP8_E4M3_MAX)
    dst_ptrs = (
        dst
        + pid_b * stride_db
        + offs_s[:, None] * stride_ds
        + pid_h * stride_dh
        + offs_d[None, :] * stride_dd
    )
    tl.store(dst_ptrs, xq.to(dst.dtype.element_ty))
    tl.store(scale + pid_b * stride_scb + pid_blk * stride_scs + pid_h * stride_sch, sc)


def fp8_block_quant(t: torch.Tensor, block_s: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
    """FP32-scale FP8 e4m3fn quantization per (B, S/block_s, H)."""
    b, s, h, d = _check_nhd(t)
    if s % block_s != 0:
        raise ValueError(f"S={s} must be divisible by block_s={block_s}")
    t = t.contiguous()
    n = s // block_s
    out = torch.empty_like(t, dtype=torch.float8_e4m3fn)
    scale = torch.empty((b, n, h), dtype=torch.float32, device=t.device)
    _fp8_block_quant_kernel[(b, n, h)](
        t, out, scale,
        t.stride(0), t.stride(1), t.stride(2), t.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        scale.stride(0), scale.stride(1), scale.stride(2),
        D=d, BLK=block_s,
    )
    return out, scale


@triton.jit
def _fp8_block_dequant_kernel(
    src, scale, dst,
    stride_sb, stride_ss, stride_sh, stride_sd,
    stride_scb, stride_scs, stride_sch,
    stride_db, stride_ds, stride_dh, stride_dd,
    D: tl.constexpr, BLK: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_blk = tl.program_id(1)
    pid_h = tl.program_id(2)
    offs_s = pid_blk * BLK + tl.arange(0, BLK)
    offs_d = tl.arange(0, D)
    sc = tl.load(scale + pid_b * stride_scb + pid_blk * stride_scs + pid_h * stride_sch)
    src_ptrs = (
        src
        + pid_b * stride_sb
        + offs_s[:, None] * stride_ss
        + pid_h * stride_sh
        + offs_d[None, :] * stride_sd
    )
    dst_ptrs = (
        dst
        + pid_b * stride_db
        + offs_s[:, None] * stride_ds
        + pid_h * stride_dh
        + offs_d[None, :] * stride_dd
    )
    tl.store(dst_ptrs, tl.load(src_ptrs).to(tl.float32) * sc)


def fp8_block_dequant(
    fp8: torch.Tensor,
    scale: torch.Tensor,
    block_s: int = 128,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    b, s, h, d = _check_nhd(fp8, name="fp8")
    if fp8.dtype != torch.float8_e4m3fn:
        raise TypeError(f"fp8 must have dtype torch.float8_e4m3fn; got {fp8.dtype}")
    if tuple(scale.shape) != (b, s // block_s, h):
        raise ValueError(f"scale must have shape {(b, s // block_s, h)}; got {tuple(scale.shape)}")
    out = torch.empty((b, s, h, d), dtype=dtype, device=fp8.device)
    _fp8_block_dequant_kernel[(b, s // block_s, h)](
        fp8, scale, out,
        fp8.stride(0), fp8.stride(1), fp8.stride(2), fp8.stride(3),
        scale.stride(0), scale.stride(1), scale.stride(2),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        D=d, BLK=block_s,
    )
    return out


@triton.jit
def _fp8_channel_quant_kernel(
    src, dst, scale,
    stride_sb, stride_ss, stride_sh, stride_sd,
    stride_db, stride_ds, stride_dh, stride_dd,
    stride_scb, stride_sch, stride_scd,
    S: tl.constexpr, BLK_S: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_d = tl.program_id(2)
    base = src + pid_b * stride_sb + pid_h * stride_sh + pid_d * stride_sd
    amax = tl.full((), _AMAX_FLOOR, dtype=tl.float32)
    for s0 in range(0, S, BLK_S):
        offs_s = s0 + tl.arange(0, BLK_S)
        mask = offs_s < S
        x = tl.load(base + offs_s * stride_ss, mask=mask, other=0.0).to(tl.float32)
        amax = tl.maximum(amax, tl.max(tl.abs(x)))
    sc = amax / _FP8_E4M3_MAX
    tl.store(scale + pid_b * stride_scb + pid_h * stride_sch + pid_d * stride_scd, sc)
    dst_base = dst + pid_b * stride_db + pid_h * stride_dh + pid_d * stride_dd
    for s0 in range(0, S, BLK_S):
        offs_s = s0 + tl.arange(0, BLK_S)
        mask = offs_s < S
        x = tl.load(base + offs_s * stride_ss, mask=mask, other=0.0).to(tl.float32)
        xq = tl.minimum(tl.maximum(x / sc, -_FP8_E4M3_MAX), _FP8_E4M3_MAX)
        tl.store(dst_base + offs_s * stride_ds, xq.to(dst.dtype.element_ty), mask=mask)


def fp8_per_channel_quant(t: torch.Tensor, block_s: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
    """FP32-scale FP8 e4m3fn quantization for V, one scale per (B, H, D)."""
    b, s, h, d = _check_nhd(t)
    t = t.contiguous()
    out = torch.empty_like(t, dtype=torch.float8_e4m3fn)
    scale = torch.empty((b, h, d), dtype=torch.float32, device=t.device)
    _fp8_channel_quant_kernel[(b, h, d)](
        t, out, scale,
        t.stride(0), t.stride(1), t.stride(2), t.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        scale.stride(0), scale.stride(1), scale.stride(2),
        S=s, BLK_S=block_s,
    )
    return out, scale


@triton.jit
def _fp8_channel_dequant_kernel(
    src, scale, dst,
    stride_sb, stride_ss, stride_sh, stride_sd,
    stride_scb, stride_sch, stride_scd,
    stride_db, stride_ds, stride_dh, stride_dd,
    S: tl.constexpr, BLK_S: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_d = tl.program_id(2)
    sc = tl.load(scale + pid_b * stride_scb + pid_h * stride_sch + pid_d * stride_scd)
    src_base = src + pid_b * stride_sb + pid_h * stride_sh + pid_d * stride_sd
    dst_base = dst + pid_b * stride_db + pid_h * stride_dh + pid_d * stride_dd
    for s0 in range(0, S, BLK_S):
        offs_s = s0 + tl.arange(0, BLK_S)
        mask = offs_s < S
        x = tl.load(src_base + offs_s * stride_ss, mask=mask, other=0.0).to(tl.float32)
        tl.store(dst_base + offs_s * stride_ds, x * sc, mask=mask)


def fp8_per_channel_dequant(
    fp8: torch.Tensor,
    scale: torch.Tensor,
    block_s: int = 128,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    b, s, h, d = _check_nhd(fp8, name="fp8")
    if tuple(scale.shape) != (b, h, d):
        raise ValueError(f"scale must have shape {(b, h, d)}; got {tuple(scale.shape)}")
    out = torch.empty((b, s, h, d), dtype=dtype, device=fp8.device)
    _fp8_channel_dequant_kernel[(b, h, d)](
        fp8, scale, out,
        fp8.stride(0), fp8.stride(1), fp8.stride(2), fp8.stride(3),
        scale.stride(0), scale.stride(1), scale.stride(2),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        S=s, BLK_S=block_s,
    )
    return out


@triton.jit
def _mxfp8_qk_quant_kernel(
    src, dst, scale,
    stride_sb, stride_ss, stride_sh, stride_sd,
    stride_db, stride_ds, stride_dh, stride_dd,
    stride_scb, stride_scs, stride_sch, stride_scd,
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_db = tl.program_id(3)
    offs_d = pid_db * BLOCK_D + tl.arange(0, BLOCK_D)
    x = tl.load(
        src + pid_b * stride_sb + pid_s * stride_ss + pid_h * stride_sh + offs_d * stride_sd
    ).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(x)), _AMAX_FLOOR)
    log2_scale = tl.ceil(tl.log2(amax / _FP8_E4M3_MAX))
    log2_scale = tl.minimum(tl.maximum(log2_scale, _UE8M0_EXP_MIN), _UE8M0_EXP_MAX)
    sc = tl.exp2(log2_scale)
    xq = tl.minimum(tl.maximum(x / sc, -_FP8_E4M3_MAX), _FP8_E4M3_MAX)
    tl.store(
        dst + pid_b * stride_db + pid_s * stride_ds + pid_h * stride_dh + offs_d * stride_dd,
        xq.to(dst.dtype.element_ty),
    )
    tl.store(
        scale + pid_b * stride_scb + pid_s * stride_scs + pid_h * stride_sch + pid_db * stride_scd,
        sc,
    )


def mxfp8_qk_quant(t: torch.Tensor, block_d: int = 32) -> Tuple[torch.Tensor, torch.Tensor]:
    """MXFP8-style Q/K quantization along D, scale shape (B, S, H, D/block_d)."""
    b, s, h, d = _check_nhd(t)
    if d % block_d != 0:
        raise ValueError(f"D={d} must be divisible by block_d={block_d}")
    t = t.contiguous()
    out = torch.empty_like(t, dtype=torch.float8_e4m3fn)
    scale = torch.empty((b, s, h, d // block_d), dtype=torch.float32, device=t.device)
    _mxfp8_qk_quant_kernel[(b, s, h, d // block_d)](
        t, out, scale,
        t.stride(0), t.stride(1), t.stride(2), t.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        scale.stride(0), scale.stride(1), scale.stride(2), scale.stride(3),
        BLOCK_D=block_d,
    )
    return out, scale


@triton.jit
def _mxfp8_qk_dequant_kernel(
    src, scale, dst,
    stride_sb, stride_ss, stride_sh, stride_sd,
    stride_scb, stride_scs, stride_sch, stride_scd,
    stride_db, stride_ds, stride_dh, stride_dd,
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_db = tl.program_id(3)
    offs_d = pid_db * BLOCK_D + tl.arange(0, BLOCK_D)
    sc = tl.load(
        scale + pid_b * stride_scb + pid_s * stride_scs + pid_h * stride_sch + pid_db * stride_scd
    )
    x = tl.load(
        src + pid_b * stride_sb + pid_s * stride_ss + pid_h * stride_sh + offs_d * stride_sd
    ).to(tl.float32)
    tl.store(
        dst + pid_b * stride_db + pid_s * stride_ds + pid_h * stride_dh + offs_d * stride_dd,
        x * sc,
    )


def mxfp8_qk_dequant(
    fp8: torch.Tensor,
    scale: torch.Tensor,
    block_d: int = 32,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    b, s, h, d = _check_nhd(fp8, name="fp8")
    if tuple(scale.shape) != (b, s, h, d // block_d):
        raise ValueError(f"scale must have shape {(b, s, h, d // block_d)}")
    out = torch.empty((b, s, h, d), dtype=dtype, device=fp8.device)
    _mxfp8_qk_dequant_kernel[(b, s, h, d // block_d)](
        fp8, scale, out,
        fp8.stride(0), fp8.stride(1), fp8.stride(2), fp8.stride(3),
        scale.stride(0), scale.stride(1), scale.stride(2), scale.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        BLOCK_D=block_d,
    )
    return out


@triton.jit
def _mxfp8_v_quant_kernel(
    src, dst, scale,
    stride_sb, stride_ss, stride_sh, stride_sd,
    stride_db, stride_ds, stride_dh, stride_dd,
    stride_scb, stride_scs, stride_sch, stride_scd,
    BLOCK_S: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_sb = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_d = tl.program_id(3)
    offs_s = pid_sb * BLOCK_S + tl.arange(0, BLOCK_S)
    x = tl.load(
        src + pid_b * stride_sb + offs_s * stride_ss + pid_h * stride_sh + pid_d * stride_sd
    ).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(x)), _AMAX_FLOOR)
    log2_scale = tl.ceil(tl.log2(amax / _FP8_E4M3_MAX))
    log2_scale = tl.minimum(tl.maximum(log2_scale, _UE8M0_EXP_MIN), _UE8M0_EXP_MAX)
    sc = tl.exp2(log2_scale)
    xq = tl.minimum(tl.maximum(x / sc, -_FP8_E4M3_MAX), _FP8_E4M3_MAX)
    tl.store(
        dst + pid_b * stride_db + offs_s * stride_ds + pid_h * stride_dh + pid_d * stride_dd,
        xq.to(dst.dtype.element_ty),
    )
    tl.store(
        scale + pid_b * stride_scb + pid_sb * stride_scs + pid_h * stride_sch + pid_d * stride_scd,
        sc,
    )


def mxfp8_v_quant(t: torch.Tensor, block_s: int = 32) -> Tuple[torch.Tensor, torch.Tensor]:
    """MXFP8-style V quantization along S, scale shape (B, S/block_s, H, D)."""
    b, s, h, d = _check_nhd(t)
    if s % block_s != 0:
        raise ValueError(f"S={s} must be divisible by block_s={block_s}")
    t = t.contiguous()
    out = torch.empty_like(t, dtype=torch.float8_e4m3fn)
    scale = torch.empty((b, s // block_s, h, d), dtype=torch.float32, device=t.device)
    _mxfp8_v_quant_kernel[(b, s // block_s, h, d)](
        t, out, scale,
        t.stride(0), t.stride(1), t.stride(2), t.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        scale.stride(0), scale.stride(1), scale.stride(2), scale.stride(3),
        BLOCK_S=block_s,
    )
    return out, scale


@triton.jit
def _mxfp8_v_dequant_kernel(
    src, scale, dst,
    stride_sb, stride_ss, stride_sh, stride_sd,
    stride_scb, stride_scs, stride_sch, stride_scd,
    stride_db, stride_ds, stride_dh, stride_dd,
    BLOCK_S: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_sb = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_d = tl.program_id(3)
    offs_s = pid_sb * BLOCK_S + tl.arange(0, BLOCK_S)
    sc = tl.load(
        scale + pid_b * stride_scb + pid_sb * stride_scs + pid_h * stride_sch + pid_d * stride_scd
    )
    x = tl.load(
        src + pid_b * stride_sb + offs_s * stride_ss + pid_h * stride_sh + pid_d * stride_sd
    ).to(tl.float32)
    tl.store(
        dst + pid_b * stride_db + offs_s * stride_ds + pid_h * stride_dh + pid_d * stride_dd,
        x * sc,
    )


def mxfp8_v_dequant(
    fp8: torch.Tensor,
    scale: torch.Tensor,
    block_s: int = 32,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    b, s, h, d = _check_nhd(fp8, name="fp8")
    if tuple(scale.shape) != (b, s // block_s, h, d):
        raise ValueError(f"scale must have shape {(b, s // block_s, h, d)}")
    out = torch.empty((b, s, h, d), dtype=dtype, device=fp8.device)
    _mxfp8_v_dequant_kernel[(b, s // block_s, h, d)](
        fp8, scale, out,
        fp8.stride(0), fp8.stride(1), fp8.stride(2), fp8.stride(3),
        scale.stride(0), scale.stride(1), scale.stride(2), scale.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        BLOCK_S=block_s,
    )
    return out
