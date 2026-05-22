"""FlashAttention-2 style Triton kernel with multiple V quant + P quant pairings.

V quant variants (selected by ``V_QUANT_KIND`` constexpr):
  0 = ``fp8_channel``: V passed in as FP8-cast-to-BF16 (raw FP8 values in BF16
      slots, no scale applied). Per (B, H, D) FP32 scale is applied to the
      output ``acc`` post-loop.
  1 = ``fp8_block``: V passed in as FP8-cast-to-BF16 too, but a per-K-block
      FP32 scalar ``s_V[blk]`` (shape ``(B, S/v_block, H)``) is multiplied
      into the per-block PV contribution INSIDE the K loop.
  2 = ``mxfp8``: V passed in as pre-dequantized BF16 (per-(S/blk, D) UE8M0
      scales already absorbed into the V values). Post-mul scale is unity.

P quant variants (selected by ``P_QUANT_KIND`` constexpr):
  0 = ``elementwise``: ``p_fp8 = p.to(e4m3fn)``; element-wise cast with the
      global ``p_max_offset`` ensuring P fits in FP8 range. Used for
      fp8_channel and fp8_block V (V's scale structure isn't tied to P).
  1 = ``mx``: per-(M-row, N-block) UE8M0 scale on P before the e4m3 cast,
      matching V's mxfp8 microscaling structure so the conceptual PV mma
      stays MXFP8 × MXFP8 compatible.

The qm correction (un-quantized K_smooth + FP32 qm) is preserved and
independent of V/P quant kind.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_LOG2E = 1.4426950408889634
_FP8_E4M3_MAX = 448.0
_UE8M0_EXP_MIN = -127.0
_UE8M0_EXP_MAX = 127.0


@triton.jit
def _fake_quant_attn_fwd_kernel(
    Q, K, V, V_SCALE, V_ALPHA, K_SMOOTH, QM, V_BLOCK_SCALE, O,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_vsz, stride_vsh, stride_vsd,
    stride_vaz, stride_van, stride_vah, stride_vad,
    stride_ksz, stride_ksh, stride_ksn, stride_ksk,
    stride_qmz, stride_qmg, stride_qmh, stride_qmd,
    stride_vbsz, stride_vbsn, stride_vbsh,
    stride_oz, stride_oh, stride_om, stride_ok,
    H: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    sm_scale,
    p_max_offset,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    D: tl.constexpr,
    HAS_V_ALPHA: tl.constexpr,
    V_SMOOTH_BLOCK: tl.constexpr,
    HAS_QM: tl.constexpr,
    Q_SMOOTH_BLOCK: tl.constexpr,
    V_QUANT_KIND: tl.constexpr,        # 0=channel, 1=fp8_block, 2=mxfp8
    V_BLOCK_QUANT_SIZE: tl.constexpr,  # only used if V_QUANT_KIND==1
    P_QUANT_KIND: tl.constexpr,        # 0=elementwise, 1=mx
    P_MX_BLOCK_N: tl.constexpr,        # MX P block along N (only if P=mx)
):
    LOG2E: tl.constexpr = 1.4426950408889634
    FP8_MAX: tl.constexpr = 448.0
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    q_off = off_z * stride_qz + off_h * stride_qh
    k_off = off_z * stride_kz + off_h * stride_kh
    v_off = off_z * stride_vz + off_h * stride_vh
    o_off = off_z * stride_oz + off_h * stride_oh
    vs_off = off_z * stride_vsz + off_h * stride_vsh
    va_off = off_z * stride_vaz + off_h * stride_vah
    ks_off = off_z * stride_ksz + off_h * stride_ksh
    qm_off = off_z * stride_qmz + off_h * stride_qmh
    vbs_off = off_z * stride_vbsz + off_h * stride_vbsh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)

    # Load Q block (BLOCK_M, D) BF16 — the FP8-cast-and-dequantized Q_centered.
    q_ptrs = Q + q_off + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    q_mask = offs_m[:, None] < M
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    # Per-channel V scale (D,) FP32 — used only when V_QUANT_KIND==0.
    v_sc_ptrs = V_SCALE + vs_off + offs_d * stride_vsd
    v_sc = tl.load(v_sc_ptrs)

    if HAS_QM:
        qm_group = (start_m * BLOCK_M) // Q_SMOOTH_BLOCK
        qm_ptrs = QM + qm_off + qm_group * stride_qmg + offs_d * stride_qmd
        qm_vec = tl.load(qm_ptrs)  # (D,) FP32

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    c_acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)  # V-smoothing correction

    for start_n in range(0, N, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        col_mask = offs_n < N

        k_ptrs = K + k_off + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
        v_ptrs = V + v_off + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
        k_v_mask = col_mask[:, None]
        k = tl.load(k_ptrs, mask=k_v_mask, other=0.0)
        v = tl.load(v_ptrs, mask=k_v_mask, other=0.0)

        s_ij = tl.dot(q, tl.trans(k)).to(tl.float32) * sm_scale

        if HAS_QM:
            ks_ptrs = K_SMOOTH + ks_off + offs_n[:, None] * stride_ksn + offs_d[None, :] * stride_ksk
            k_smooth = tl.load(ks_ptrs, mask=k_v_mask, other=0.0).to(tl.float32)
            corr_n = tl.sum(qm_vec[None, :] * k_smooth, axis=1) * sm_scale  # (BLOCK_N,)
            s_ij = s_ij + corr_n[None, :]

        s_ij = tl.where(col_mask[None, :], s_ij, float("-inf"))

        m_ij = tl.maximum(m_i, tl.max(s_ij, 1))
        alpha = tl.exp2((m_i - m_ij) * LOG2E)
        z = (s_ij - m_ij[:, None]) * LOG2E + p_max_offset
        p = tl.exp2(z)
        p = tl.where(col_mask[None, :], p, 0.0)

        l_i = l_i * alpha + tl.sum(p, 1)

        # --- P quantization, dispatched by P_QUANT_KIND ---
        if P_QUANT_KIND == 0:
            # Element-wise e4m3fn cast (p_max_offset already keeps p in range).
            p_bf16 = p.to(tl.float8e4nv).to(tl.bfloat16)
        else:
            # MX: per (M-row, N-sub-block of P_MX_BLOCK_N) UE8M0 scale on P,
            # then cast e4m3. The mma is conceptually MXFP8 × MXFP8.
            # Compute amax per (row, P_MX_BLOCK_N sub-block). Here we use
            # one scale for the whole BLOCK_N (so P_MX_BLOCK_N should be a
            # multiple of BLOCK_N or equal to BLOCK_N; enforced at call).
            row_amax = tl.max(tl.abs(p), 1)  # (BLOCK_M,)
            row_amax = tl.maximum(row_amax, 1e-4)
            log2_scale = tl.ceil(tl.log2(row_amax / FP8_MAX))
            log2_scale = tl.minimum(tl.maximum(log2_scale, -127.0), 127.0)
            s_P = tl.exp2(log2_scale)  # (BLOCK_M,)
            p_scaled = p / s_P[:, None]
            p_scaled = tl.minimum(tl.maximum(p_scaled, -FP8_MAX), FP8_MAX)
            p_fp8 = p_scaled.to(tl.float8e4nv).to(tl.float32)
            p_recovered = p_fp8 * s_P[:, None]
            p_bf16 = p_recovered.to(tl.bfloat16)

        # --- PV accumulation, dispatched by V_QUANT_KIND ---
        pv = tl.dot(p_bf16, v).to(tl.float32)

        if V_QUANT_KIND == 1:
            # fp8_block: load per-K-block FP32 scalar s_V and multiply this
            # block's pv contribution. The K-block we're processing starts
            # at start_n; we need start_n // V_BLOCK_QUANT_SIZE.
            v_blk_q_idx = start_n // V_BLOCK_QUANT_SIZE
            vbs_ptr = V_BLOCK_SCALE + vbs_off + v_blk_q_idx * stride_vbsn
            s_V_blk = tl.load(vbs_ptr)  # scalar FP32
            pv = pv * s_V_blk
        # else (kind 0 or 2): no per-block scale needed here.
        # Kind 0 applies per-D scale post-loop. Kind 2 absorbed scale into V.

        acc = acc * alpha[:, None] + pv

        if HAS_V_ALPHA:
            v_blk_idx = start_n // V_SMOOTH_BLOCK
            va_ptrs = V_ALPHA + va_off + v_blk_idx * stride_van + offs_d * stride_vad
            alpha_v = tl.load(va_ptrs)  # (D,) FP32
            rowsum_p_fp8 = tl.sum(p_bf16.to(tl.float32), 1)
            c_acc = c_acc * alpha[:, None] + rowsum_p_fp8[:, None] * alpha_v[None, :]

        m_i = m_ij

    inv_l = 1.0 / l_i[:, None]
    # Apply per-D post-mul scale only for V=fp8_channel (kind 0). For kind 1
    # the per-block scale was multiplied inside the loop; for kind 2 nothing
    # extra is needed.
    if V_QUANT_KIND == 0:
        out_main = acc * v_sc[None, :]
    else:
        out_main = acc

    if HAS_V_ALPHA:
        out_acc = (out_main + c_acc) * inv_l
    else:
        out_acc = out_main * inv_l

    o_ptrs = O + o_off + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(o_ptrs, out_acc.to(O.dtype.element_ty), mask=q_mask)


def fake_quant_attention_triton(
    q_bhsd: torch.Tensor,
    k_bhsd: torch.Tensor,
    v_bhsd_bf16: torch.Tensor,
    v_scale_bhd: torch.Tensor,
    *,
    sm_scale: float,
    p_max_offset: int,
    block_m: int = 64,
    block_n: int = 64,
    v_alpha: torch.Tensor | None = None,
    v_smooth_block: int = 0,
    k_smooth_bhsd: torch.Tensor | None = None,
    qm_bhgd: torch.Tensor | None = None,
    q_smooth_block: int = 0,
    v_block_scale_bsh: torch.Tensor | None = None,
    v_block_size: int = 0,
    p_quant_mode: str = "elementwise",
    p_mx_block_n: int = 0,
) -> torch.Tensor:
    """FA2-style fake-quant attention kernel.

    Three V quant kinds (selected implicitly by which scale tensor you pass):
      * ``v_block_scale_bsh=None``, V values come in as either FP8-cast-bf16
        or pre-dequantized-bf16. ``v_scale_bhd`` is the per-D post-mul
        scale (use FP32 ones for the pre-dequantized case).
      * ``v_block_scale_bsh`` given → ``fp8_block``: per-(B, S/v_block_size, H)
        FP32 scalar applied inside the K loop. v_scale_bhd must be ones.

    P quant mode:
      * ``elementwise`` (default): plain e4m3fn cast on P.
      * ``mx``: per-K-block UE8M0 scale on P before the e4m3 cast. Match
        ``p_mx_block_n`` with V's mxfp8 block size.
    """
    B, H, S, D = q_bhsd.shape
    assert k_bhsd.shape == (B, H, S, D)
    assert v_bhsd_bf16.shape == (B, H, S, D)
    assert v_scale_bhd.shape == (B, H, D)
    assert q_bhsd.dtype == torch.bfloat16
    assert k_bhsd.dtype == torch.bfloat16
    assert v_bhsd_bf16.dtype == torch.bfloat16
    assert v_scale_bhd.dtype == torch.float32

    q_bhsd = q_bhsd.contiguous()
    k_bhsd = k_bhsd.contiguous()
    v_bhsd_bf16 = v_bhsd_bf16.contiguous()
    v_scale_bhd = v_scale_bhd.contiguous()

    has_v_alpha = v_alpha is not None
    if has_v_alpha:
        if v_smooth_block <= 0:
            raise ValueError("v_smooth_block must be > 0 when v_alpha is given")
        if v_smooth_block < block_n or v_smooth_block % block_n != 0:
            raise ValueError(
                f"v_smooth_block ({v_smooth_block}) must be >= block_n ({block_n}) "
                "and an integer multiple of block_n"
            )
        if S % v_smooth_block != 0:
            raise ValueError(f"S ({S}) must be divisible by v_smooth_block ({v_smooth_block})")
        assert v_alpha.shape == (B, S // v_smooth_block, H, D)
        assert v_alpha.dtype == torch.float32
        v_alpha = v_alpha.contiguous()
        va_strides = v_alpha.stride()
    else:
        v_alpha = torch.empty(1, dtype=torch.float32, device=q_bhsd.device)
        va_strides = (0, 0, 0, 0)

    has_qm = qm_bhgd is not None
    if has_qm:
        if k_smooth_bhsd is None:
            raise ValueError("k_smooth_bhsd required with qm_bhgd")
        if q_smooth_block <= 0:
            raise ValueError("q_smooth_block must be > 0 with qm_bhgd")
        if q_smooth_block < block_m or q_smooth_block % block_m != 0:
            raise ValueError(
                f"q_smooth_block ({q_smooth_block}) must be >= block_m ({block_m}) "
                "and an integer multiple of block_m"
            )
        if S % q_smooth_block != 0:
            raise ValueError(f"S ({S}) must be divisible by q_smooth_block ({q_smooth_block})")
        assert k_smooth_bhsd.shape == (B, H, S, D)
        assert k_smooth_bhsd.dtype == torch.bfloat16
        assert qm_bhgd.shape == (B, S // q_smooth_block, H, D)
        assert qm_bhgd.dtype == torch.float32
        k_smooth_bhsd = k_smooth_bhsd.contiguous()
        qm_bhgd = qm_bhgd.contiguous()
        ks_strides = k_smooth_bhsd.stride()
        qm_strides = qm_bhgd.stride()
    else:
        k_smooth_bhsd = torch.empty(1, dtype=torch.bfloat16, device=q_bhsd.device)
        qm_bhgd = torch.empty(1, dtype=torch.float32, device=q_bhsd.device)
        ks_strides = (0, 0, 0, 0)
        qm_strides = (0, 0, 0, 0)

    # V quant kind
    if v_block_scale_bsh is not None:
        v_quant_kind = 1  # fp8_block
        if v_block_size <= 0:
            raise ValueError("v_block_size must be > 0 when v_block_scale_bsh is given")
        if v_block_size < block_n or v_block_size % block_n != 0:
            raise ValueError(
                f"v_block_size ({v_block_size}) must be >= block_n ({block_n}) "
                "and an integer multiple of block_n"
            )
        if S % v_block_size != 0:
            raise ValueError(f"S ({S}) must be divisible by v_block_size ({v_block_size})")
        assert v_block_scale_bsh.shape == (B, S // v_block_size, H)
        assert v_block_scale_bsh.dtype == torch.float32
        v_block_scale_bsh = v_block_scale_bsh.contiguous()
        vbs_strides = v_block_scale_bsh.stride()
    else:
        # 0 (fp8_channel) or 2 (mxfp8). The kernel distinguishes purely by
        # whether v_scale_bhd is meaningful (kind 0) vs unity (kind 2). For
        # simplicity we use kind 0 in both — when v_scale_bhd is unity (as
        # we set up for mxfp8), the post-mul is a no-op and the per-block
        # branch isn't taken either. P quant mode disambiguates.
        v_quant_kind = 0
        v_block_scale_bsh = torch.empty(1, dtype=torch.float32, device=q_bhsd.device)
        vbs_strides = (0, 0, 0)
        v_block_size = block_n  # placeholder

    # P quant mode
    if p_quant_mode == "elementwise":
        p_quant_kind = 0
        p_mx_block_n_arg = block_n  # placeholder
    elif p_quant_mode == "mx":
        p_quant_kind = 1
        if p_mx_block_n <= 0:
            raise ValueError("p_mx_block_n must be > 0 when p_quant_mode='mx'")
        if p_mx_block_n != block_n:
            raise ValueError(
                f"p_mx_block_n ({p_mx_block_n}) currently must equal block_n ({block_n}); "
                "sub-block scaling within a K block is not implemented"
            )
        p_mx_block_n_arg = p_mx_block_n
    else:
        raise ValueError(f"unknown p_quant_mode: {p_quant_mode!r}")

    o = torch.empty_like(q_bhsd)
    grid = (triton.cdiv(S, block_m), B * H)
    _fake_quant_attn_fwd_kernel[grid](
        q_bhsd, k_bhsd, v_bhsd_bf16, v_scale_bhd, v_alpha,
        k_smooth_bhsd, qm_bhgd, v_block_scale_bsh, o,
        q_bhsd.stride(0), q_bhsd.stride(1), q_bhsd.stride(2), q_bhsd.stride(3),
        k_bhsd.stride(0), k_bhsd.stride(1), k_bhsd.stride(2), k_bhsd.stride(3),
        v_bhsd_bf16.stride(0), v_bhsd_bf16.stride(1), v_bhsd_bf16.stride(2), v_bhsd_bf16.stride(3),
        v_scale_bhd.stride(0), v_scale_bhd.stride(1), v_scale_bhd.stride(2),
        va_strides[0], va_strides[1], va_strides[2], va_strides[3],
        ks_strides[0], ks_strides[1], ks_strides[2], ks_strides[3],
        qm_strides[0], qm_strides[1], qm_strides[2], qm_strides[3],
        vbs_strides[0], vbs_strides[1], vbs_strides[2],
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        H=H, M=S, N=S,
        sm_scale=sm_scale,
        p_max_offset=p_max_offset,
        BLOCK_M=block_m, BLOCK_N=block_n,
        D=D,
        HAS_V_ALPHA=has_v_alpha,
        V_SMOOTH_BLOCK=v_smooth_block if has_v_alpha else block_n,
        HAS_QM=has_qm,
        Q_SMOOTH_BLOCK=q_smooth_block if has_qm else block_m,
        V_QUANT_KIND=v_quant_kind,
        V_BLOCK_QUANT_SIZE=v_block_size,
        P_QUANT_KIND=p_quant_kind,
        P_MX_BLOCK_N=p_mx_block_n_arg,
        num_warps=4,
        num_stages=2,
    )
    return o


__all__ = ["fake_quant_attention_triton"]
