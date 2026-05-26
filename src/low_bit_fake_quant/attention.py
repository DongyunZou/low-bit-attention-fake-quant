"""Fake-quant attention kernel.

Two execution paths share the same quant/preprocess pipeline:

1. ``cfg.p_requant = False`` — Q/K/V are quantized then dequantized back to
   BF16/FP16 and fed to ``torch.nn.functional.scaled_dot_product_attention``.
   Captures Q/K/V cast errors but P stays in FP32. Acts as the "upper bound"
   reference in ``docs/quant_precision_test_plan.md``.

2. ``cfg.p_requant = True`` — chunked-Q attention that materializes the
   per-Q-block score matrix, applies ``P = exp2((s - row_max) * sm_scale *
   log2(e) + p_max_offset)``, casts the resulting P to ``float8_e4m3fn``,
   reads it back to FP32, and does the PV matmul against the *FP8* V with the
   per-channel scale applied **outside** the matmul (matching the real H100
   FP8 attention kernel's compute path: ``PV_acc[m,d] = sum_n P_fp8 * V_fp8;
   O[m,d] = PV_acc * v_scale[d] / row_sum[m]``). This models the four-source
   quant error (Q, K, V, P) and the per-channel V scale post-multiply
   explicitly.

The streaming attention never materializes the full ``(B, H, S, S)`` score
matrix — only ``(B, H, q_chunk, S)`` per Q chunk, so the wan21 workload
(S=69120, H=40, D=128) fits comfortably on a single H100 80GB.

Supported quant strategies
--------------------------
- ``cfg.qk_quant = "fp8_block"`` — FP32-scale block FP8 e4m3fn, one
  scale per ``(B, S/block_s, H)``.
- ``cfg.qk_quant = "mxfp8"``    — MXFP8 power-of-two scale along D, one
  scale per ``(B, S, H, D/block_d)``.
- ``cfg.v_quant = "fp8_channel"`` — FP32-scale FP8 e4m3fn per ``(B, H, D)``.
- ``cfg.v_quant = "fp8_block"``   — FP32-scale FP8 e4m3fn per S-block.
- ``cfg.v_quant = "mxfp8"``       — MXFP8 power-of-two scale along S.
- ``cfg.smoothing = "off" | "k_only" | "full"`` — SageAttention-style
  K-mean smoothing, optionally with grouped Q centering.
- ``cfg.q_kmeans_k = 32 | 64 | None`` — Q-token reorder by k-means,
  inverse-reordered after attention.
- ``cfg.p_requant`` toggles the P FP8 cast.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from .config import QuantConfig
from .kmeans import (
    KMeansReorderResult,
    apply_kv_permutation,
    q_kmeans_reorder,
    v_kmeans_reorder,
)
from .preprocess import group_mean_q, smooth_k, smooth_v_per_block
from .attention_triton import fake_quant_attention_triton
from .quant_triton import (
    fp8_block_dequant,
    fp8_block_quant,
    fp8_per_channel_dequant,
    fp8_per_channel_quant,
    mxfp8_qk_dequant,
    mxfp8_qk_quant,
    mxfp8_v_dequant,
    mxfp8_v_quant,
)


_FP8_E4M3_MAX = 448.0
_LOG2_E = 1.4426950408889634


@dataclass
class FakeQuantArtifacts:
    """Side data emitted by :func:`fake_quant_attention` for debugging."""

    qm: Optional[torch.Tensor] = None
    k_mean: Optional[torch.Tensor] = None
    kmeans: Optional[KMeansReorderResult] = None
    q_eff_dtype: Optional[torch.dtype] = None
    p_requant: bool = False


def _quant_qk(t: torch.Tensor, cfg: QuantConfig) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """Quantize a Q or K tensor; return ``(fp8_tensor, scale, meta)``."""
    if cfg.qk_quant == "fp8_block":
        q, s = fp8_block_quant(t, block_s=cfg.fp8_block_size)
        return q, s, {"kind": "fp8_block", "block_s": cfg.fp8_block_size}
    if cfg.qk_quant == "mxfp8":
        q, s = mxfp8_qk_quant(t, block_d=cfg.mxfp8_block_size)
        return q, s, {"kind": "mxfp8", "block_d": cfg.mxfp8_block_size}
    raise ValueError(f"unsupported qk_quant: {cfg.qk_quant!r}")


def _dequant_qk(fp8: torch.Tensor, scale: torch.Tensor, meta: dict, dtype: torch.dtype) -> torch.Tensor:
    if meta["kind"] == "fp8_block":
        return fp8_block_dequant(fp8, scale, block_s=meta["block_s"], dtype=dtype)
    if meta["kind"] == "mxfp8":
        return mxfp8_qk_dequant(fp8, scale, block_d=meta["block_d"], dtype=dtype)
    raise ValueError(f"bad meta: {meta!r}")


def _quant_v(t: torch.Tensor, cfg: QuantConfig) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """Quantize V; return ``(fp8_tensor, scale, meta)``.

    Three schemes (see :data:`config.VQuant`):
      - ``fp8_channel``: one FP32 scale per ``(B, H, D)``.
      - ``fp8_block``: one FP32 scalar per ``(B, S/block, H)`` — matches
        Algorithm 2's ``phi(V_j)`` where each K/V block has its own scale.
      - ``mxfp8``: per ``(B, S/block, H, D)`` UE8M0 power-of-2 scale —
        full microscaling along both S and D.
    """
    if cfg.v_quant == "fp8_channel":
        q, s = fp8_per_channel_quant(t)
        return q, s, {"kind": "fp8_channel"}
    if cfg.v_quant == "fp8_block":
        # Reuse the QK fp8_block kernel — it already produces one FP32 scale
        # per (B, S/block, H), which is exactly the algorithm's phi(V_j).
        q, s = fp8_block_quant(t, block_s=cfg.v_fp8_block_size)
        return q, s, {"kind": "fp8_block", "block_s": cfg.v_fp8_block_size}
    if cfg.v_quant == "mxfp8":
        q, s = mxfp8_v_quant(t, block_s=cfg.v_mxfp8_block_size)
        return q, s, {"kind": "mxfp8", "block_s": cfg.v_mxfp8_block_size}
    raise ValueError(f"unsupported v_quant: {cfg.v_quant!r}")


def _dequant_v(fp8: torch.Tensor, scale: torch.Tensor, meta: dict, dtype: torch.dtype) -> torch.Tensor:
    if meta["kind"] == "fp8_channel":
        return fp8_per_channel_dequant(fp8, scale, dtype=dtype)
    if meta["kind"] == "fp8_block":
        return fp8_block_dequant(fp8, scale, block_s=meta["block_s"], dtype=dtype)
    if meta["kind"] == "mxfp8":
        return mxfp8_v_dequant(fp8, scale, block_s=meta["block_s"], dtype=dtype)
    raise ValueError(f"bad meta: {meta!r}")


def _broadcast_qm(qm: torch.Tensor, target_s: int) -> torch.Tensor:
    """Tile ``qm`` of shape (B, n, H, D) to (B, S, H, D)."""
    b, n, h, d = qm.shape
    if target_s % n != 0:
        raise ValueError(f"qm groups {n} do not evenly tile S={target_s}")
    block_q = target_s // n
    return qm.repeat_interleave(block_q, dim=1)


def _preprocess(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cfg: QuantConfig,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor,
    Optional[torch.Tensor], Optional[torch.Tensor],
    Optional[KMeansReorderResult], Optional[KMeansReorderResult],
]:
    """Run Q kmeans + Q/K smoothing + V kmeans (co-permute K) + V smoothing.

    Returns ``(q_work, k_work, v_work, qm, v_alpha, q_kmeans, v_kmeans)``.

    Order of operations is deliberate:
      1. Q kmeans reorder (permutes Q only — Q's S axis is independent).
      2. K smoothing (`K -= mean(K, S)`) — invariant under any S-permutation.
      3. Q smoothing (`Q -= group_mean_q(Q)`) on the reordered Q.
      4. V kmeans reorder → permutation π_v. We co-permute ``k_work`` (the
         smoothed un-quantized K) by π_v so that the (K, V) row pairing is
         preserved; attention output is mathematically unchanged. We do
         NOT permute Q (Q's S is independent of K/V's S).
      5. V smoothing (per-block mean) on the reordered V.
    """
    q_work = q.contiguous()
    k_work = k.contiguous()
    v_work = v.contiguous()

    # 1. Q kmeans
    q_kmeans: Optional[KMeansReorderResult] = None
    if cfg.q_kmeans_k is not None:
        q_kmeans = q_kmeans_reorder(
            q_work,
            n_clusters=cfg.q_kmeans_k,
            max_iters=cfg.q_kmeans_iters,
            seed=cfg.q_kmeans_seed,
        )
        q_work = q_kmeans.tensor_reordered

    # 2 + 3. Q/K smoothing.
    qm: Optional[torch.Tensor] = None
    if cfg.smoothing == "off":
        pass
    elif cfg.smoothing == "k_only":
        k_work, _ = smooth_k(k_work)
    elif cfg.smoothing == "full":
        k_work, _ = smooth_k(k_work)
        q_work, qm = group_mean_q(q_work, block_q=cfg.q_smooth_block_size)
    else:
        raise ValueError(f"unsupported smoothing: {cfg.smoothing!r}")

    # 4. V kmeans (after K smoothing so the co-permuted K stays smoothed).
    v_kmeans: Optional[KMeansReorderResult] = None
    if cfg.v_kmeans_k is not None:
        v_kmeans = v_kmeans_reorder(
            v_work,
            n_clusters=cfg.v_kmeans_k,
            max_iters=cfg.v_kmeans_iters,
            seed=cfg.v_kmeans_seed,
        )
        v_work = v_kmeans.tensor_reordered
        # Co-permute K so each (K[n], V[n]) pair stays matched.
        k_work = apply_kv_permutation(k_work, v_kmeans.order)

    # 5. V smoothing on (possibly reordered) V.
    v_alpha: Optional[torch.Tensor] = None
    if cfg.v_smooth_mode == "off":
        pass
    elif cfg.v_smooth_mode == "per_block":
        v_work, v_alpha = smooth_v_per_block(v_work, block_s=cfg.v_smooth_block_size)
    else:
        raise ValueError(f"unsupported v_smooth_mode: {cfg.v_smooth_mode!r}")

    return q_work, k_work, v_work, qm, v_alpha, q_kmeans, v_kmeans


# ---------------------------------------------------------------------------
# Path A: dequant Q/K/V → SDPA (no P requant). Upper-bound reference.
# ---------------------------------------------------------------------------


def _fake_quant_attention_sdpa(
    q_work: torch.Tensor,
    k_work: torch.Tensor,
    v_work: torch.Tensor,
    qm: Optional[torch.Tensor],
    v_alpha: Optional[torch.Tensor],
    cfg: QuantConfig,
    sm_scale: float,
    sdpa_dtype: torch.dtype,
) -> torch.Tensor:
    """SDPA-equivalent reference path. P stays in FP32 (upper bound).

    Score model (matching Sage2's CuTe-DSL kernel — critical for the qm
    correction to actually pay off):

        score[m, n] = Q_centered_deq[m] @ K_smooth_deq[n]   (FP8-cast K)
                    + qm[group(m)]    @ K_smooth[n]         (un-quant BF16 K!)

    Adding qm into Q before the matmul (the previous shortcut) would route
    the correction through the FP8-quantized K, which reintroduces K's
    dominant quant noise scaled by qm and nullifies most of the Q-smoothing
    benefit. We therefore keep ``k_work`` (un-quantized BF16 K_smooth)
    around and apply the correction term separately.

    With V smoothing on, ``v_work`` is V_centered; we reconstitute by adding
    alpha[block] back to V_deq before the PV matmul.
    """
    b, s, h, d = q_work.shape
    q_fp8, q_scale, q_meta = _quant_qk(q_work, cfg)
    k_fp8, k_scale, k_meta = _quant_qk(k_work, cfg)
    v_fp8, v_scale, v_meta = _quant_v(v_work, cfg)
    q_deq = _dequant_qk(q_fp8, q_scale, q_meta, sdpa_dtype)
    k_deq = _dequant_qk(k_fp8, k_scale, k_meta, sdpa_dtype)
    v_deq = _dequant_v(v_fp8, v_scale, v_meta, sdpa_dtype)
    if v_alpha is not None:
        block_v = cfg.v_smooth_block_size
        if s % block_v != 0:
            raise ValueError(f"S={s} not divisible by v_smooth_block_size={block_v}")
        v_alpha_full = v_alpha.repeat_interleave(block_v, dim=1).to(sdpa_dtype)
        v_deq = v_deq + v_alpha_full

    # Fast path: no Q smoothing → qm correction is unnecessary, SDPA is fine.
    if qm is None:
        qb = q_deq.permute(0, 2, 1, 3).contiguous()
        kb = k_deq.permute(0, 2, 1, 3).contiguous()
        vb = v_deq.permute(0, 2, 1, 3).contiguous()
        o = F.scaled_dot_product_attention(qb, kb, vb, scale=sm_scale)
        return o.permute(0, 2, 1, 3).contiguous()

    # Q-smoothing path: chunked manual attention so we can inject the
    # un-quantized qm @ K_smooth.T correction into the score matrix.
    block_q = cfg.q_smooth_block_size
    chunk_m = max(block_q, 256)  # process at least one qm-group per chunk
    q_bh = q_deq.permute(0, 2, 1, 3).contiguous()         # (B,H,S,D) bf16
    qm_bh = qm.permute(0, 2, 1, 3).contiguous()           # (B,H,n_groups,D) fp32
    k_deq_bh = k_deq.permute(0, 2, 1, 3).contiguous()     # (B,H,S,D) bf16  (FP8-cast)
    k_smooth_bh = k_work.permute(0, 2, 1, 3).contiguous() # (B,H,S,D) bf16  (un-quantized)
    v_bh = v_deq.permute(0, 2, 1, 3).contiguous()         # (B,H,S,D) bf16

    out_bh = torch.empty_like(q_bh)
    grp_idx = torch.arange(s, device=q_work.device) // block_q  # (S,)
    for m0 in range(0, s, chunk_m):
        m1 = min(m0 + chunk_m, s)
        q_chunk = q_bh[:, :, m0:m1, :]
        qm_rows = qm_bh[:, :, grp_idx[m0:m1], :]          # (B,H,Mq,D) fp32
        # Main term: FP8 Q_centered @ FP8 K_smooth
        s_main = torch.matmul(q_chunk, k_deq_bh.transpose(-2, -1)).float() * sm_scale
        # Correction: FP32 qm @ BF16 K_smooth (un-quantized!). Compute in FP32.
        s_corr = torch.matmul(qm_rows, k_smooth_bh.float().transpose(-2, -1)) * sm_scale
        scores = s_main + s_corr
        p = torch.softmax(scores, dim=-1).to(sdpa_dtype)
        out_bh[:, :, m0:m1, :] = torch.matmul(p, v_bh)

    return out_bh.permute(0, 2, 1, 3).contiguous()


# ---------------------------------------------------------------------------
# Path B: chunked-Q with full P requant and explicit per-channel V scale.
# ---------------------------------------------------------------------------


def _expand_v_scale_full(
    v_scale: torch.Tensor,
    v_meta: dict,
    *,
    b: int,
    s: int,
    h: int,
    d: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return a per-(B,S,H,D) FP32 scale tensor for V regardless of v_quant kind.

    ``fp8_channel``: one scale per (B,H,D), broadcast across S.
    ``fp8_block``:   one scale per (B, S/block_s, H), broadcast over D and
                     each S-block.
    ``mxfp8``:       one scale per (B, S/block_s, H, D), broadcast across each
                     S-block of length block_s.
    """
    if v_meta["kind"] == "fp8_channel":
        # (B, H, D) → (B, 1, H, D) → (B, S, H, D)
        return v_scale.to(dtype).view(b, 1, h, d).expand(b, s, h, d).contiguous()
    if v_meta["kind"] == "fp8_block":
        block_s = v_meta["block_s"]
        return (
            v_scale.to(dtype)
            .unsqueeze(-1)
            .repeat_interleave(block_s, dim=1)
            .expand(b, s, h, d)
            .contiguous()
        )
    if v_meta["kind"] == "mxfp8":
        block_s = v_meta["block_s"]
        # (B, S/block_s, H, D) → (B, S, H, D) by repeating each row block_s times.
        return v_scale.to(dtype).repeat_interleave(block_s, dim=1).contiguous()
    raise ValueError(f"bad v_meta: {v_meta!r}")


def _estimate_rowmax_from_qm_k(
    qm: torch.Tensor,
    k_work: torch.Tensor,
    *,
    block_q: int,
    sm_scale: float,
    chunk_n: int = 4096,
) -> torch.Tensor:
    """Estimate per-row softmax max with ``max_n(qm[group] @ K_smooth[n])``.

    Returns ``(B, H, S)`` in scaled-score units, matching the score tensor
    passed to the Triton P-requant kernel. The estimate is shared by all rows
    in the same Q-smoothing group and repeated to per-row shape because the
    kernel accepts a generic per-row rowmax tensor.
    """
    b, n_groups, h, d = qm.shape
    _, s, hk, dk = k_work.shape
    if h != hk or d != dk:
        raise ValueError(f"qm/k shape mismatch: qm={tuple(qm.shape)} k={tuple(k_work.shape)}")
    if s % block_q != 0 or s // block_q != n_groups:
        raise ValueError(f"qm groups {n_groups} do not match S={s}, block_q={block_q}")

    qm_bhgd = qm.permute(0, 2, 1, 3).contiguous().float()
    k_bhsd = k_work.permute(0, 2, 1, 3).contiguous().float()
    rowmax_g = torch.full((b, h, n_groups), -float("inf"), dtype=torch.float32, device=qm.device)
    for n0 in range(0, s, chunk_n):
        n1 = min(n0 + chunk_n, s)
        scores = torch.matmul(qm_bhgd, k_bhsd[:, :, n0:n1, :].transpose(-2, -1))
        rowmax_g = torch.maximum(rowmax_g, scores.amax(dim=-1))
    return (rowmax_g * float(sm_scale)).repeat_interleave(block_q, dim=2).contiguous()


def _estimate_rowmax_from_q_kmeans_labels(
    q_work: torch.Tensor,
    qm: torch.Tensor,
    k_work: torch.Tensor,
    q_kmeans: KMeansReorderResult,
    *,
    block_q: int,
    sm_scale: float,
    chunk_n: int = 4096,
) -> torch.Tensor:
    """Estimate rowmax with per-segment means only at kmeans boundaries.

    Fixed-size Q smoothing blocks can straddle kmeans cluster boundaries after
    stable label sorting. Single-cluster blocks keep the original fixed-block
    ``qm @ K`` estimate. Mixed blocks are split into consecutive same-label
    segments; each segment uses its own local Q mean to avoid mixing two
    clusters across the boundary.
    """
    b, s, h, d = q_work.shape
    if s % block_q != 0:
        raise ValueError(f"S={s} must be divisible by block_q={block_q}")
    if qm.shape != (b, s // block_q, h, d):
        raise ValueError(f"qm shape {tuple(qm.shape)} does not match q shape {tuple(q_work.shape)}")
    if k_work.shape != q_work.shape:
        raise ValueError(f"k_work shape {tuple(k_work.shape)} does not match q shape {tuple(q_work.shape)}")

    labels_orig = q_kmeans.labels.reshape(b, h, s)
    order = q_kmeans.order.reshape(b, h, s)
    labels_re = torch.gather(labels_orig, 2, order)

    rowmax = _estimate_rowmax_from_qm_k(
        qm,
        k_work,
        block_q=block_q,
        sm_scale=sm_scale,
        chunk_n=chunk_n,
    )
    qm_rows = _broadcast_qm(qm, s)
    q_pre = (q_work.float() + qm_rows.float()).permute(0, 2, 1, 3).contiguous()
    k_bhsd = k_work.permute(0, 2, 1, 3).contiguous().float()
    n_groups = s // block_q
    labels_g = labels_re.reshape(b, h, n_groups, block_q)
    mixed = labels_g.amax(dim=-1) != labels_g.amin(dim=-1)

    # Mixed blocks are sparse, but scanning all K tokens once per segment is
    # very slow in end-to-end fake quant. Collect segment means for every
    # (B,H), pad them to one batch, then scan K with strided batched GEMMs.
    bh = b * h
    starts_by_bh: list[list[int]] = [[] for _ in range(bh)]
    ends_by_bh: list[list[int]] = [[] for _ in range(bh)]
    for bi in range(b):
        for hi in range(h):
            groups = torch.nonzero(mixed[bi, hi], as_tuple=False).flatten()
            if groups.numel() == 0:
                continue

            flat = bi * h + hi
            for gi in groups.tolist():
                base = gi * block_q
                lab = labels_re[bi, hi, base : base + block_q]
                change = torch.nonzero(lab[1:] != lab[:-1], as_tuple=False).flatten() + 1
                bounds = [0, *change.tolist(), block_q]
                for start, end in zip(bounds[:-1], bounds[1:]):
                    starts_by_bh[flat].append(base + start)
                    ends_by_bh[flat].append(base + end)

    max_segments = max((len(starts) for starts in starts_by_bh), default=0)
    if max_segments == 0:
        return rowmax.contiguous()

    means_bhd = torch.zeros((bh, max_segments, d), dtype=torch.float32, device=q_work.device)
    valid = torch.zeros((bh, max_segments), dtype=torch.bool, device=q_work.device)
    for bi in range(b):
        for hi in range(h):
            flat = bi * h + hi
            starts = starts_by_bh[flat]
            if not starts:
                continue
            starts_t = torch.tensor(starts, device=q_work.device, dtype=torch.long)
            ends_t = torch.tensor(ends_by_bh[flat], device=q_work.device, dtype=torch.long)
            prefix = torch.nn.functional.pad(q_pre[bi, hi].cumsum(dim=0), (0, 0, 1, 0))
            lengths = (ends_t - starts_t).to(torch.float32).unsqueeze(-1)
            means = (prefix[ends_t] - prefix[starts_t]) / lengths
            n_segments = means.shape[0]
            means_bhd[flat, :n_segments] = means
            valid[flat, :n_segments] = True

    k_bsd = k_bhsd.reshape(bh, s, d)
    seg_rowmax = torch.full(
        (bh, max_segments),
        -float("inf"),
        dtype=torch.float32,
        device=q_work.device,
    )
    for n0 in range(0, s, chunk_n):
        n1 = min(n0 + chunk_n, s)
        scores = torch.bmm(means_bhd, k_bsd[:, n0:n1, :].transpose(1, 2))
        seg_rowmax = torch.maximum(seg_rowmax, scores.amax(dim=-1))
    seg_rowmax = (seg_rowmax * float(sm_scale)).masked_fill(~valid, 0.0)

    rowmax_bhs = rowmax.reshape(bh, s)
    for flat, (starts, ends) in enumerate(zip(starts_by_bh, ends_by_bh, strict=True)):
        for idx, (start, end) in enumerate(zip(starts, ends, strict=True)):
            rowmax_bhs[flat, start:end] = seg_rowmax[flat, idx]
    return rowmax.contiguous()


def _fake_quant_attention_p_requant(
    q_work: torch.Tensor,
    k_work: torch.Tensor,
    v_work: torch.Tensor,
    qm: Optional[torch.Tensor],
    v_alpha: Optional[torch.Tensor],
    q_kmeans: Optional[KMeansReorderResult],
    cfg: QuantConfig,
    sm_scale: float,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Streaming attention with P→FP8 cast and per-channel V scale post-mul.

    For ``v_quant=fp8_channel`` this dispatches to the Triton kernel in
    :mod:`attention_triton`, which fuses online softmax + P FP8 cast +
    per-channel V scale into a single pass over K.

    For ``v_quant=mxfp8`` we pre-dequantize V (so the per-K-block scale is
    absorbed) and reuse the Triton kernel with a unit v_scale, since the
    only remaining cast error inside the kernel is the P FP8 cast.
    """
    b, s, h, d = q_work.shape
    device = q_work.device

    # Quantize and dequantize Q, K → BF16 for the matmul. (Q/K cast errors
    # enter as input perturbations.)
    q_fp8, q_scale, q_meta = _quant_qk(q_work, cfg)
    k_fp8, k_scale, k_meta = _quant_qk(k_work, cfg)
    q_deq = _dequant_qk(q_fp8, q_scale, q_meta, torch.bfloat16)
    k_deq = _dequant_qk(k_fp8, k_scale, k_meta, torch.bfloat16)

    # NOTE: do NOT add qm into q_eff. Instead pass k_smooth_bf16 (un-quantized)
    # and qm to the kernel so the correction term `qm @ K_smooth^T` is added
    # in FP32 inside the kernel — matching the production CuTe-DSL Sage2
    # semantics. Adding qm to q_deq here would route the correction through
    # the FP8-cast K and reintroduce K's dominant quant noise scaled by qm.

    # Quantize V and prepare the kernel arguments. The handling differs by
    # V quant kind so we can faithfully model the matching P quant scheme:
    #
    #   fp8_channel: v_bf16 = V_fp8-cast-to-bf16; per-D scale applied
    #                post-matmul. P quant: element-wise e4m3.
    #   fp8_block:   v_bf16 = V_fp8-cast-to-bf16; per-K-block FP32 scalar
    #                multiplied INSIDE the K-loop per block. P quant:
    #                element-wise e4m3.
    #   mxfp8:       absorb the per-(S-block, D) UE8M0 scale into v_bf16
    #                by pre-dequantizing. P quant: MX-style (per-K-block
    #                UE8M0 on P before the e4m3 cast).
    v_fp8, v_scale, v_meta = _quant_v(v_work, cfg)
    v_kind = v_meta["kind"]
    v_block_scale_arg: Optional[torch.Tensor] = None  # (B, S/blk, H) FP32 for fp8_block
    v_block_size_arg: int = 0
    if v_kind == "fp8_channel":
        v_bf16 = v_fp8.to(torch.bfloat16)
        v_scale_bhd = v_scale.to(torch.float32).contiguous()  # (B,H,D)
    elif v_kind == "fp8_block":
        v_bf16 = v_fp8.to(torch.bfloat16)
        v_scale_bhd = torch.ones((b, h, d), dtype=torch.float32, device=device)
        v_block_scale_arg = v_scale.to(torch.float32).contiguous()  # (B, S/blk, H)
        v_block_size_arg = v_meta["block_s"]
    elif v_kind == "mxfp8":
        v_bf16 = _dequant_v(v_fp8, v_scale, v_meta, torch.bfloat16)
        v_scale_bhd = torch.ones((b, h, d), dtype=torch.float32, device=device)
    else:
        raise ValueError(f"bad v_meta: {v_meta!r}")

    # Permute to (B, H, S, D) layout for the Triton kernel.
    q_bhsd = q_deq.permute(0, 2, 1, 3).contiguous()
    k_bhsd = k_deq.permute(0, 2, 1, 3).contiguous()
    v_bhsd = v_bf16.permute(0, 2, 1, 3).contiguous()

    # V smoothing alpha stays in NHD-style (B, S/block, H, D) layout so the
    # kernel can read it with strides (b, n, h, d).
    v_alpha_arg: Optional[torch.Tensor] = None
    if v_alpha is not None:
        v_alpha_arg = v_alpha.to(torch.float32).contiguous()

    # Q-smooth correction inputs: un-quantized K_smooth (BHSD) and per-group
    # qm. The kernel expects qm in (B, n_g, H, D) layout — that matches the
    # natural output of group_mean_q so no permute is needed.
    k_smooth_arg: Optional[torch.Tensor] = None
    qm_arg: Optional[torch.Tensor] = None
    if qm is not None:
        # Defensive: force BF16 (the kernel asserts it) in case input was FP16.
        k_smooth_arg = k_work.to(torch.bfloat16).permute(0, 2, 1, 3).contiguous()
        qm_arg = qm.to(torch.float32).contiguous()              # (B,n_g,H,D) fp32

    rowmax_est_arg: Optional[torch.Tensor] = None
    if cfg.rowmax_mode == "online":
        pass
    elif cfg.rowmax_mode == "qm_k":
        if qm is None:
            raise ValueError("rowmax_mode='qm_k' requires smoothing='full' so qm exists")
        # The Triton kernel now guards large rowmax-estimation misses with a
        # FA4-style fallback update. In that mode the cheap fixed-block
        # estimate is accurate enough for tested Wan workloads, while the
        # boundary-segment estimate is too expensive for end-to-end debugging.
        rowmax_est_arg = _estimate_rowmax_from_qm_k(
            qm,
            k_work,
            block_q=cfg.q_smooth_block_size,
            sm_scale=sm_scale,
        )
    else:
        raise ValueError(f"unsupported rowmax_mode: {cfg.rowmax_mode!r}")

    # Choose P quant mode: auto pairs with V quant per the spec.
    p_mode = cfg.p_quant
    if p_mode == "auto":
        p_mode = "mx" if v_kind == "mxfp8" else "elementwise"
    p_mx_block_n = cfg.p_mx_block_n if cfg.p_mx_block_n > 0 else cfg.v_mxfp8_block_size

    o_bhsd = fake_quant_attention_triton(
        q_bhsd, k_bhsd, v_bhsd, v_scale_bhd,
        sm_scale=float(sm_scale),
        p_max_offset=int(cfg.p_max_offset),
        block_m=cfg.p_requant_block_m,
        block_n=cfg.p_requant_block_n,
        v_alpha=v_alpha_arg,
        v_smooth_block=cfg.v_smooth_block_size if v_alpha is not None else 0,
        k_smooth_bhsd=k_smooth_arg,
        qm_bhgd=qm_arg,
        q_smooth_block=cfg.q_smooth_block_size if qm is not None else 0,
        v_block_scale_bsh=v_block_scale_arg,
        v_block_size=v_block_size_arg,
        p_quant_mode=p_mode,
        p_mx_block_n=p_mx_block_n,
        rowmax_est_bhs=rowmax_est_arg,
    )
    return o_bhsd.permute(0, 2, 1, 3).contiguous().to(out_dtype)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


@dataclass
class PreprocessCache:
    """Cached output of :func:`prepare_for_attention`.

    Reused across configs that share the same preprocess state — i.e.
    ``(smoothing, q_kmeans_k, q_smooth_block_size, q_kmeans_iters,
    q_kmeans_seed, v_smooth_mode, v_smooth_block_size, v_kmeans_k,
    v_kmeans_iters, v_kmeans_seed)`` — so the eval script does not redo
    kmeans / smoothing for every downstream quant combo.
    """

    q_work: torch.Tensor
    k_work: torch.Tensor
    v_work: torch.Tensor
    qm: Optional[torch.Tensor]
    v_alpha: Optional[torch.Tensor]
    q_kmeans: Optional[KMeansReorderResult]
    v_kmeans: Optional[KMeansReorderResult]
    smoothing: str
    q_kmeans_k: Optional[int]
    q_smooth_block_size: int
    v_smooth_mode: str
    v_smooth_block_size: int
    v_kmeans_k: Optional[int]


def prepare_for_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cfg: QuantConfig,
) -> PreprocessCache:
    """Run kmeans + Q/K smoothing + V smoothing without any quantization.

    Useful for sharing preprocess work across configs that vary only in
    ``qk_quant``, ``v_quant``, or ``p_requant``.
    """
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q/k/v shapes must match")
    if q.ndim != 4 or not q.is_cuda:
        raise ValueError("q/k/v must be 4-D NHD on CUDA")
    q_work, k_work, v_work, qm, v_alpha, q_kmeans, v_kmeans = _preprocess(q, k, v, cfg)
    return PreprocessCache(
        q_work=q_work,
        k_work=k_work,
        v_work=v_work,
        qm=qm,
        v_alpha=v_alpha,
        q_kmeans=q_kmeans,
        v_kmeans=v_kmeans,
        smoothing=cfg.smoothing,
        q_kmeans_k=cfg.q_kmeans_k,
        q_smooth_block_size=cfg.q_smooth_block_size,
        v_smooth_mode=cfg.v_smooth_mode,
        v_smooth_block_size=cfg.v_smooth_block_size,
        v_kmeans_k=cfg.v_kmeans_k,
    )


def fake_quant_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cfg: Optional[QuantConfig] = None,
    *,
    sm_scale: Optional[float] = None,
    return_artifacts: bool = False,
    preprocess_cache: Optional[PreprocessCache] = None,
) -> torch.Tensor | tuple[torch.Tensor, FakeQuantArtifacts]:
    """Fake-quant attention with FP8/MXFP8 Q/K/V (and optional P requant).

    Parameters
    ----------
    q, k, v : torch.Tensor
        Shape ``(B, S, H, D)``, ``D in {64, 128}``, BF16/FP16 on CUDA.
    cfg : QuantConfig
        Quantization + preprocessing knobs. ``None`` falls back to defaults.
    sm_scale : float, optional
        Softmax scaling. Defaults to ``1 / sqrt(D)``.
    return_artifacts : bool
        If True, also return intermediate tensors (qm, kmeans, etc.).

    Returns
    -------
    torch.Tensor (B, S, H, D) — attention output in the input dtype.
    """
    if cfg is None:
        cfg = QuantConfig()
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError(
            f"q/k/v shapes must match; got {tuple(q.shape)}, {tuple(k.shape)}, {tuple(v.shape)}"
        )
    if q.ndim != 4:
        raise ValueError(f"q/k/v must be 4-D (B,S,H,D); got {q.shape}")
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        raise ValueError("q/k/v must live on CUDA")

    out_dtype = q.dtype
    if out_dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(f"unsupported dtype {out_dtype}")
    sdpa_dtype = torch.float16 if out_dtype == torch.float16 else torch.bfloat16

    b, s, h, d = q.shape
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(d)

    if preprocess_cache is not None:
        if (
            preprocess_cache.smoothing != cfg.smoothing
            or preprocess_cache.q_kmeans_k != cfg.q_kmeans_k
            or preprocess_cache.q_smooth_block_size != cfg.q_smooth_block_size
            or preprocess_cache.v_smooth_mode != cfg.v_smooth_mode
            or preprocess_cache.v_smooth_block_size != cfg.v_smooth_block_size
            or preprocess_cache.v_kmeans_k != cfg.v_kmeans_k
        ):
            raise ValueError(
                "preprocess_cache does not match cfg's smoothing/kmeans/v_smooth settings"
            )
        q_work = preprocess_cache.q_work
        k_work = preprocess_cache.k_work
        v_work = preprocess_cache.v_work
        qm = preprocess_cache.qm
        v_alpha = preprocess_cache.v_alpha
        q_kmeans = preprocess_cache.q_kmeans
        # v_kmeans is captured but doesn't need inversion (output indexed by Q's S).
    else:
        q_work, k_work, v_work, qm, v_alpha, q_kmeans, _ = _preprocess(q, k, v, cfg)

    if cfg.p_requant:
        o = _fake_quant_attention_p_requant(
            q_work, k_work, v_work, qm, v_alpha, q_kmeans, cfg, sm_scale, out_dtype
        )
    else:
        o = _fake_quant_attention_sdpa(
            q_work, k_work, v_work, qm, v_alpha, cfg, sm_scale, sdpa_dtype
        )
        o = o.to(out_dtype)

    # Inverse Q kmeans reorder: gather rows back to original Q order.
    # (V kmeans does NOT need inverting — output rows are indexed by Q's S,
    # and the V/K joint permutation is internally self-cancelling.)
    if q_kmeans is not None:
        inv = q_kmeans.inverse_order  # (B*H, S)
        bb, ss, hh, dd = o.shape
        o_bh = o.permute(0, 2, 1, 3).reshape(bb * hh, ss, dd)
        o_back = torch.gather(o_bh, 1, inv.unsqueeze(-1).expand(-1, -1, dd))
        o = o_back.reshape(bb, hh, ss, dd).permute(0, 2, 1, 3).contiguous()

    if return_artifacts:
        art = FakeQuantArtifacts(
            qm=qm,
            kmeans=q_kmeans,
            q_eff_dtype=sdpa_dtype,
            p_requant=cfg.p_requant,
        )
        return o, art
    return o


def reference_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    sm_scale: Optional[float] = None,
) -> torch.Tensor:
    """Reference attention via torch SDPA on the raw inputs (no quantization)."""
    if q.ndim != 4:
        raise ValueError(f"q/k/v must be (B,S,H,D); got {q.shape}")
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(q.shape[-1])
    qb = q.permute(0, 2, 1, 3).contiguous()
    kb = k.permute(0, 2, 1, 3).contiguous()
    vb = v.permute(0, 2, 1, 3).contiguous()
    o = F.scaled_dot_product_attention(qb, kb, vb, scale=sm_scale)
    return o.permute(0, 2, 1, 3).contiguous()


__all__ = ["FakeQuantArtifacts", "fake_quant_attention", "reference_attention"]
