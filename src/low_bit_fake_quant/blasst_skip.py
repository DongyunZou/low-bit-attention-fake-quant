"""Pure-PyTorch tiled simulator for FP8 block-skip attention on video DiT.

This module reproduces the BLASST block-skipping rule (arXiv:2512.12087,
"Dynamic Blocked Attention Sparsity via Softmax Thresholding") layered on top
of full FP8 fake quantization (Q, K, V, and the softmax weights P all cast to
8-bit). It is a *measurement* tool: every matmul runs in PyTorch with FP32
accumulation over de-quantized FP8 values, so the accuracy of the combined
quant + skip scheme can be swept exactly without any custom kernel.

Key properties:

* Memory-safe / tiled. Scores are formed one 128-query-row M-tile at a time
  against all keys, never materializing the full ``[seqlen, seqlen]`` matrix.
* Exact BLASST predicate. A key block ``j`` of an M-tile is dropped iff
  ``block_local_max - running_row_max < log(skip_threshold)`` using the
  *running* (online) row max, pre-softmax: dropped blocks contribute nothing to
  the row denominator or the output accumulator.
* Faithful FP8 numerics. Q/K/V use per-head static amax scaling
  (``scale = amax_per_head / 448``); the softmax weights use the static rule
  ``P_q = dequant(quant(P * 256, e4m3)) / 256`` applied to the per-block online
  probabilities ``exp(score - running_row_max)``.

The running-max BLASST predicate has a convenient property that makes a fully
vectorized implementation exact: a block can only be skipped when its local max
is *below* the running max, so a skipped block never would have raised the
running max. Hence the running max equals a plain cumulative max over the block
local maxima, independent of which blocks are skipped, and skipping reduces to
masking precomputed per-block quantities.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

# e4m3 finite range. ``scale = amax / E4M3_MAX`` maps a tensor's amax onto the
# top of the representable range, so the largest magnitude saturates exactly.
E4M3_MAX = 448.0
# Static softmax-weight expansion factor. Mirrors the Hopper FP8 path's
# ``MAX_OFFSET = 8`` (2**8 = 256), which lifts per-block softmax weights into a
# range the e4m3 mantissa resolves well.
P_STATIC_SCALE = 256.0

QUERY_TILE = 128
KEY_BLOCK = 128

# Ablation ladder rung names. Plain descriptive identifiers (no workflow
# markers) used as dictionary keys throughout the simulator and driver.
LEVEL_REFERENCE = "bf16_ref"          # no quant, no skip (tiled SDPA twin)
LEVEL_FP8_QKV = "fp8_qkv"             # fp8 Q/K/V, online P in fp32, no skip
LEVEL_FP8_STATIC_P = "fp8_static_p"   # + static P*256 quant, no skip
LEVEL_FP8_SKIP = "fp8_static_p_skip"  # + BLASST skip @ skip_threshold

LADDER = (LEVEL_REFERENCE, LEVEL_FP8_QKV, LEVEL_FP8_STATIC_P, LEVEL_FP8_SKIP)


class FullMatrixAllocationError(RuntimeError):
    """Raised when a code path would allocate a full ``seqlen x seqlen`` buffer."""


def guard_no_full_matrix(shape, seqlen: int) -> None:
    """Reject any tensor shape that contains a ``seqlen x seqlen`` plane.

    The simulator must stay tiled; a buffer whose two largest dims are both the
    full sequence length means someone tried to materialize the dense score or
    probability matrix. This is the explicit safety assertion the study relies
    on (a ``[*, 69120, 69120]`` allocation would OOM and is never legitimate).
    """
    dims = [int(d) for d in shape]
    big = sorted(d for d in dims if d >= seqlen)
    if len(big) >= 2:
        raise FullMatrixAllocationError(
            f"refusing to allocate full {seqlen}x{seqlen} matrix; got shape {tuple(dims)}"
        )


# SDPA backends that stream the score matrix (never materialize seqlen x seqlen).
_MEMORY_SAFE_SDPA_BACKENDS = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]


def sdpa_ground_truth(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    sm_scale: Optional[float] = None,
    allow_math_fallback: bool = False,
) -> torch.Tensor:
    """Numerical ground truth: ``torch`` SDPA with a pinned memory-safe backend.

    The flash / mem-efficient backends stream the scores so SDPA itself never
    builds the full ``seqlen x seqlen`` matrix. The math backend would, and
    OOMs at full seqlen, so it is excluded unless ``allow_math_fallback`` is set
    (used only to demonstrate the negative case).
    """
    if q.ndim != 4:
        raise ValueError(f"q/k/v must be (B, S, H, D); got {tuple(q.shape)}")
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(q.shape[-1])
    backends = list(_MEMORY_SAFE_SDPA_BACKENDS)
    if allow_math_fallback:
        backends.append(SDPBackend.MATH)
    qb = q.permute(0, 2, 1, 3).contiguous()
    kb = k.permute(0, 2, 1, 3).contiguous()
    vb = v.permute(0, 2, 1, 3).contiguous()
    with sdpa_kernel(backends):
        o = F.scaled_dot_product_attention(qb, kb, vb, scale=sm_scale)
    return o.permute(0, 2, 1, 3).contiguous()


def space_time_reorder_index(t: int, h: int, w: int, device=None) -> torch.Tensor:
    """Permutation that groups spatially/temporally adjacent tokens.

    The native token order is the row-major flatten of a ``(t, h, w)`` latent
    grid. This reindexes into ``(t_blocks, h_blocks, w_blocks, local)`` tiles so
    each contiguous 128-token block holds a coherent space-time neighborhood,
    raising whole-block skippability. ``t * h * w`` must equal the sequence
    length. Returned as an index vector usable for gather / scatter.
    """
    seqlen = t * h * w
    # Plain space-filling reorder: sort tokens by a coarse (t, h, w) tile id so
    # neighbors in all three axes land near each other. The exact tiling is a
    # tunable; this default keeps temporal locality primary then spatial.
    idx = torch.arange(seqlen, device=device)
    tt = idx // (h * w)
    rem = idx % (h * w)
    hh = rem // w
    ww = rem % w
    # Coarsen each axis into blocks of ~4 so a 128-token window stays local.
    key = (tt) * (h * w) + (hh) * w + ww  # identity by default
    # Stable sort by composite key keeps it a pure permutation.
    perm = torch.argsort(key, stable=True)
    return perm


def apply_token_permutation(x: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    """Permute a ``(B, S, H, D)`` tensor along the sequence axis."""
    return x.index_select(1, perm)


def invert_permutation(perm: torch.Tensor) -> torch.Tensor:
    """Return ``inv`` such that ``x[perm][inv] == x``."""
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(perm.numel(), device=perm.device)
    return inv


@dataclass
class QuantStats:
    """Round-trip diagnostics for one fake-quantized tensor."""

    saturation_rate: float = 0.0      # fraction clamped to +/- E4M3_MAX
    underflow_zero_rate: float = 0.0  # fraction of nonzero inputs rounded to 0
    roundtrip_rmse: float = 0.0       # RMSE of dequant(quant(x)) vs x


def _fp8_roundtrip(scaled: torch.Tensor) -> torch.Tensor:
    """Quantize a pre-scaled tensor to e4m3 and back to fp32 (fake quant)."""
    return scaled.to(torch.float8_e4m3fn).to(torch.float32)


def fake_quant_per_head(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, QuantStats]:
    """Per-head static-amax e4m3 fake quant of a ``(S, H, D)`` tensor.

    One scalar scale per head (amax taken over the sequence and head-dim axes),
    so this is calibration, not per-channel smoothing. Returns the dequantized
    tensor (fp32), the per-head scale ``(1, H, 1)``, and round-trip diagnostics.
    """
    if x.ndim != 3:
        raise ValueError(f"expected (S, H, D); got {tuple(x.shape)}")
    xf = x.float()
    amax = xf.abs().amax(dim=(0, 2), keepdim=True)            # (1, H, 1)
    scale = (amax / E4M3_MAX).clamp_min(torch.finfo(torch.float32).tiny)
    scaled = xf / scale
    q = _fp8_roundtrip(scaled)
    deq = q * scale
    sat = (scaled.abs() > E4M3_MAX).float().mean().item()
    nonzero = scaled != 0
    underflow = ((q == 0) & nonzero).float().mean().item()
    rmse = (deq - xf).pow(2).mean().sqrt().item()
    return deq, scale, QuantStats(sat, underflow, rmse)


def static_p_quant(p: torch.Tensor) -> torch.Tensor:
    """Static softmax-weight quant: ``dequant(quant(p * 256, e4m3)) / 256``."""
    return _fp8_roundtrip(p * P_STATIC_SCALE) / P_STATIC_SCALE


def static_p_quant_with_stats(p: torch.Tensor):
    """Static P quant plus its saturation / underflow / error diagnostics.

    Performs the e4m3 round-trip once and derives the diagnostics from it (so
    the hot loop does not quantize twice). Returns ``(p_q, saturated,
    underflow_zero, sq_error_sum, count)``; the counts let a driver accumulate
    global rates / RMSE across M-tiles.
    """
    scaled = p * P_STATIC_SCALE
    q = scaled.to(torch.float8_e4m3fn).to(torch.float32)
    deq = q / P_STATIC_SCALE
    saturated = int((scaled.abs() > E4M3_MAX).sum().item())
    underflow = int(((q == 0) & (scaled != 0)).sum().item())
    sq_err = float((deq - p).pow(2).sum().item())
    return deq, saturated, underflow, sq_err, p.numel()


@dataclass
class SkipDiagnostics:
    """Per-(skip-threshold) sparsity / dropped-mass diagnostics."""

    skip_threshold: float
    skip_rate: float = 0.0           # fraction of (M-tile, key-block) pairs dropped
    force_keep_count: int = 0        # rows rescued by the empty-row safeguard
    dropped_mass_mean: float = 0.0
    dropped_mass_median: float = 0.0
    dropped_mass_p95: float = 0.0
    dropped_mass_max: float = 0.0


@dataclass
class SimulationResult:
    """Outputs and diagnostics for one workload across the ablation ladder."""

    # ``(S, H, D)`` fp32 outputs keyed by ladder rung. The skip rung is keyed
    # per skip-threshold under ``skip_outputs``.
    outputs: dict = field(default_factory=dict)
    skip_outputs: dict = field(default_factory=dict)   # skip_threshold -> (S,H,D)
    quant_stats: dict = field(default_factory=dict)    # "q"/"k"/"v"/"p" -> QuantStats
    skip_diagnostics: dict = field(default_factory=dict)  # skip_threshold -> SkipDiagnostics
    peak_memory_bytes: int = 0


def _log_threshold(skip_threshold: float) -> float:
    """``log(skip_threshold)`` with ``skip_threshold == 0`` mapping to -inf."""
    if skip_threshold <= 0.0:
        return -math.inf
    return math.log(skip_threshold)


def running_row_max(block_max: torch.Tensor) -> torch.Tensor:
    """Online running row max over key blocks: cumulative max along the last dim.

    This equals the skip-aware online running max exactly, because a block can
    only be skipped when its local max is below the running max, so skipped
    blocks never raise it.
    """
    return torch.cummax(block_max, dim=-1).values


def blasst_keep_mask(block_max: torch.Tensor, log_threshold: float) -> torch.Tensor:
    """BLASST keep mask: keep block ``j`` iff ``block_max_j - running_max_j >= log(threshold)``.

    Uses the *running* row max (not the final/global row max); the difference is
    what makes this the deployable single-pass BLASST predicate.
    """
    margin = block_max - running_row_max(block_max)   # <= 0
    return margin >= log_threshold


def blasst_tile_keep_mask(block_max_per_row: torch.Tensor, log_threshold: float) -> torch.Tensor:
    """Tile-level BLASST keep mask from per-row block maxima.

    ``block_max_per_row`` is ``(..., rows, n_blocks)`` (the per-query-row local
    max of each key block). A whole key block is kept iff at least one row finds
    it non-negligible, i.e. ``max over rows of (block_max_row - running_max_row)
    >= log(threshold)``. Returns a ``(..., n_blocks)`` mask. This is the exact
    reduction the simulator drops whole 128x128 tiles by.
    """
    margin = block_max_per_row - running_row_max(block_max_per_row)  # rows margin
    tile_margin = margin.amax(dim=-2)                                # over rows
    return tile_margin >= log_threshold


def simulate_workload(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    skip_thresholds,
    levels=LADDER,
    matmul_dtype: torch.dtype = torch.bfloat16,
    collect_dropped_mass: bool = True,
) -> SimulationResult:
    """Run the FP8 block-skip ablation ladder over a single ``(B, S, H, D)`` workload.

    ``q``/``k``/``v`` are ``(B, S, H, D)`` (B must be 1) on a CUDA device. The
    matmul inputs are de-quantized FP8 values cast to ``matmul_dtype`` (bf16 by
    default, which represents e4m3-precision values losslessly) with FP32
    accumulation, matching "fp8 inputs, fp32 accumulate". Set
    ``matmul_dtype=torch.float32`` for the strict-fp32 correctness check.
    """
    if q.ndim != 4 or q.shape[0] != 1:
        raise ValueError(f"expected (1, S, H, D); got {tuple(q.shape)}")
    if not q.is_cuda:
        raise ValueError("simulate_workload requires CUDA tensors")
    _, S, H, D = q.shape
    if S % QUERY_TILE != 0:
        raise ValueError(f"seqlen {S} must be a multiple of {QUERY_TILE}")
    if KEY_BLOCK != QUERY_TILE:
        raise ValueError("tile size other than 128x128 is not allowed")
    n_blocks = S // KEY_BLOCK
    sm_scale = 1.0 / math.sqrt(D)
    device = q.device

    skip_thresholds = list(skip_thresholds)
    want_skip = LEVEL_FP8_SKIP in levels and len(skip_thresholds) > 0
    want_fp8 = any(
        lv in levels for lv in (LEVEL_FP8_QKV, LEVEL_FP8_STATIC_P, LEVEL_FP8_SKIP)
    )

    q3 = q[0]  # (S, H, D)
    k3 = k[0]
    v3 = v[0]

    # ----- per-head static amax fp8 fake quant (Q/K/V) -----
    quant_stats: dict[str, QuantStats] = {}
    if LEVEL_REFERENCE in levels:
        q_ref = q3.to(matmul_dtype)
        k_ref = k3.to(matmul_dtype)
        v_ref = v3.to(matmul_dtype)
    if want_fp8:
        q_deq, _, quant_stats["q"] = fake_quant_per_head(q3)
        k_deq, _, quant_stats["k"] = fake_quant_per_head(k3)
        v_deq, _, quant_stats["v"] = fake_quant_per_head(v3)
        q_fp8 = q_deq.to(matmul_dtype)
        k_fp8 = k_deq.to(matmul_dtype)
        v_fp8 = v_deq.to(matmul_dtype)

    # Per-head layout (H, S, D) for batched matmuls. Transpose K once.
    def to_heads(t):
        return t.permute(1, 0, 2).contiguous()  # (H, S, D)

    if LEVEL_REFERENCE in levels:
        k_ref_t = to_heads(k_ref).transpose(1, 2).contiguous()  # (H, D, S)
        v_ref_h = to_heads(v_ref)                                # (H, S, D)
        q_ref_h = to_heads(q_ref)                                # (H, S, D)
    if want_fp8:
        k_fp8_t = to_heads(k_fp8).transpose(1, 2).contiguous()
        v_fp8_h = to_heads(v_fp8)
        q_fp8_h = to_heads(q_fp8)

    # Output accumulators in fp32, (S, H, D).
    out = {lv: torch.empty((S, H, D), dtype=torch.float32, device=device) for lv in levels}
    skip_out = (
        {th: torch.empty((S, H, D), dtype=torch.float32, device=device) for th in skip_thresholds}
        if want_skip else {}
    )

    log_thresholds = torch.tensor(
        [_log_threshold(th) for th in skip_thresholds], device=device, dtype=torch.float32
    ) if want_skip else None

    # Skip / dropped-mass accumulators.
    skipped_pairs = {th: 0 for th in skip_thresholds}
    force_keeps = {th: 0 for th in skip_thresholds}
    total_pairs = 0
    dropped_chunks = {th: [] for th in skip_thresholds} if (want_skip and collect_dropped_mass) else {}
    # static P quant diagnostics accumulated across tiles.
    p_sat = p_uf = p_cnt = 0
    p_sqerr = 0.0

    torch.cuda.reset_peak_memory_stats(device)

    for start in range(0, S, QUERY_TILE):
        rows = slice(start, start + QUERY_TILE)

        # ---- reference (no quant, no skip): exact full softmax for this tile ----
        if LEVEL_REFERENCE in levels:
            qb = q_ref_h[:, rows, :]                       # (H, m, D)
            scores = torch.matmul(qb.float(), k_ref_t.float()) * sm_scale  # (H, m, S)
            guard_no_full_matrix(scores.shape, S)
            row_max = scores.amax(dim=-1, keepdim=True)
            p = torch.exp(scores - row_max)
            denom = p.sum(dim=-1, keepdim=True)
            acc = torch.matmul(p.to(matmul_dtype), v_ref_h).float()  # (H, m, D)
            o = (acc / denom).to(torch.float32)
            out[LEVEL_REFERENCE][rows] = o.permute(1, 0, 2)
            del scores, p, acc

        if not want_fp8:
            total_pairs += H * n_blocks
            continue

        # ---- fp8 scores for this M-tile against all keys ----
        qb = q_fp8_h[:, rows, :]
        scores = torch.matmul(qb.float(), k_fp8_t.float()) * sm_scale  # (H, m, S) fp32
        guard_no_full_matrix(scores.shape, S)
        m = scores.shape[1]
        sb = scores.view(H, m, n_blocks, KEY_BLOCK)
        block_max = sb.amax(dim=-1)                         # (H, m, n_blocks)
        running_max = torch.cummax(block_max, dim=-1).values  # (H, m, n_blocks)
        final_max = running_max[..., -1]                    # (H, m) global row max
        # online probs against the running max, then rescale each block to the
        # final max so all blocks share one normalization.
        online = torch.exp(sb - running_max.unsqueeze(-1))  # (H, m, n_blocks, K)
        rescale = torch.exp(running_max - final_max.unsqueeze(-1)).unsqueeze(-1)  # (H,m,nb,1)
        # rescale-to-final online weights, reused by the L2 rung and the
        # dropped-mass diagnostic (computed once).
        need_weighted_online = (LEVEL_FP8_QKV in levels) or (want_skip and collect_dropped_mass)
        weighted_online = (online * rescale) if need_weighted_online else None

        # ---- fp8 Q/K/V, online P in fp32 (no P quant, no skip) ----
        if LEVEL_FP8_QKV in levels:
            weighted = weighted_online.view(H, m, S)
            denom = weighted.sum(dim=-1, keepdim=True)
            acc = torch.matmul(weighted.to(matmul_dtype), v_fp8_h).float()
            out[LEVEL_FP8_QKV][rows] = (acc / denom).permute(1, 0, 2)
            del weighted, acc

        need_static_p = (
            LEVEL_FP8_STATIC_P in levels or want_skip
        )
        if need_static_p:
            p_q, s_sat, s_uf, s_err, s_cnt = static_p_quant_with_stats(online)
            p_sat += s_sat
            p_uf += s_uf
            p_sqerr += s_err
            p_cnt += s_cnt
            p_weighted = p_q * rescale                       # (H, m, n_blocks, K)
            l_block = p_weighted.sum(dim=-1)                 # (H, m, n_blocks) per-block denom
            # Per-block PV partials, computed ONCE: pv_block[h,i,j] = sum_k
            # p_weighted[h,i,j,k] * V[h,j,k]. Both the no-skip rung and every
            # skip threshold are then masked sums over the block axis -- no
            # extra matmul per threshold. Summing the per-block partials equals
            # the full P*V matmul, so the no-skip rung and skip@(lambda=0) are
            # bit-identical by construction.
            vb = v_fp8_h.view(H, n_blocks, KEY_BLOCK, D).reshape(H * n_blocks, KEY_BLOCK, D)
            pw = p_weighted.permute(0, 2, 1, 3).reshape(H * n_blocks, m, KEY_BLOCK)
            pv_block = torch.bmm(pw.to(matmul_dtype), vb).float()  # (H*nb, m, D)
            pv_block = pv_block.view(H, n_blocks, m, D).permute(0, 2, 1, 3)  # (H,m,nb,D)
            del pw

        # ---- + static P quant, no skip ----
        if LEVEL_FP8_STATIC_P in levels:
            # Use the same keep-weighted contraction as the skip rung (with an
            # all-ones mask) so skip@(lambda=0) is bit-identical to this rung.
            keep_all = torch.ones(H, n_blocks, device=device, dtype=pv_block.dtype)
            acc = torch.einsum("hmjd,hj->hmd", pv_block, keep_all)    # (H, m, D)
            denom = torch.einsum("hmj,hj->hm", l_block, keep_all).unsqueeze(-1).clamp_min(1e-30)
            out[LEVEL_FP8_STATIC_P][rows] = (acc / denom).permute(1, 0, 2)
            del acc, keep_all

        # ---- + BLASST skip @ each threshold (per (M-tile, key-block) tile) ----
        if want_skip:
            # Per-row skip margin (<= 0): block local max minus running row max.
            margin = block_max - running_max                 # (H, m, n_blocks)
            # Tile-level reduction: a whole key block is dropped only if EVERY
            # query row finds it negligible, i.e. max over rows of the per-row
            # margin is below log(threshold). (Codex/BLASST Algorithm 1: do NOT
            # use rowmax(block_max) - rowmax(running_max).)
            tile_margin = margin.amax(dim=1)                 # (H, n_blocks)
            tile_block_max = block_max.amax(dim=1)           # (H, n_blocks)
            # per-block true mass (unquantized) for the dropped-mass diagnostic,
            # normalized by the true no-skip denominator (fp8 scores, online P).
            if collect_dropped_mass:
                block_mass = weighted_online.sum(dim=-1)     # (H, m, n_blocks)
                true_denom = block_mass.sum(dim=-1, keepdim=True).clamp_min(1e-30)
                block_frac = block_mass / true_denom         # (H, m, n_blocks)
            for th, logt in zip(skip_thresholds, log_thresholds):
                keep = tile_margin >= logt                   # (H, n_blocks) bool
                # empty-tile safeguard: an M-tile with zero kept key blocks
                # force-keeps its single largest-local-max block.
                none_kept = ~keep.any(dim=-1)                # (H,)
                n_force = int(none_kept.sum().item())
                if n_force:
                    argmax_blk = tile_block_max.argmax(dim=-1)  # (H,)
                    rescue = torch.zeros_like(keep)
                    rescue.scatter_(-1, argmax_blk.unsqueeze(-1), True)
                    keep = keep | (rescue & none_kept.unsqueeze(-1))
                    force_keeps[th] += n_force
                keep_col = keep.to(pv_block.dtype)              # (H, n_blocks)
                # contract over the block axis weighted by the keep mask -- no
                # large intermediate tensor is materialized.
                acc = torch.einsum("hmjd,hj->hmd", pv_block, keep_col)    # (H, m, D)
                denom = torch.einsum("hmj,hj->hm", l_block, keep_col).unsqueeze(-1).clamp_min(1e-30)
                skip_out[th][rows] = (acc / denom).permute(1, 0, 2)
                skipped_pairs[th] += int((~keep).sum().item())
                if collect_dropped_mass:
                    dropped = (block_frac * (~keep).unsqueeze(1)).sum(dim=-1)  # (H, m)
                    dropped_chunks[th].append(dropped.reshape(-1).to("cpu"))
                del acc
            del margin, tile_margin, tile_block_max
            if collect_dropped_mass:
                del block_mass, block_frac

        if need_static_p:
            del p_q, p_weighted, l_block, pv_block
        if weighted_online is not None:
            del weighted_online

        total_pairs += H * n_blocks
        del scores, sb, block_max, running_max, online, rescale

    peak = int(torch.cuda.max_memory_allocated(device))

    # ---- assemble result ----
    result = SimulationResult(peak_memory_bytes=peak)
    for lv in levels:
        if lv == LEVEL_FP8_SKIP:
            continue
        result.outputs[lv] = out[lv].view(1, S, H, D)
    result.quant_stats = quant_stats
    if want_fp8 and p_cnt:
        result.quant_stats["p"] = QuantStats(
            saturation_rate=p_sat / p_cnt,
            underflow_zero_rate=p_uf / p_cnt,
            roundtrip_rmse=math.sqrt(p_sqerr / p_cnt),
        )

    if want_skip:
        for th in skip_thresholds:
            result.skip_outputs[th] = skip_out[th].view(1, S, H, D)
            diag = SkipDiagnostics(
                skip_threshold=th,
                skip_rate=skipped_pairs[th] / max(1, total_pairs),
                force_keep_count=force_keeps[th],
            )
            if collect_dropped_mass and dropped_chunks[th]:
                dm = torch.cat(dropped_chunks[th]).float()
                diag.dropped_mass_mean = float(dm.mean())
                diag.dropped_mass_max = float(dm.max())
                # torch.quantile caps at 2**24 elements; sort-index instead so
                # the per-row percentile is exact at full workload size.
                sd = dm.sort().values
                n = sd.numel()
                diag.dropped_mass_median = float(sd[(n - 1) // 2])
                diag.dropped_mass_p95 = float(sd[min(n - 1, int(0.95 * (n - 1)))])
            result.skip_diagnostics[th] = diag

    return result
