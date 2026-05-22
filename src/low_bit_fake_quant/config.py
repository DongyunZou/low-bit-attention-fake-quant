from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

QKQuant = Literal["fp8_block", "mxfp8"]
# V quantization scheme (algorithm 2 line 9, ``phi(V_j)``):
#   - ``fp8_channel``: one FP32 scale per (B, H, D), shared across S. The
#     production "per-channel V" pattern in Sage-style FP8 attention. P quant
#     is element-wise e4m3fn; V's scale post-multiplies at the end.
#   - ``fp8_block``: one FP32 scale per (B, S/v_fp8_block_size, H). A single
#     scalar per S-block of V — matches the algorithm's ``phi(V_j)`` exactly
#     where each K/V block has its own scale. P quant is element-wise; V's
#     per-block scale multiplies inside the K-loop, per block.
#   - ``mxfp8``: microscaling FP8 — per (B, S/v_mxfp8_block_size, H, D) UE8M0
#     power-of-2 scale. The most fine-grained option; P quant is matched
#     MX-style (per-(M-row, K-block) UE8M0 scale on P before the e4m3 cast)
#     to keep the FP8 mma compatible with V's microscaling.
VQuant = Literal["fp8_channel", "fp8_block", "mxfp8"]
Smoothing = Literal["off", "k_only", "full"]
VSmoothMode = Literal["off", "per_block"]
# P quantization mode. ``auto`` picks based on ``v_quant``:
#   fp8_channel, fp8_block → ``elementwise`` (current default behavior)
#   mxfp8                   → ``mx`` (per-K-block UE8M0 on P)
PQuant = Literal["auto", "elementwise", "mx"]


@dataclass(frozen=True)
class QuantConfig:
    """Configuration for QKV fake-quant preprocessing.

    The defaults mirror the fp8-attention fork's production path: Q/K use
    FP32-scale block FP8, V uses per-channel FP8, full smoothing is enabled,
    and P requant probes use p_max_offset=8.
    """

    qk_quant: QKQuant = "fp8_block"
    v_quant: VQuant = "fp8_channel"
    smoothing: Smoothing = "full"
    q_smooth_block_size: int = 256
    q_kmeans_k: int | None = 32
    q_kmeans_iters: int = 10
    q_kmeans_seed: int = 0
    fp8_block_size: int = 128         # QK fp8_block: per (B, S/blk, H)
    mxfp8_block_size: int = 32        # QK mxfp8: per (B, S, H, D/blk)
    # V quant block sizes (separate from QK to allow independent tuning).
    v_fp8_block_size: int = 64        # V fp8_block: per (B, S/blk, H)
    # V mxfp8 block size along S. Set to 64 (matching default p_requant_block_n)
    # so MX P quant aligns with V's micro-block boundaries; can be 32 if
    # block_n is also set to 32.
    v_mxfp8_block_size: int = 64      # V mxfp8: per (B, S/blk, H, D)
    # P quant mode (auto/elementwise/mx). ``auto`` picks per V quant.
    p_quant: PQuant = "auto"
    # P MX block size along N (kv) when p_quant=mx. Defaults to V's mxfp8
    # block size when not specified (so they align).
    p_mx_block_n: int = 0   # 0 → match v_mxfp8_block_size
    p_max_offset: int = 8
    correction_dtype: torch.dtype = torch.float16
    # P requant: cast P (softmax output) to FP8 e4m3fn before PV matmul.
    # When True, the kernel models the full FP8 attention numerics (Q/K/V/P
    # all cast). When False, the kernel only fake-quants Q/K/V and runs P
    # in FP32, matching the "upper bound" reference in the test plan.
    p_requant: bool = True
    # Query-chunk size for the (legacy torch) P-requant streaming attention.
    # Kept for backward compatibility / debugging; the Triton kernel uses
    # ``p_requant_block_m`` / ``p_requant_block_n`` below.
    p_requant_q_chunk: int = 512
    # Triton kernel block sizes for the P-requant attention forward pass.
    # 64x64 is a safe default on H100; head dim is fixed by the input.
    p_requant_block_m: int = 64
    p_requant_block_n: int = 64
    # V per-block smoothing (Algorithm 2 from the SVG attention paper).
    # ``off``: no V smoothing.
    # ``per_block``: split V into S-blocks of size ``v_smooth_block_size``,
    # subtract each block's per-channel mean (D-vector), and reconstitute the
    # mean as a correction term either inline (SDPA path) or via the
    # streaming C accumulator inside the Triton kernel (P-requant path).
    # Reduces V dynamic range so FP8 quantization of V is more accurate.
    v_smooth_mode: VSmoothMode = "off"
    v_smooth_block_size: int = 64
    # V k-means token reorder. When set, clusters V tokens by Euclidean
    # distance (flash-kmeans, same kernel as Q kmeans) and permutes V along
    # S so similar-magnitude tokens are bunched. K is co-permuted with the
    # same permutation so attention semantics are preserved (this is
    # mathematically a no-op when K and V are jointly permuted along their
    # shared S dim). The benefit comes from V smoothing's per-block mean
    # being tighter on reordered V → smaller V_centered dynamic range →
    # better FP8 V quant. Typical values 32/64; None disables it.
    v_kmeans_k: int | None = None
    v_kmeans_iters: int = 10
    v_kmeans_seed: int = 0
