from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

QKQuant = Literal["fp8_block", "mxfp8"]
VQuant = Literal["fp8_channel", "mxfp8_s"]
Smoothing = Literal["off", "k_only", "full"]


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
    fp8_block_size: int = 128
    mxfp8_block_size: int = 32
    p_max_offset: int = 8
    correction_dtype: torch.dtype = torch.float16
