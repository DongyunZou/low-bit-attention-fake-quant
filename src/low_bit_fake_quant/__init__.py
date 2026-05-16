"""Low-bit fake-quant helper package."""

from .config import QuantConfig
from .kmeans import KMeansReorderResult, q_kmeans_reorder
from .pipeline import PreparedQKV, prepare_qkv
from .preprocess import group_mean_q, smooth_k
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

__all__ = [
    "KMeansReorderResult",
    "PreparedQKV",
    "QuantConfig",
    "fp8_block_dequant",
    "fp8_block_quant",
    "fp8_per_channel_dequant",
    "fp8_per_channel_quant",
    "group_mean_q",
    "mxfp8_qk_dequant",
    "mxfp8_qk_quant",
    "mxfp8_v_dequant",
    "mxfp8_v_quant",
    "prepare_qkv",
    "q_kmeans_reorder",
    "smooth_k",
]
