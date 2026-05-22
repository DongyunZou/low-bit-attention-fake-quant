"""Low-bit fake-quant helper package."""

from .attention import (
    FakeQuantArtifacts,
    PreprocessCache,
    fake_quant_attention,
    prepare_for_attention,
    reference_attention,
)
from .config import QuantConfig
from .kmeans import (
    KMeansReorderResult,
    apply_kv_permutation,
    kmeans_reorder_tokens,
    q_kmeans_reorder,
    v_kmeans_reorder,
)
from .pipeline import PreparedQKV, prepare_qkv
from .preprocess import group_mean_q, smooth_k, smooth_v_per_block
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
    "FakeQuantArtifacts",
    "KMeansReorderResult",
    "PreparedQKV",
    "PreprocessCache",
    "QuantConfig",
    "fake_quant_attention",
    "prepare_for_attention",
    "fp8_block_dequant",
    "fp8_block_quant",
    "fp8_per_channel_dequant",
    "fp8_per_channel_quant",
    "group_mean_q",
    "smooth_v_per_block",
    "mxfp8_qk_dequant",
    "mxfp8_qk_quant",
    "mxfp8_v_dequant",
    "mxfp8_v_quant",
    "prepare_qkv",
    "apply_kv_permutation",
    "kmeans_reorder_tokens",
    "q_kmeans_reorder",
    "v_kmeans_reorder",
    "reference_attention",
    "smooth_k",
]
