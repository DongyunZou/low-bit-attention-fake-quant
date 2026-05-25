from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import QuantConfig
from .kmeans import KMeansReorderResult, q_kmeans_reorder
from .preprocess import group_mean_q, smooth_k
from .quant_triton import (
    fp8_block_quant,
    fp8_per_channel_quant,
    mxfp8_qk_quant,
    mxfp8_v_quant,
)


@dataclass
class PreparedQKV:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    q_scale: torch.Tensor
    k_scale: torch.Tensor
    v_scale: torch.Tensor
    qm: torch.Tensor | None = None
    k_mean: torch.Tensor | None = None
    correction: torch.Tensor | None = None
    kmeans: KMeansReorderResult | None = None


def _validate_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError(f"q/k/v shapes must match; got {q.shape}, {k.shape}, {v.shape}")
    if q.ndim != 4:
        raise ValueError(f"q/k/v must be 4-D NHD; got {q.shape}")
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        raise ValueError("q/k/v must live on CUDA")


def prepare_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, cfg: QuantConfig) -> PreparedQKV:
    """Run the configured QKV preprocessing and quantization pipeline."""
    _validate_qkv(q, k, v)
    q_work = q.contiguous()
    k_work = k.contiguous()
    v_work = v.contiguous()

    kmeans = None
    if cfg.q_kmeans_k is not None:
        kmeans = q_kmeans_reorder(
            q_work,
            n_clusters=cfg.q_kmeans_k,
            max_iters=cfg.q_kmeans_iters,
            seed=cfg.q_kmeans_seed,
        )
        q_work = kmeans.q_reordered

    qm = None
    k_mean = None
    correction = None
    if cfg.smoothing == "k_only":
        k_work, k_mean = smooth_k(k_work)
    elif cfg.smoothing == "full":
        k_work, k_mean = smooth_k(k_work)
        q_work, qm = group_mean_q(q_work, block_q=cfg.q_smooth_block_size)
        correction = torch.einsum("bqhd,bshd->bqhs", qm.float(), k_work.float()).to(
            cfg.correction_dtype
        )
    elif cfg.smoothing != "off":
        raise ValueError(f"unsupported smoothing mode: {cfg.smoothing!r}")

    if cfg.qk_quant == "fp8_block":
        q_quant, q_scale = fp8_block_quant(q_work, block_s=cfg.fp8_block_size)
        k_quant, k_scale = fp8_block_quant(k_work, block_s=cfg.fp8_block_size)
    elif cfg.qk_quant == "mxfp8":
        q_quant, q_scale = mxfp8_qk_quant(q_work, block_d=cfg.mxfp8_block_size)
        k_quant, k_scale = mxfp8_qk_quant(k_work, block_d=cfg.mxfp8_block_size)
    else:
        raise ValueError(f"unsupported qk_quant: {cfg.qk_quant!r}")

    if cfg.v_quant == "fp8_channel":
        v_quant, v_scale = fp8_per_channel_quant(v_work)
    elif cfg.v_quant == "fp8_block":
        v_quant, v_scale = fp8_block_quant(v_work, block_s=cfg.v_fp8_block_size)
    elif cfg.v_quant == "mxfp8":
        v_quant, v_scale = mxfp8_v_quant(v_work, block_s=cfg.v_mxfp8_block_size)
    else:
        raise ValueError(f"unsupported v_quant: {cfg.v_quant!r}")

    return PreparedQKV(
        q=q_quant,
        k=k_quant,
        v=v_quant,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        qm=qm,
        k_mean=k_mean,
        correction=correction,
        kmeans=kmeans,
    )
