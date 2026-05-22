"""Correctness tests for V k-means reorder with co-permuted K.

Critical invariant: V kmeans permutes V along S, and K must be co-permuted by
the same permutation so attention output stays mathematically equivalent.
"""
from __future__ import annotations

import math

import pytest
import torch

from low_bit_fake_quant import (
    QuantConfig,
    apply_kv_permutation,
    fake_quant_attention,
    reference_attention,
    v_kmeans_reorder,
)

CUDA_OK = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(not CUDA_OK, reason="CUDA required")


def _make_qkv(b=1, s=512, h=2, d=64, dtype=torch.bfloat16, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(b, s, h, d, device="cuda", dtype=dtype, generator=g)
    k = torch.randn(b, s, h, d, device="cuda", dtype=dtype, generator=g)
    v = torch.randn(b, s, h, d, device="cuda", dtype=dtype, generator=g)
    return q, k, v


def _cosine(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def test_co_permuted_kv_preserves_attention_output():
    """Attention(Q, K, V) == Attention(Q, K[π_v], V[π_v]) for any permutation π_v."""
    q, k, v = _make_qkv(b=1, s=512, h=2, d=64)
    kmr = v_kmeans_reorder(v, n_clusters=32, max_iters=10, seed=0)
    v_re = kmr.tensor_reordered
    k_re = apply_kv_permutation(k, kmr.order)

    o_orig = reference_attention(q, k, v)
    o_perm = reference_attention(q, k_re, v_re)
    cos = _cosine(o_orig, o_perm)
    assert cos > 0.9999, f"co-permuted attention output mismatch: cos={cos}"
    diff = (o_orig.float() - o_perm.float()).abs().max().item()
    assert diff < 1e-2, f"max abs diff after co-permute = {diff}"


def test_v_kmeans_pipeline_runs_in_fake_quant_attention():
    q, k, v = _make_qkv(b=1, s=512, h=2, d=64)
    cfg = QuantConfig(
        qk_quant="mxfp8",
        v_quant="fp8_channel",
        smoothing="full",
        q_smooth_block_size=128,
        q_kmeans_k=32,
        v_smooth_mode="per_block",
        v_smooth_block_size=64,
        v_kmeans_k=32,
        p_requant=False,
    )
    o = fake_quant_attention(q, k, v, cfg)
    assert o.shape == q.shape
    assert torch.isfinite(o).all()
    ref = reference_attention(q, k, v)
    cos = _cosine(o, ref)
    assert cos > 0.95, f"fake-quant w/ V kmeans cos={cos:.4f}"


def test_v_kmeans_works_with_triton_p_requant():
    q, k, v = _make_qkv(b=1, s=512, h=2, d=64)
    cfg = QuantConfig(
        qk_quant="mxfp8",
        v_quant="fp8_channel",
        smoothing="full",
        q_smooth_block_size=128,
        q_kmeans_k=32,
        v_smooth_mode="per_block",
        v_smooth_block_size=64,
        v_kmeans_k=32,
        p_requant=True,
        p_requant_block_m=64,
        p_requant_block_n=64,
    )
    o = fake_quant_attention(q, k, v, cfg)
    assert torch.isfinite(o).all()
    ref = reference_attention(q, k, v)
    cos = _cosine(o, ref)
    assert cos > 0.93, f"fake-quant Triton w/ V kmeans cos={cos:.4f}"


def test_v_kmeans_helps_under_clustered_v_bias():
    """On synthetic data where V tokens form magnitude clusters, V kmeans +
    V smoothing should outperform V smoothing alone."""
    torch.manual_seed(0)
    b, s, h, d = 1, 1024, 2, 64
    q = torch.randn(b, s, h, d, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(b, s, h, d, device="cuda", dtype=torch.bfloat16)
    # V has 8 clusters with very different per-cluster magnitudes,
    # but tokens are shuffled along S.
    v_clusters = torch.randn(8, h, d, device="cuda", dtype=torch.float32)
    v_clusters = v_clusters * torch.tensor([0.1, 0.5, 1.0, 2.0, 5.0, 0.3, 1.5, 4.0],
                                            device="cuda", dtype=torch.float32).view(8, 1, 1)
    cluster_id = torch.randint(0, 8, (s,), device="cuda")
    v = (v_clusters[cluster_id] +
         0.1 * torch.randn(s, h, d, device="cuda", dtype=torch.float32)).to(torch.bfloat16)
    v = v.unsqueeze(0)  # (B=1, S, H, D)

    base = dict(
        qk_quant="fp8_block", v_quant="fp8_channel", smoothing="off",
        q_kmeans_k=None,
        v_smooth_mode="per_block", v_smooth_block_size=64,
        p_requant=False,
    )
    cfg_no_vkm = QuantConfig(**base, v_kmeans_k=None)
    cfg_vkm = QuantConfig(**base, v_kmeans_k=32)

    ref = reference_attention(q, k, v)
    o_no = fake_quant_attention(q, k, v, cfg_no_vkm)
    o_vk = fake_quant_attention(q, k, v, cfg_vkm)

    mse_no = (o_no.float() - ref.float()).pow(2).mean().item()
    mse_vk = (o_vk.float() - ref.float()).pow(2).mean().item()
    assert mse_vk <= mse_no * 1.05, (
        f"V kmeans did not help under clustered V: no_vkm MSE={mse_no:.3e}, "
        f"vkm MSE={mse_vk:.3e}"
    )
