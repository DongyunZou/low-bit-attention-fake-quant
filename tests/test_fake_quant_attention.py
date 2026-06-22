"""Smoke tests for fake_quant_attention on small synthetic shapes.

Verifies that:
* the kernel runs without error for every supported quant/smoothing/kmeans combo;
* output is finite;
* cosine similarity vs. torch SDPA is non-trivially close on random inputs
  (loose bound — FP8 with random data is noisier than BF16/FP32).
"""
from __future__ import annotations

import pytest
import torch

from low_bit_fake_quant import (
    QuantConfig,
    fake_quant_attention,
    reference_attention,
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


@pytest.mark.parametrize("qk_quant", ["fp8_block", "mxfp8"])
@pytest.mark.parametrize("smoothing", ["off", "k_only", "full"])
@pytest.mark.parametrize("kmeans_k", [None, 32])
@pytest.mark.parametrize("p_requant", [False, True])
def test_fake_quant_runs(qk_quant, smoothing, kmeans_k, p_requant):
    cfg = QuantConfig(
        qk_quant=qk_quant,
        v_quant="fp8_channel",
        smoothing=smoothing,
        q_kmeans_k=kmeans_k,
        q_smooth_block_size=128,
        fp8_block_size=128,
        mxfp8_block_size=32,
        p_requant=p_requant,
        p_requant_q_chunk=128,
    )
    q, k, v = _make_qkv(b=1, s=512, h=2, d=64)
    o = fake_quant_attention(q, k, v, cfg)
    assert o.shape == q.shape
    assert torch.isfinite(o).all()
    ref = reference_attention(q, k, v)
    cos = _cosine(o, ref)
    # FP8 random-data cosine should be at least decent; loose lower bound.
    assert cos > 0.93, f"cos={cos:.4f} for {qk_quant=}, {smoothing=}, {kmeans_k=}, {p_requant=}"


def test_p_requant_and_sdpa_paths_agree_closely_when_p_is_exact():
    """When p_requant is on the only extra error is the P FP8 cast.
    For small shapes with mild dynamic range the two paths should land
    within a small delta of each other.
    """
    cfg_a = QuantConfig(qk_quant="fp8_block", smoothing="off", q_kmeans_k=None,
                        q_smooth_block_size=128, p_requant=False)
    cfg_b = QuantConfig(qk_quant="fp8_block", smoothing="off", q_kmeans_k=None,
                        q_smooth_block_size=128, p_requant=True, p_requant_q_chunk=128)
    q, k, v = _make_qkv(b=1, s=256, h=2, d=64, seed=7)
    o_a = fake_quant_attention(q, k, v, cfg_a)
    o_b = fake_quant_attention(q, k, v, cfg_b)
    cos = _cosine(o_a, o_b)
    # Should be close — both quant Q/K/V the same; only P cast differs.
    assert cos > 0.99, f"P-requant vs SDPA path cosine={cos:.4f}"


@pytest.mark.parametrize("fill", ["mean_a1.5", "uta16_a1.5"])
def test_fake_quant_attention_runs_with_blasst_fill(fill):
    q, k, v = _make_qkv(b=1, s=256, h=2, d=64, seed=13)
    cfg = QuantConfig(
        qk_quant="fp8_block",
        v_quant="fp8_channel",
        smoothing="off",
        q_kmeans_k=None,
        q_smooth_block_size=128,
        fp8_block_size=128,
        p_requant=True,
        p_requant_block_m=64,
        p_requant_block_n=64,
        blasst_lambda=0.8,
        blasst_fill=fill,
    )
    out = fake_quant_attention(q, k, v, cfg)
    assert out.shape == q.shape
    assert torch.isfinite(out).all()


def test_blasst_fill_requires_p_requant_path():
    q, k, v = _make_qkv(b=1, s=256, h=2, d=64, seed=14)
    cfg = QuantConfig(
        qk_quant="fp8_block",
        smoothing="off",
        q_kmeans_k=None,
        q_smooth_block_size=128,
        p_requant=False,
        blasst_lambda=0.8,
        blasst_fill="mean_a1.5",
    )
    with pytest.raises(ValueError, match="requires cfg.p_requant=True"):
        fake_quant_attention(q, k, v, cfg)
