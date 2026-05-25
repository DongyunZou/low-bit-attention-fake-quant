"""Tests for V quant variants × V smoothing × P quant.

Three V quant modes — fp8_channel, fp8_block, mxfp8 — must all run end-to-end
through both SDPA path and Triton P-requant path, with optional V smoothing
on top.
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
    return (
        torch.randn(b, s, h, d, device="cuda", dtype=dtype, generator=g),
        torch.randn(b, s, h, d, device="cuda", dtype=dtype, generator=g),
        torch.randn(b, s, h, d, device="cuda", dtype=dtype, generator=g),
    )


def _cosine(a, b):
    a = a.float().flatten(); b = b.float().flatten()
    return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


@pytest.mark.parametrize("v_quant", ["fp8_channel", "fp8_block", "mxfp8"])
@pytest.mark.parametrize("v_smooth", ["off", "per_block"])
@pytest.mark.parametrize("p_requant", [False, True])
def test_v_quant_variants_run(v_quant, v_smooth, p_requant):
    q, k, v = _make_qkv(b=1, s=512, h=2, d=64)
    cfg = QuantConfig(
        qk_quant="fp8_block",
        v_quant=v_quant,
        smoothing="off",
        q_kmeans_k=None,
        q_smooth_block_size=128,
        fp8_block_size=128,
        mxfp8_block_size=32,
        v_fp8_block_size=64,
        v_mxfp8_block_size=64,
        v_smooth_mode=v_smooth,
        v_smooth_block_size=64,
        p_requant=p_requant,
        p_requant_block_m=64,
        p_requant_block_n=64,
    )
    o = fake_quant_attention(q, k, v, cfg)
    assert o.shape == q.shape
    assert torch.isfinite(o).all()
    ref = reference_attention(q, k, v)
    cos = _cosine(o, ref)
    assert cos > 0.93, f"{v_quant}/{v_smooth}/p_requant={p_requant}: cos={cos:.4f}"


def test_v_quant_with_full_q_smoothing_and_v_kmeans():
    """End-to-end best stack: every trick on, all three V quants should produce
    output close to the torch SDPA reference."""
    q, k, v = _make_qkv(b=1, s=512, h=2, d=64)
    ref = reference_attention(q, k, v)
    for v_quant in ["fp8_channel", "fp8_block", "mxfp8"]:
        cfg = QuantConfig(
            qk_quant="mxfp8",
            v_quant=v_quant,
            smoothing="full",
            q_smooth_block_size=128,
            q_kmeans_k=16,
            fp8_block_size=128,
            mxfp8_block_size=32,
            v_fp8_block_size=64,
            v_mxfp8_block_size=64,
            v_smooth_mode="per_block",
            v_smooth_block_size=64,
            v_kmeans_k=16,
            p_requant=True,
            p_requant_block_m=64,
            p_requant_block_n=64,
        )
        o = fake_quant_attention(q, k, v, cfg)
        cos = _cosine(o, ref)
        assert cos > 0.95, f"V={v_quant}: cos={cos:.4f}"


def test_p_quant_auto_picks_mx_for_mxfp8_v():
    """When V=mxfp8 and p_quant='auto', kernel should run with P=mx without errors."""
    q, k, v = _make_qkv(b=1, s=256, h=2, d=64)
    cfg = QuantConfig(
        qk_quant="mxfp8",
        v_quant="mxfp8",
        smoothing="off", q_kmeans_k=None,
        v_smooth_mode="off",
        v_mxfp8_block_size=64,
        p_quant="auto",  # should resolve to mx
        p_requant=True,
        p_requant_block_m=64,
        p_requant_block_n=64,
    )
    o = fake_quant_attention(q, k, v, cfg)
    assert torch.isfinite(o).all()


def test_p_quant_elementwise_for_fp8_block_v():
    """V=fp8_block uses P=elementwise; per-K-block s_V should be applied inside the loop."""
    q, k, v = _make_qkv(b=1, s=256, h=2, d=64)
    cfg = QuantConfig(
        qk_quant="mxfp8",
        v_quant="fp8_block",
        smoothing="off", q_kmeans_k=None,
        v_smooth_mode="off",
        v_fp8_block_size=64,
        p_quant="auto",
        p_requant=True,
        p_requant_block_m=64,
        p_requant_block_n=64,
    )
    o = fake_quant_attention(q, k, v, cfg)
    ref = reference_attention(q, k, v)
    cos = _cosine(o, ref)
    assert cos > 0.95, f"V=fp8_block: cos={cos:.4f}"


def test_dynamic_p_quant_with_qm_k_rowmax_matches_online_dynamic():
    """Using qm@K as rowmax estimate should be numerically close to online
    rowmax when P uses dynamic per-row/block scaling."""
    q, k, v = _make_qkv(b=1, s=512, h=2, d=64, seed=3)
    base = dict(
        qk_quant="mxfp8",
        v_quant="fp8_channel",
        smoothing="full",
        q_smooth_block_size=128,
        q_kmeans_k=16,
        p_quant="dynamic",
        p_requant=True,
        p_requant_block_m=64,
        p_requant_block_n=64,
    )
    online = fake_quant_attention(q, k, v, QuantConfig(**base, rowmax_mode="online"))
    estimated = fake_quant_attention(q, k, v, QuantConfig(**base, rowmax_mode="qm_k"))
    cos = _cosine(estimated, online)
    max_abs = (estimated.float() - online.float()).abs().max().item()
    assert cos > 0.9999, f"qm_k rowmax dynamic P vs online dynamic cosine={cos:.6f}"
    assert max_abs < 5e-3
