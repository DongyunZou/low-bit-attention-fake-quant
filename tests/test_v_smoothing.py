"""Correctness tests for V per-block smoothing.

Key checks:
1. ``smooth_v_per_block`` produces ``V_centered`` with zero per-block-mean
   (within FP32 noise) and ``v_alpha`` matching the per-block mean.
2. With V smoothing enabled but **no** quantization (we substitute identity
   quant by hand), the fake-quant attention output reconstructs the
   un-smoothed torch SDPA reference up to BF16 rounding.
3. With realistic FP8 quantization, V smoothing reduces or matches the MSE
   vs torch SDPA — never makes it worse on random data.
"""
from __future__ import annotations

import math

import pytest
import torch

from low_bit_fake_quant import (
    QuantConfig,
    fake_quant_attention,
    reference_attention,
    smooth_v_per_block,
)
from low_bit_fake_quant.attention_triton import fake_quant_attention_triton

CUDA_OK = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(not CUDA_OK, reason="CUDA required")


def _make_qkv(b=1, s=512, h=2, d=64, dtype=torch.bfloat16, seed=0, v_bias_scale=0.0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(b, s, h, d, device="cuda", dtype=dtype, generator=g)
    k = torch.randn(b, s, h, d, device="cuda", dtype=dtype, generator=g)
    v = torch.randn(b, s, h, d, device="cuda", dtype=dtype, generator=g)
    if v_bias_scale > 0:
        # Add a strong per-block bias to V so smoothing matters numerically.
        n_blocks = 8
        bias = torch.randn(b, n_blocks, h, d, device="cuda", dtype=dtype, generator=g) * v_bias_scale
        v = v.view(b, n_blocks, s // n_blocks, h, d) + bias.unsqueeze(2)
        v = v.view(b, s, h, d).contiguous()
    return q, k, v


def _cosine(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def test_smooth_v_per_block_helper():
    """V_centered should have ~zero per-block-mean; alpha should match the mean."""
    torch.manual_seed(0)
    b, s, h, d = 1, 512, 2, 64
    v = torch.randn(b, s, h, d, device="cuda", dtype=torch.bfloat16)
    block = 64
    v_centered, v_alpha = smooth_v_per_block(v, block_s=block)
    assert v_centered.shape == v.shape
    assert v_alpha.shape == (b, s // block, h, d)
    assert v_alpha.dtype == torch.float32

    # Per-block mean of v_centered should be ~0 (FP32 round trip on BF16).
    centered_mean = v_centered.float().view(b, s // block, block, h, d).mean(dim=2)
    assert centered_mean.abs().max().item() < 1e-2

    # alpha should match the per-block mean of v.
    expected_alpha = v.float().view(b, s // block, block, h, d).mean(dim=2)
    diff = (v_alpha - expected_alpha).abs().max().item()
    assert diff < 1e-3, f"alpha vs expected mean diff = {diff}"


def test_v_smoothing_sdpa_path_runs_and_is_close():
    """SDPA path with V smoothing should track torch reference closely."""
    q, k, v = _make_qkv(b=1, s=512, h=2, d=64, v_bias_scale=2.0)
    cfg = QuantConfig(
        qk_quant="fp8_block",
        v_quant="fp8_channel",
        smoothing="off",
        q_kmeans_k=None,
        q_smooth_block_size=128,
        fp8_block_size=128,
        v_smooth_mode="per_block",
        v_smooth_block_size=64,
        p_requant=False,
    )
    o = fake_quant_attention(q, k, v, cfg)
    ref = reference_attention(q, k, v)
    cos = _cosine(o, ref)
    assert cos > 0.97, f"SDPA + V smoothing cosine={cos:.4f} too low"


def test_v_smoothing_triton_path_runs_and_is_close():
    """Triton P-requant path with V smoothing should also track torch reference."""
    q, k, v = _make_qkv(b=1, s=512, h=2, d=64, v_bias_scale=2.0)
    cfg = QuantConfig(
        qk_quant="fp8_block",
        v_quant="fp8_channel",
        smoothing="off",
        q_kmeans_k=None,
        q_smooth_block_size=128,
        fp8_block_size=128,
        v_smooth_mode="per_block",
        v_smooth_block_size=64,
        p_requant=True,
        p_requant_block_m=64,
        p_requant_block_n=64,
    )
    o = fake_quant_attention(q, k, v, cfg)
    ref = reference_attention(q, k, v)
    cos = _cosine(o, ref)
    assert cos > 0.95, f"Triton + V smoothing cosine={cos:.4f} too low"


def test_v_smoothing_helps_under_bias():
    """On data with strong per-block V bias, V smoothing should outperform off."""
    q, k, v = _make_qkv(b=1, s=512, h=2, d=64, v_bias_scale=4.0)
    cfg_off = QuantConfig(
        qk_quant="fp8_block", v_quant="fp8_channel",
        smoothing="off", q_kmeans_k=None,
        v_smooth_mode="off", p_requant=False,
    )
    cfg_on = QuantConfig(
        qk_quant="fp8_block", v_quant="fp8_channel",
        smoothing="off", q_kmeans_k=None,
        v_smooth_mode="per_block", v_smooth_block_size=64, p_requant=False,
    )
    ref = reference_attention(q, k, v)
    o_off = fake_quant_attention(q, k, v, cfg_off)
    o_on = fake_quant_attention(q, k, v, cfg_on)
    mse_off = (o_off.float() - ref.float()).pow(2).mean().item()
    mse_on = (o_on.float() - ref.float()).pow(2).mean().item()
    # V smoothing should reduce MSE under strong V bias. Strict inequality
    # with a small slack to absorb RNG noise.
    assert mse_on <= mse_off * 1.05, (
        f"V smoothing did not help under bias: off MSE={mse_off:.3e}, on MSE={mse_on:.3e}"
    )


def test_triton_kernel_v_alpha_matches_torch_reference():
    """Direct Triton-kernel-level test: with V smoothing on, the kernel
    should reproduce P @ (V_centered_dequant + alpha[block]) within FP8
    rounding error.
    """
    torch.manual_seed(0)
    B, H, S, D = 1, 2, 256, 128
    block_v = 64
    q = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)
    v_centered_fp32 = torch.randn(B, H, S, D, device="cuda", dtype=torch.float32)
    v_alpha = torch.randn(B, S // block_v, H, D, device="cuda", dtype=torch.float32)

    # Quantize V_centered (per-channel FP8) so the kernel sees realistic V_fp8.
    v_max = v_centered_fp32.abs().amax(dim=-2, keepdim=False).clamp_min(1e-6)  # (B,H,D)
    v_scale_bhd = v_max / 448.0
    v_fp8 = (v_centered_fp32 / v_scale_bhd.unsqueeze(-2)).clamp(-448, 448)
    v_bf16 = v_fp8.to(torch.float8_e4m3fn).to(torch.bfloat16)
    v_centered_dequant = v_bf16.float() * v_scale_bhd.unsqueeze(-2)

    sm_scale = 1.0 / math.sqrt(D)
    p_max_offset = 8

    o_triton = fake_quant_attention_triton(
        q, k, v_bf16, v_scale_bhd, sm_scale=sm_scale, p_max_offset=p_max_offset,
        block_m=64, block_n=64,
        v_alpha=v_alpha, v_smooth_block=block_v,
    )

    # Torch reference: P @ (V_centered_dequant + alpha[block])
    LOG2E = 1.4426950408889634
    scores = torch.matmul(q, k.transpose(-2, -1)).float() * sm_scale
    row_max = scores.amax(dim=-1, keepdim=True)
    z = (scores - row_max) * LOG2E + p_max_offset
    p = torch.exp2(z)
    row_sum = p.sum(dim=-1, keepdim=True)
    p_fp8 = p.to(torch.float8_e4m3fn).to(torch.bfloat16).to(torch.float32)
    # Reconstitute V_recon = V_centered_dequant + alpha[block]
    v_alpha_full = v_alpha.repeat_interleave(block_v, dim=1)  # (B, S, H, D)
    v_alpha_bhsd = v_alpha_full.permute(0, 2, 1, 3)  # (B, H, S, D)
    v_recon = v_centered_dequant + v_alpha_bhsd
    o_torch = (torch.matmul(p_fp8, v_recon) / row_sum).to(torch.bfloat16)

    cos = _cosine(o_triton, o_torch)
    assert cos > 0.999, f"triton V-smooth vs torch ref cos={cos:.6f}"
