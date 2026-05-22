"""Verify that the Triton fake-quant attention kernel agrees with a torch
chunked reference that does the same P→FP8→back cast.
"""
from __future__ import annotations

import math

import pytest
import torch

from low_bit_fake_quant import QuantConfig, fake_quant_attention
from low_bit_fake_quant.attention_triton import fake_quant_attention_triton

CUDA_OK = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(not CUDA_OK, reason="CUDA required")


def _cosine(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def _torch_chunked_p_requant(q_bhsd, k_bhsd, v_bf16_bhsd, v_scale_bhd, sm_scale, p_max_offset, q_chunk=64):
    """Reference torch implementation of the same algorithm."""
    B, H, S, D = q_bhsd.shape
    LOG2E = 1.4426950408889634
    sm_log2 = sm_scale * LOG2E
    v_unscaled_fp32 = v_bf16_bhsd.to(torch.float32)
    vsc = v_scale_bhd.view(B, H, 1, D).to(torch.float32)
    out = torch.empty_like(q_bhsd, dtype=torch.float32)
    for q0 in range(0, S, q_chunk):
        q1 = min(q0 + q_chunk, S)
        qb = q_bhsd[:, :, q0:q1, :]
        scores = torch.matmul(qb, k_bhsd.transpose(-2, -1)).float() * sm_scale
        row_max = scores.amax(dim=-1, keepdim=True)
        z = (scores - row_max) * LOG2E + p_max_offset
        p = torch.exp2(z)
        row_sum = p.sum(dim=-1, keepdim=True)
        p_fp8 = p.to(torch.float8_e4m3fn).to(torch.bfloat16).to(torch.float32)
        pv = torch.matmul(p_fp8, v_unscaled_fp32)
        out[:, :, q0:q1, :] = pv * vsc / row_sum
    return out.to(torch.bfloat16)


def test_triton_matches_torch_reference():
    torch.manual_seed(0)
    B, H, S, D = 1, 2, 256, 128
    q = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)
    v_fp = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)
    # Synthesize V as if cast through FP8: pick a scale, quant→dequant.
    v_scale = v_fp.float().abs().amax(dim=-2, keepdim=False) / 448.0  # (B,H,D)
    v_scale = v_scale.clamp_min(1e-6)
    v_fp8_vals = (v_fp.float() / v_scale.unsqueeze(-2)).clamp(-448, 448)
    v_bf16 = v_fp8_vals.to(torch.float8_e4m3fn).to(torch.bfloat16)
    sm_scale = 1.0 / math.sqrt(D)
    p_max_offset = 8

    o_triton = fake_quant_attention_triton(
        q, k, v_bf16, v_scale, sm_scale=sm_scale, p_max_offset=p_max_offset, block_m=64, block_n=64
    )
    o_torch = _torch_chunked_p_requant(q, k, v_bf16, v_scale, sm_scale, p_max_offset, q_chunk=64)
    cos = _cosine(o_triton, o_torch)
    assert cos > 0.999, f"triton vs torch chunked cosine={cos:.6f}"
    diff = (o_triton.float() - o_torch.float()).abs().max().item()
    assert diff < 5e-2, f"max abs diff {diff:.3e} too large"
