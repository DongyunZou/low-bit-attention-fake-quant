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


def _blasst_fill_p(
    mode: str,
    *,
    s_block: torch.Tensor,
    block_max: torch.Tensor,
    m_i: torch.Tensor,
    skip_row: torch.Tensor,
    lam: float,
    p_max_offset: int,
) -> torch.Tensor:
    """Torch mirror of the Triton BLASST fill modes for elementwise P quant."""
    if mode == "zero":
        return torch.zeros_like(s_block)

    Bn = s_block.shape[-1]
    fill_offset = float(p_max_offset)
    max_bound = Bn * torch.exp2((block_max - m_i) * 1.4426950408889634 + fill_offset)

    if mode == "max_a0.25":
        mass = max_bound * 0.25
        p_fill = mass[..., None] / Bn
    elif mode == "mean_a1.5":
        mean = s_block.mean(dim=-1)
        mass = 1.5 * Bn * torch.exp2((mean - m_i) * 1.4426950408889634 + fill_offset)
        p_fill = mass.clamp_max(max_bound)[..., None] / Bn
    elif mode == "logn":
        mean = s_block.mean(dim=-1)
        var = s_block.var(dim=-1, unbiased=False)
        mass = Bn * torch.exp2((mean + 0.5 * var - m_i) * 1.4426950408889634 + fill_offset)
        p_fill = mass.clamp_max(max_bound)[..., None] / Bn
    elif mode == "sample8_a1.25":
        step = max(1, Bn // 8)
        sample = s_block[..., ::step]
        mass = 1.25 * Bn * torch.exp2((sample - m_i[..., None]) * 1.4426950408889634 + fill_offset).mean(dim=-1)
        p_fill = mass.clamp_max(max_bound)[..., None] / Bn
    elif mode == "thr_a0.25":
        mass = 0.25 * Bn * lam * (2.0 ** p_max_offset)
        p_fill = torch.full_like(s_block, mass / Bn)
    elif mode == "uta16_a1.5":
        bins = 16
        assert Bn % bins == 0
        bin_size = Bn // bins
        sb = s_block.view(*s_block.shape[:-1], bins, bin_size)
        mean = sb.mean(dim=-1)
        max_ = sb.amax(dim=-1)
        bound = bin_size * torch.exp2((max_ - m_i[..., None]) * 1.4426950408889634 + fill_offset)
        mass = 1.5 * bin_size * torch.exp2((mean - m_i[..., None]) * 1.4426950408889634 + fill_offset)
        p_fill = (mass.clamp_max(bound) / bin_size).repeat_interleave(bin_size, dim=-1)
    else:
        raise ValueError(mode)

    return torch.where(skip_row[..., None], p_fill, torch.zeros_like(s_block))


def _torch_blasst_fill_reference(
    q_bhsd,
    k_bhsd,
    v_bf16_bhsd,
    v_scale_bhd,
    *,
    sm_scale,
    p_max_offset,
    block_m,
    block_n,
    lam,
    fill_mode,
):
    B, H, S, D = q_bhsd.shape
    LOG2E = 1.4426950408889634
    out = torch.empty_like(q_bhsd, dtype=torch.float32)
    v = v_bf16_bhsd.float()
    for q0 in range(0, S, block_m):
        q1 = min(q0 + block_m, S)
        qb = q_bhsd[:, :, q0:q1, :]
        rows = q1 - q0
        m_i = torch.full((B, H, rows), -float("inf"), dtype=torch.float32, device=q_bhsd.device)
        l_i = torch.zeros((B, H, rows), dtype=torch.float32, device=q_bhsd.device)
        acc = torch.zeros((B, H, rows, D), dtype=torch.float32, device=q_bhsd.device)
        for n0 in range(0, S, block_n):
            n1 = min(n0 + block_n, S)
            s = torch.matmul(qb.float(), k_bhsd[:, :, n0:n1, :].float().transpose(-2, -1)) * sm_scale
            block_max = s.amax(dim=-1)
            m_ij = torch.maximum(m_i, block_max)
            skip_row = (block_max - m_ij) < math.log(lam)
            alpha = torch.exp2((m_i - m_ij) * LOG2E)
            alpha = torch.where(skip_row, torch.ones_like(alpha), alpha)
            m_ij = torch.where(skip_row, m_i, m_ij)

            p = torch.exp2((s - m_ij[..., None]) * LOG2E + p_max_offset)
            p = torch.where(skip_row[..., None], torch.zeros_like(p), p)
            p = p + _blasst_fill_p(
                fill_mode,
                s_block=s,
                block_max=block_max,
                m_i=m_ij,
                skip_row=skip_row,
                lam=lam,
                p_max_offset=p_max_offset,
            )
            l_i = l_i * alpha + p.sum(dim=-1)
            p_fp8 = p.to(torch.float8_e4m3fn).to(torch.bfloat16).float()
            pv = torch.matmul(p_fp8, v[:, :, n0:n1, :])
            acc = acc * alpha[..., None] + pv
            m_i = m_ij
        out[:, :, q0:q1, :] = acc * v_scale_bhd[:, :, None, :] / l_i[..., None]
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


@pytest.mark.parametrize(
    "fill_mode",
    ["max_a0.25", "mean_a1.5", "logn", "sample8_a1.25", "thr_a0.25", "uta16_a1.5"],
)
def test_triton_blasst_fill_matches_torch_reference(fill_mode):
    torch.manual_seed(12)
    B, H, S, D = 1, 1, 128, 64
    block_m = block_n = 32
    u = torch.randn(D, device="cuda", dtype=torch.float32)
    u = u / u.norm()
    q = (u.view(1, 1, 1, D) + 0.01 * torch.randn(B, H, S, D, device="cuda")).to(torch.bfloat16)
    k = (0.05 * torch.randn(B, H, S, D, device="cuda")).to(torch.float32)
    k[:, :, :block_n, :] = 4.0 * u.view(1, 1, 1, D)
    k = k.to(torch.bfloat16)
    v = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)
    v_scale = torch.ones((B, H, D), dtype=torch.float32, device="cuda")
    sm_scale = 1.0 / math.sqrt(D)
    lam = 0.8
    p_max_offset = 8

    o_triton, stats = fake_quant_attention_triton(
        q,
        k,
        v,
        v_scale,
        sm_scale=sm_scale,
        p_max_offset=p_max_offset,
        block_m=block_m,
        block_n=block_n,
        p_quant_mode="elementwise",
        blasst_lambda=lam,
        blasst_fill_mode=fill_mode,
        return_blasst_stats=True,
    )
    o_torch = _torch_blasst_fill_reference(
        q,
        k,
        v,
        v_scale,
        sm_scale=sm_scale,
        p_max_offset=p_max_offset,
        block_m=block_m,
        block_n=block_n,
        lam=lam,
        fill_mode=fill_mode,
    )
    assert stats[..., 0].sum().item() > 0
    cos = _cosine(o_triton, o_torch)
    assert cos > 0.999, f"{fill_mode}: triton vs torch cosine={cos:.6f}"
    diff = (o_triton.float() - o_torch.float()).abs().max().item()
    assert diff < 7e-2, f"{fill_mode}: max abs diff {diff:.3e} too large"


@pytest.mark.parametrize("p_quant_mode", ["mx", "dynamic"])
def test_triton_blasst_fill_runs_with_scaled_p_quant(p_quant_mode):
    torch.manual_seed(21)
    B, H, S, D = 1, 1, 128, 64
    q = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)
    v_scale = torch.ones((B, H, D), dtype=torch.float32, device="cuda")
    out = fake_quant_attention_triton(
        q,
        k,
        v,
        v_scale,
        sm_scale=1.0 / math.sqrt(D),
        p_max_offset=8,
        block_m=32,
        block_n=32,
        p_quant_mode=p_quant_mode,
        p_mx_block_n=32 if p_quant_mode == "mx" else 0,
        blasst_lambda=0.8,
        blasst_fill_mode="mean_a1.5",
    )
    assert out.shape == q.shape
    assert torch.isfinite(out).all()
