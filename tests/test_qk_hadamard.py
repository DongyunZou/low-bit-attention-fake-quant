from __future__ import annotations

import pytest
import torch

from low_bit_fake_quant import QuantConfig, apply_qk_hadamard, fake_quant_attention
from low_bit_fake_quant.attention import _dequant_qk, _qk_inputs_for_quant, _quant_qk

CUDA_OK = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(not CUDA_OK, reason="CUDA required")


def _score_mse(q: torch.Tensor, k: torch.Tensor, cfg: QuantConfig) -> float:
    q_quant_in, k_quant_in = _qk_inputs_for_quant(q, k, cfg)
    q_fp8, q_scale, q_meta = _quant_qk(q_quant_in, cfg)
    k_fp8, k_scale, k_meta = _quant_qk(k_quant_in, cfg)
    q_deq = _dequant_qk(q_fp8, q_scale, q_meta, torch.bfloat16)
    k_deq = _dequant_qk(k_fp8, k_scale, k_meta, torch.bfloat16)
    q_ref = q.permute(0, 2, 1, 3).float()
    k_ref = k.permute(0, 2, 1, 3).float()
    q_quant = q_deq.permute(0, 2, 1, 3).float()
    k_quant = k_deq.permute(0, 2, 1, 3).float()
    ref_scores = torch.matmul(q_ref, k_ref.transpose(-2, -1))
    quant_scores = torch.matmul(q_quant, k_quant.transpose(-2, -1))
    return float((quant_scores - ref_scores).square().mean().item())


def test_qk_hadamard_preserves_unquantized_logits():
    gen = torch.Generator(device="cuda").manual_seed(0)
    q = torch.randn(1, 128, 2, 64, device="cuda", dtype=torch.float32, generator=gen)
    k = torch.randn(1, 128, 2, 64, device="cuda", dtype=torch.float32, generator=gen)
    q_rot, k_rot = apply_qk_hadamard(q, k, random_sign=True, seed=123)

    ref = torch.matmul(q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3).transpose(-2, -1))
    rot = torch.matmul(q_rot.permute(0, 2, 1, 3), k_rot.permute(0, 2, 1, 3).transpose(-2, -1))
    assert (rot - ref).abs().max().item() < 2e-4


def test_qk_hadamard_reduces_fp8_block_score_error_with_smoothing_off():
    gen = torch.Generator(device="cuda").manual_seed(1)
    b, s, h, d = 1, 256, 2, 64
    q = torch.randn(b, s, h, d, device="cuda", dtype=torch.bfloat16, generator=gen) * 0.2
    k = torch.randn(b, s, h, d, device="cuda", dtype=torch.bfloat16, generator=gen) * 0.2
    # Put the Q/K dynamic range into one channel. Hadamard rotation spreads it
    # across D while keeping exact logits unchanged before quantization.
    trend = torch.linspace(-30, 30, s, device="cuda", dtype=torch.bfloat16).view(1, s, 1)
    q[:, :, :, 0] += trend
    k[:, :, :, 0] -= trend

    base = QuantConfig(
        qk_quant="fp8_block",
        smoothing="off",
        q_kmeans_k=None,
        v_smooth_mode="off",
        qk_hadamard=False,
        p_requant=False,
    )
    hadamard = QuantConfig(
        qk_quant="fp8_block",
        smoothing="off",
        q_kmeans_k=None,
        v_smooth_mode="off",
        qk_hadamard=True,
        qk_hadamard_seed=0,
        p_requant=False,
    )

    base_mse = _score_mse(q, k, base)
    had_mse = _score_mse(q, k, hadamard)
    assert had_mse < base_mse * 0.25


@pytest.mark.parametrize("p_requant", [False, True])
def test_fake_quant_attention_qk_hadamard_runs_with_smoothing_off(p_requant):
    gen = torch.Generator(device="cuda").manual_seed(2)
    q = torch.randn(1, 256, 2, 64, device="cuda", dtype=torch.bfloat16, generator=gen)
    k = torch.randn(1, 256, 2, 64, device="cuda", dtype=torch.bfloat16, generator=gen)
    v = torch.randn(1, 256, 2, 64, device="cuda", dtype=torch.bfloat16, generator=gen)
    cfg = QuantConfig(
        qk_quant="fp8_block",
        v_quant="fp8_channel",
        smoothing="off",
        q_kmeans_k=None,
        v_smooth_mode="off",
        v_kmeans_k=None,
        qk_hadamard=True,
        p_requant=p_requant,
        p_requant_block_m=64,
        p_requant_block_n=64,
    )

    out = fake_quant_attention(q, k, v, cfg)
    assert out.shape == q.shape
    assert torch.isfinite(out).all()
