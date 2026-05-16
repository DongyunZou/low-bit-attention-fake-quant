from __future__ import annotations

from typing import Tuple

import torch
import triton
import triton.language as tl

_LOG2_E = 1.4426950408889634


@triton.jit
def _p_requant_rows_kernel(
    scores, p_fp8, row_sum, lse,
    stride_sb, stride_sm, stride_sn,
    stride_pb, stride_pm, stride_pn,
    B: tl.constexpr, M: tl.constexpr, N: tl.constexpr, BLOCK_N: tl.constexpr,
    SM_SCALE: tl.constexpr, P_MAX_OFFSET: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    offs_n = tl.arange(0, BLOCK_N)
    mask = offs_n < N
    vals = tl.load(
        scores + pid_b * stride_sb + pid_m * stride_sm + offs_n * stride_sn,
        mask=mask,
        other=-float("inf"),
    ).to(tl.float32)
    row_max = tl.max(vals, axis=0)
    z = (vals - row_max) * (SM_SCALE * _LOG2_E) + P_MAX_OFFSET
    p = tl.exp2(z)
    tl.store(
        p_fp8 + pid_b * stride_pb + pid_m * stride_pm + offs_n * stride_pn,
        p.to(p_fp8.dtype.element_ty),
        mask=mask,
    )
    rs = tl.sum(p, axis=0)
    tl.store(row_sum + pid_b * M + pid_m, rs)
    tl.store(lse + pid_b * M + pid_m, row_max * SM_SCALE + tl.log(rs) - P_MAX_OFFSET * 0.6931471805599453)


def p_requant_rows(
    scores: torch.Tensor,
    *,
    sm_scale: float,
    p_max_offset: int = 8,
    block_n: int | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Small-shape Triton probe for FP8 P requant.

    ``scores`` is ``(B, M, N)`` unscaled logits. This helper materializes
    P, so it is intentionally for unit tests and small probes, not Wan21.
    Full Wan21 testing should use the streaming attention kernel described
    in ``docs/quant_precision_test_plan.md``.
    """
    if scores.ndim != 3 or not scores.is_cuda:
        raise ValueError(f"scores must be 3-D CUDA tensor; got {tuple(scores.shape)}")
    b, m, n = scores.shape
    if block_n is None:
        block_n = triton.next_power_of_2(n)
    if block_n < n:
        raise ValueError(f"block_n={block_n} must be >= N={n}")
    p = torch.empty_like(scores, dtype=torch.float8_e4m3fn)
    row_sum = torch.empty((b, m), dtype=torch.float32, device=scores.device)
    lse = torch.empty((b, m), dtype=torch.float32, device=scores.device)
    _p_requant_rows_kernel[(b, m)](
        scores, p, row_sum, lse,
        scores.stride(0), scores.stride(1), scores.stride(2),
        p.stride(0), p.stride(1), p.stride(2),
        B=b, M=m, N=n, BLOCK_N=block_n,
        SM_SCALE=float(sm_scale), P_MAX_OFFSET=int(p_max_offset),
    )
    return p, row_sum, lse
