"""Correctness tests for the FP8 block-skip attention simulator.

Each test pins down one falsifiable property of the simulator: the no-quant
path matches a dense FP32 oracle, the skip predicate reproduces BLASST exactly
(running max, pre-softmax, empty-row safeguard), and the FP8 numeric rules hold.
"""

from __future__ import annotations

import math

import pytest
import torch

import importlib.util
from pathlib import Path

from low_bit_fake_quant.blasst_skip import (
    LEVEL_FP8_STATIC_P,
    LEVEL_REFERENCE,
    MATH_BACKEND_SEQLEN_LIMIT,
    FullMatrixAllocationError,
    apply_token_permutation,
    blasst_keep_mask,
    blasst_tile_keep_mask,
    choose_local_block,
    fake_quant_per_head,
    guard_no_full_matrix,
    invert_permutation,
    running_row_max,
    sdpa_ground_truth,
    select_sdpa_backends,
    simulate_workload,
    space_time_reorder_index,
    static_p_quant,
)
from torch.nn.attention import SDPBackend

_RANK_PATH = Path(__file__).resolve().parents[1] / "bench/rank_reorder_layouts.py"


def _load_select_layout():
    spec = importlib.util.spec_from_file_location("rank_reorder_layouts", _RANK_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.select_layout

_AR_PATH = Path(__file__).resolve().parents[2] / "low-bit-flash-attention/flash_attn/cute/testing.py"


def _load_attention_ref():
    """Load FlashAttention's fp32 ``attention_ref`` oracle directly by file path.

    The package __init__ pulls a compiled CUDA extension that is not built here,
    so we import the standalone testing module instead.
    """
    if not _AR_PATH.exists():
        return None
    spec = importlib.util.spec_from_file_location("fa_cute_testing", _AR_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.attention_ref

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _dense_reference(q, k, v):
    """Dense FP32 softmax attention oracle for a small cropped workload."""
    b, s, h, d = q.shape
    qf = q[0].float().permute(1, 0, 2)            # (H, S, D)
    kf = k[0].float().permute(1, 0, 2)
    vf = v[0].float().permute(1, 0, 2)
    scores = torch.matmul(qf, kf.transpose(1, 2)) / math.sqrt(d)
    p = torch.softmax(scores, dim=-1)
    o = torch.matmul(p, vf)                        # (H, S, D)
    return o.permute(1, 0, 2).unsqueeze(0)         # (1, S, H, D)


# ----------------------------------------------------------------------------
# Reference path vs dense FP32 oracle (cropped)
# ----------------------------------------------------------------------------


@CUDA
def test_reference_matches_dense_fp32_oracle():
    torch.manual_seed(0)
    s, h, d = 512, 3, 64
    q = torch.randn(1, s, h, d, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, s, h, d, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, s, h, d, device="cuda", dtype=torch.bfloat16)
    res = simulate_workload(
        q, k, v, skip_thresholds=[], levels=[LEVEL_REFERENCE],
        matmul_dtype=torch.float32,
    )
    ref = _dense_reference(q, k, v)
    pred = res.outputs[LEVEL_REFERENCE]
    diff = (pred - ref).float()
    rel_rmse = float(diff.norm() / ref.float().norm())
    cos = float(
        torch.dot(pred.reshape(-1).float(), ref.reshape(-1).float())
        / pred.float().norm() / ref.float().norm()
    )
    assert cos >= 0.9999, cos
    assert rel_rmse <= 1e-3, rel_rmse


@CUDA
def test_wrong_softmax_scale_fails_oracle():
    """Negative: a 1/d scale (instead of 1/sqrt(d)) breaks the oracle match."""
    torch.manual_seed(1)
    s, h, d = 256, 2, 64
    q = torch.randn(1, s, h, d, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, s, h, d, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, s, h, d, device="cuda", dtype=torch.bfloat16)
    ref = _dense_reference(q, k, v)
    # emulate a wrong scale by scaling q so QK uses 1/d instead of 1/sqrt(d)
    q_wrong = q.float() * math.sqrt(d) / d
    res = simulate_workload(
        q_wrong.to(torch.bfloat16), k, v, skip_thresholds=[],
        levels=[LEVEL_REFERENCE], matmul_dtype=torch.float32,
    )
    pred = res.outputs[LEVEL_REFERENCE]
    cos = float(
        torch.dot(pred.reshape(-1).float(), ref.reshape(-1).float())
        / pred.float().norm() / ref.float().norm()
    )
    assert cos < 0.999, cos


@CUDA
def test_cropped_fp32_matches_attention_ref_4096():
    # Cropped fp32 oracle: length-4096 no-quant/no-skip reference vs FlashAttention
    # attention_ref(upcast=True), the specified oracle.
    attention_ref = _load_attention_ref()
    if attention_ref is None:
        pytest.skip("attention_ref oracle not available")
    torch.manual_seed(7)
    s, h, d = 4096, 2, 128                       # head_dim 128 matches the study
    # fp32 inputs so attention_ref(upcast=True) keeps an fp32 output dtype
    # (it casts its result back to the input dtype), giving a clean fp32-vs-fp32
    # comparison rather than one bounded by bf16 output rounding.
    q = torch.randn(1, s, h, d, device="cuda", dtype=torch.float32)
    k = torch.randn(1, s, h, d, device="cuda", dtype=torch.float32)
    v = torch.randn(1, s, h, d, device="cuda", dtype=torch.float32)
    res = simulate_workload(q, k, v, skip_thresholds=[], levels=[LEVEL_REFERENCE],
                            matmul_dtype=torch.float32)
    pred = res.outputs[LEVEL_REFERENCE]
    out_ref = attention_ref(q, k, v, causal=False, upcast=True)
    ref = out_ref[0] if isinstance(out_ref, tuple) else out_ref   # (1, S, H, D)
    diff = (pred - ref).float()
    rel_rmse = float(diff.norm() / ref.float().norm())
    cos = float(torch.dot(pred.reshape(-1).float(), ref.reshape(-1).float())
                / pred.float().norm() / ref.float().norm())
    assert cos >= 0.9999, cos
    assert rel_rmse <= 1e-3, rel_rmse


def _bad_rescale_online_attention(q, k, v, scale):
    """Online-softmax attention with an intentionally WRONG accumulator rescale.

    The correct flash rescale on a max increase from m to m_new multiplies the
    running denominator/accumulator by exp(m - m_new) (<= 1). This buggy variant
    uses exp(m_new - m) (>= 1), inflating earlier blocks, so the normalization is
    wrong while the output stays finite. Inputs ``(1, S, H, D)`` fp32; keys are
    streamed in 128-blocks.
    """
    qh = q[0].permute(1, 0, 2).float()       # (H, S, D)
    kh = k[0].permute(1, 0, 2).float()
    vh = v[0].permute(1, 0, 2).float()
    H, S, D = qh.shape
    m = denom = acc = None
    for j in range(0, S, 128):
        kb = kh[:, j:j + 128, :]
        vb = vh[:, j:j + 128, :]
        sj = torch.matmul(qh, kb.transpose(1, 2)) * scale     # (H, S, nb_keys)
        blk_max = sj.amax(dim=-1)
        if m is None:
            m = blk_max
            p = torch.exp(sj - m.unsqueeze(-1))
            denom = p.sum(dim=-1)
            acc = torch.matmul(p, vb)
        else:
            m_new = torch.maximum(m, blk_max)
            alpha = torch.exp(m_new - m)                      # BUG: should be exp(m - m_new)
            p = torch.exp(sj - m_new.unsqueeze(-1))
            denom = denom * alpha + p.sum(dim=-1)
            acc = acc * alpha.unsqueeze(-1) + torch.matmul(p, vb)
            m = m_new
    o = acc / denom.unsqueeze(-1)
    return o.permute(1, 0, 2).unsqueeze(0)


@CUDA
def test_bad_online_rescale_fails_attention_ref():
    # Negative: a wrong online-softmax accumulator rescale must fail the fp32
    # attention_ref oracle on a cropped case.
    attention_ref = _load_attention_ref()
    if attention_ref is None:
        pytest.skip("attention_ref oracle not available")
    torch.manual_seed(8)
    s, h, d = 2048, 2, 128
    q = torch.randn(1, s, h, d, device="cuda", dtype=torch.float32)
    k = torch.randn(1, s, h, d, device="cuda", dtype=torch.float32)
    v = torch.randn(1, s, h, d, device="cuda", dtype=torch.float32)
    bad = _bad_rescale_online_attention(q, k, v, 1.0 / math.sqrt(d))
    out_ref = attention_ref(q, k, v, causal=False, upcast=True)
    ref = out_ref[0] if isinstance(out_ref, tuple) else out_ref
    diff = (bad - ref).float()
    rel_rmse = float(diff.norm() / ref.float().norm())
    cos = float(torch.dot(bad.reshape(-1).float(), ref.reshape(-1).float())
                / bad.float().norm() / ref.float().norm())
    # must FAIL at least one registered threshold
    assert (cos < 0.9999) or (rel_rmse > 1e-3), (cos, rel_rmse)


@CUDA
def test_forced_math_backend_rejected_at_full_seqlen():
    # Negative: forcing the math SDPA backend at full seqlen is flagged
    # (raised) up front rather than silently attempted (which would OOM).
    s = MATH_BACKEND_SEQLEN_LIMIT
    # tiny head_dim so the guard fires before any large allocation
    q = torch.zeros(1, s, 1, 8, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(FullMatrixAllocationError):
        sdpa_ground_truth(q, q, q, force_math_backend=True)
    # below the limit, forced math is allowed (small shapes are safe)
    qs = torch.randn(1, 256, 1, 8, device="cuda", dtype=torch.bfloat16)
    out = sdpa_ground_truth(qs, qs, qs, force_math_backend=True)
    assert out.shape == qs.shape


# ----------------------------------------------------------------------------
# BLASST skip predicate
# ----------------------------------------------------------------------------


def test_keep_mask_uses_running_not_final_max():
    # block local maxima where block 0 sets the running max, later blocks larger
    block_max = torch.tensor([[1.0, 5.0, 3.0]])
    logt = math.log(0.1)  # ~ -2.30
    keep_running = blasst_keep_mask(block_max, logt)
    # running max = [1, 5, 5]; margins = [0, 0, -2] -> all >= -2.30 -> all kept
    assert keep_running.tolist() == [[True, True, True]]
    # final-max variant would use 5 everywhere: margins [-4, 0, -2]
    final_margin = block_max - block_max.amax(dim=-1, keepdim=True)
    keep_final = final_margin >= logt
    assert keep_final.tolist() == [[False, True, True]]
    # the two predicates disagree -> running-max is the BLASST one
    assert keep_running.tolist() != keep_final.tolist()


def test_tile_keep_mask_is_all_rows_reduction():
    # 2 rows, 2 key blocks. Row 0 finds block1 important, row 1 does not.
    # block_max_per_row[row, block]
    block_max = torch.tensor([
        [0.0, 0.0],   # row 0: blocks tie -> running [0,0], margins [0,0]
        [5.0, 0.0],   # row 1: running [5,5], margins [0,-5]
    ]).unsqueeze(0)   # (1, rows=2, n_blocks=2)
    logt = math.log(0.1)  # -2.30
    keep = blasst_tile_keep_mask(block_max, logt)   # (1, n_blocks)
    # per-row margins: row0 [0,0], row1 [0,-5]; max over rows -> [0, 0]
    # tile margin block1 = max(0, -5) = 0 >= -2.30 -> kept (row 0 still needs it)
    assert keep.tolist() == [[True, True]]
    # the wrong (rowmax-rowmax) reduction would give block1 margin
    # = max([0,0]) - max([5,0])?? -> demonstrate all-rows keeps block1
    # Now make block1 negligible for BOTH rows:
    block_max2 = torch.tensor([[5.0, 0.0], [5.0, 0.0]]).unsqueeze(0)
    keep2 = blasst_tile_keep_mask(block_max2, logt)
    # margins both rows [0, -5]; max over rows [0, -5]; block1 -5 < -2.30 -> drop
    assert keep2.tolist() == [[True, False]]


def test_running_row_max_is_cummax():
    bm = torch.tensor([[2.0, 1.0, 4.0, 3.0]])
    assert running_row_max(bm).tolist() == [[2.0, 2.0, 4.0, 4.0]]


@CUDA
def test_skip_zero_threshold_equals_no_skip_bitwise():
    torch.manual_seed(2)
    s, h, d = 512, 2, 64
    q = torch.randn(1, s, h, d, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, s, h, d, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, s, h, d, device="cuda", dtype=torch.bfloat16)
    res = simulate_workload(
        q, k, v, skip_thresholds=[0.0],
        levels=[LEVEL_FP8_STATIC_P, "fp8_static_p_skip"],
    )
    no_skip = res.outputs[LEVEL_FP8_STATIC_P]
    skip0 = res.skip_outputs[0.0]
    assert torch.equal(no_skip, skip0)
    assert res.skip_diagnostics[0.0].skip_rate == 0.0


@CUDA
def test_skip_increases_with_threshold():
    # Structured workload: the first key block dominates (large-norm keys),
    # later blocks have near-zero keys, so every query row finds them
    # negligible and the tile-level (all-rows) predicate drops them.
    torch.manual_seed(3)
    s, h, d = 1024, 2, 64
    q = torch.randn(1, s, h, d, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, s, h, d, device="cuda", dtype=torch.bfloat16)
    k[:, :128] *= 8.0       # dominant key block
    k[:, 128:] *= 0.02      # negligible remaining blocks
    v = torch.randn(1, s, h, d, device="cuda", dtype=torch.bfloat16)
    res = simulate_workload(
        q, k, v, skip_thresholds=[0.0, 0.1, 0.9],
        levels=["fp8_static_p_skip"],
    )
    r0 = res.skip_diagnostics[0.0].skip_rate
    r1 = res.skip_diagnostics[0.1].skip_rate
    r2 = res.skip_diagnostics[0.9].skip_rate
    assert r0 == 0.0
    assert r0 <= r1 <= r2
    assert r2 > 0.0
    assert r2 <= 1.0


@CUDA
def test_empty_row_safeguard_forces_a_block():
    # threshold = 1.0 -> log = 0 -> keep only blocks whose local max equals the
    # running max; every row still keeps at least its first block via cummax,
    # but the safeguard must guarantee no row is fully empty and outputs finite.
    torch.manual_seed(4)
    s, h, d = 512, 2, 64
    q = torch.randn(1, s, h, d, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, s, h, d, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, s, h, d, device="cuda", dtype=torch.bfloat16)
    res = simulate_workload(
        q, k, v, skip_thresholds=[1.0], levels=["fp8_static_p_skip"],
    )
    out = res.skip_outputs[1.0]
    assert torch.isfinite(out).all()


def test_reorder_is_non_identity_and_blocks_are_local():
    # The space-time reorder must NOT be the identity; for the planned
    # 20x48x72 Wan grid the 128-token block is (2,8,8).
    t, h, w = 20, 48, 72
    assert t * h * w == 69120
    block = choose_local_block(t, h, w)
    assert block[0] * block[1] * block[2] == 128
    perm = space_time_reorder_index(t, h, w, block=block)
    ident = torch.arange(t * h * w)
    assert not torch.equal(perm, ident)               # genuinely reorders
    assert sorted(perm.tolist()) == list(range(t * h * w))  # still a permutation
    # the first 128-token block must span >1 temporal/spatial coordinate
    first = perm[:128]
    tt = first // (h * w)
    assert tt.unique().numel() == block[0]             # spans bt temporal slices


@CUDA
def test_reorder_lambda0_equivalence():
    # With reordering applied but skip disabled, the output after inverse
    # permutation equals the native-order no-skip output, proving the
    # permutation is a pure reindexing.
    torch.manual_seed(6)
    t, h, w = 8, 8, 16           # 1024 tokens = 8 blocks of 128
    s = t * h * w
    nh, d = 2, 64
    q = torch.randn(1, s, nh, d, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, s, nh, d, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, s, nh, d, device="cuda", dtype=torch.bfloat16)
    native = simulate_workload(q, k, v, skip_thresholds=[], levels=[LEVEL_REFERENCE])
    o_native = native.outputs[LEVEL_REFERENCE]

    perm = space_time_reorder_index(t, h, w, device=q.device)
    assert not torch.equal(perm, torch.arange(s, device=q.device))  # non-identity
    inv = invert_permutation(perm)
    qp = apply_token_permutation(q, perm)
    kp = apply_token_permutation(k, perm)
    vp = apply_token_permutation(v, perm)
    reordered = simulate_workload(qp, kp, vp, skip_thresholds=[], levels=[LEVEL_REFERENCE])
    o_restored = apply_token_permutation(reordered.outputs[LEVEL_REFERENCE], inv)

    cos = float(
        torch.dot(o_restored.reshape(-1).float(), o_native.reshape(-1).float())
        / o_restored.float().norm() / o_native.float().norm()
    )
    assert cos >= 0.9999, cos
    # Negative: omitting the inverse permutation must NOT match native order
    o_no_inverse = reordered.outputs[LEVEL_REFERENCE]
    cos_bad = float(
        torch.dot(o_no_inverse.reshape(-1).float(), o_native.reshape(-1).float())
        / o_no_inverse.float().norm() / o_native.float().norm()
    )
    assert cos_bad < 0.9999, cos_bad


# ----------------------------------------------------------------------------
# FP8 numerics
# ----------------------------------------------------------------------------


@CUDA
def test_per_head_quant_scale_shape_and_saturation():
    torch.manual_seed(5)
    s, h, d = 256, 4, 64
    x = torch.randn(s, h, d, device="cuda", dtype=torch.bfloat16) * 7.0
    deq, scale, stats = fake_quant_per_head(x)
    assert scale.shape == (1, h, 1)            # per-head, not per-channel
    # amax maps to E4M3_MAX exactly, so saturation is ~0.
    assert stats.saturation_rate < 1e-4
    assert deq.shape == x.shape


def test_static_p_quant_rule():
    p = torch.linspace(0, 1, 257)
    pq = static_p_quant(p)
    # exact rule: dequant(quant(p*256, e4m3)) / 256
    manual = (p * 256.0).to(torch.float8_e4m3fn).to(torch.float32) / 256.0
    assert torch.equal(pq, manual)
    assert pq.max() <= 1.5  # stays in a sane range


@CUDA
def test_tile_size_guard_rejects_non_multiple():
    q = torch.randn(1, 200, 2, 64, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError):
        simulate_workload(q, q, q, skip_thresholds=[], levels=[LEVEL_REFERENCE])


def test_token_permutation_roundtrip():
    t, h, w = 4, 8, 8            # 256 tokens, admits a 128-token (2,8,8) block
    seqlen = t * h * w
    perm = space_time_reorder_index(t, h, w)
    inv = invert_permutation(perm)
    x = torch.randn(1, seqlen, 2, 16)
    permuted = apply_token_permutation(x, perm)
    restored = apply_token_permutation(permuted, inv)
    assert torch.equal(restored, x)
    # it is a genuine permutation (every index used once)
    assert sorted(perm.tolist()) == list(range(seqlen))


def test_sdpa_backend_selection_guards_math_at_full_seqlen():
    # below the limit, the math fallback is permitted
    small = select_sdpa_backends(512, force_math_backend=False, allow_math_fallback=True)
    assert SDPBackend.MATH in small
    # at/above the limit, the math fallback is NOT added (cannot OOM on dense S x S)
    big = select_sdpa_backends(MATH_BACKEND_SEQLEN_LIMIT, force_math_backend=False,
                               allow_math_fallback=True)
    assert SDPBackend.MATH not in big
    assert SDPBackend.FLASH_ATTENTION in big
    # forcing math at full seqlen is rejected up front
    with pytest.raises(FullMatrixAllocationError):
        select_sdpa_backends(MATH_BACKEND_SEQLEN_LIMIT, force_math_backend=True,
                             allow_math_fallback=False)
    # forcing math below the limit is allowed
    assert select_sdpa_backends(256, force_math_backend=True, allow_math_fallback=False) == [SDPBackend.MATH]


def test_ranker_native_can_win_selection():
    select_layout = _load_select_layout()
    # native (grid None) has the highest diagonal mass -> recommend no reorder
    res_native_best = [
        {"grid": None, "block": None, "native_axis_order": None, "diagonal_mass": 0.30},
        {"grid": [8, 8, 16], "block": [2, 8, 8], "native_axis_order": ["t", "h", "w"], "diagonal_mass": 0.20},
    ]
    best, best_reorder = select_layout(res_native_best)
    assert best["grid"] is None                       # native wins -> no reorder
    assert best_reorder["grid"] == [8, 8, 16]         # still reports best candidate
    # a reorder beats native -> select that reorder
    res_reorder_best = [
        {"grid": None, "block": None, "native_axis_order": None, "diagonal_mass": 0.20},
        {"grid": [8, 8, 16], "block": [2, 8, 8], "native_axis_order": ["t", "h", "w"], "diagonal_mass": 0.35},
    ]
    best, _ = select_layout(res_reorder_best)
    assert best["grid"] == [8, 8, 16]


def test_full_matrix_guard_rejects_square():
    seqlen = 4096
    # a benign tiled shape passes
    guard_no_full_matrix((8, 128, seqlen), seqlen)
    # a full square plane is rejected
    with pytest.raises(FullMatrixAllocationError):
        guard_no_full_matrix((8, seqlen, seqlen), seqlen)
