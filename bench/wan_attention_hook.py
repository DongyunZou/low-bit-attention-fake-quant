"""Monkey-patch Wan2.1's ``wan.modules.attention.attention`` to dispatch
self-attention through our :func:`fake_quant_attention` while keeping
cross-attention on plain torch SDPA.

Usage:
    from bench.wan_attention_hook import install_hook, set_quant_cfg, NO_QUANT
    install_hook()                  # patch wan.modules.attention.attention
    set_quant_cfg(NO_QUANT)         # SDPA baseline (no fake quant)
    # ... run wan inference ...
    set_quant_cfg(my_cfg)           # subsequent runs use my_cfg

Heuristic for self vs cross attention:
    - Self-attention: q.shape[1] == k.shape[1] AND q.shape[1] > 512 (the
      Wan text encoder context length is 512). Cross-attention has the K/V
      sequence as the text tokens, much shorter than Q (video tokens), and
      our fake-quant kernel isn't tuned for it.
"""

from __future__ import annotations

import math
import threading
from typing import Optional

import torch
import torch.nn.functional as F

from low_bit_fake_quant import QuantConfig, fake_quant_attention


# ----- Module-level mutable state (thread-safe enough for single-GPU eval) ----

_state_lock = threading.Lock()
_current_cfg: Optional[QuantConfig] = None
_call_log: list[dict] = []  # diagnostic record of every dispatched call


def set_quant_cfg(cfg: Optional[QuantConfig]) -> None:
    """Set the active QuantConfig. ``None`` means run plain SDPA (baseline)."""
    global _current_cfg
    with _state_lock:
        _current_cfg = cfg
        _call_log.clear()


def get_call_log() -> list[dict]:
    with _state_lock:
        return list(_call_log)


def _log_call(record: dict) -> None:
    with _state_lock:
        _call_log.append(record)


# Sentinel for "no quant, just SDPA".
NO_QUANT: Optional[QuantConfig] = None


def _padded_or_packed(x: torch.Tensor, lens: Optional[torch.Tensor], total_len: int) -> torch.Tensor:
    """Wan flash_attention takes k_lens/q_lens for varlen packing. For our
    eval (single-prompt batch=1) we just trust the provided lengths."""
    if lens is None:
        return x
    # For B=1, lens is just [S]; nothing to repack.
    return x


# ----- The patched attention function ----------------------------------------


def _fake_or_sdpa_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_lens: Optional[torch.Tensor] = None,
    k_lens: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    q_scale: Optional[float] = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    deterministic: bool = False,
    dtype: torch.dtype = torch.bfloat16,
    fa_version=None,
    version=None,
) -> torch.Tensor:
    """Replacement for wan.modules.attention.attention / flash_attention.

    Q/K/V are ``(B, L, N, C)`` per Wan's convention (== our NHD layout).

    Self-attention path (``q.shape[1] == k.shape[1]`` AND it's the long video
    sequence) → dispatch to our ``fake_quant_attention`` if a cfg is set;
    otherwise plain SDPA.

    Cross-attention path (short K/V from the text encoder) → always plain SDPA.
    """
    B, Lq, Nq, C = q.shape
    Lk = k.shape[1]
    cfg = _current_cfg
    is_self_attn = (Lq == Lk and Lq > 1024)  # heuristic: long sequence ⇒ video self-attn

    # Cast to the requested dtype if necessary (Wan does this internally).
    half = (torch.float16, torch.bfloat16)
    if q.dtype not in half:
        q = q.to(dtype)
    if k.dtype not in half:
        k = k.to(dtype)
    if v.dtype not in half:
        v = v.to(dtype)
    if q_scale is not None:
        q = q * q_scale

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(C)

    if is_self_attn and cfg is not None:
        # Run our fake-quant attention. The fake-quant preprocess kernels
        # require S to divide block sizes for Q smooth (q_smooth_block_size),
        # V smooth (v_smooth_block_size), FP8 block (fp8_block_size), and the
        # Triton kernel's BLOCK_M/N. We pad S to LCM of all of these.
        block_sizes = [
            cfg.fp8_block_size, cfg.v_fp8_block_size,
            cfg.p_requant_block_m, cfg.p_requant_block_n,
        ]
        if cfg.smoothing == "full":
            block_sizes.append(cfg.q_smooth_block_size)
        if cfg.v_smooth_mode == "per_block":
            block_sizes.append(cfg.v_smooth_block_size)
        import functools
        lcm = functools.reduce(lambda a, b: a * b // math.gcd(a, b), block_sizes, 1)
        pad = (lcm - Lq % lcm) % lcm

        if pad == 0:
            out = fake_quant_attention(q, k, v, cfg, sm_scale=softmax_scale)
        else:
            # Pad Q with zeros (those rows are stripped from output anyway).
            # Pad K/V with REPLICATED last row so attention weight on padded
            # keys flows to the same V as the last real token — a small
            # bounded perturbation (pad ratio = pad/Lq, ~0.6% for Wan21).
            q_pad = q.new_zeros(B, pad, Nq, C)
            k_last = k[:, -1:, :, :].expand(B, pad, k.shape[2], k.shape[3])
            v_last = v[:, -1:, :, :].expand(B, pad, v.shape[2], v.shape[3])
            q_p = torch.cat([q, q_pad], dim=1)
            k_p = torch.cat([k, k_last], dim=1)
            v_p = torch.cat([v, v_last], dim=1)
            out = fake_quant_attention(q_p, k_p, v_p, cfg, sm_scale=softmax_scale)
            out = out[:, :Lq]
        _log_call({"kind": "self-fq", "Lq": Lq, "Lk": Lk, "D": C, "H": Nq, "pad": pad})
        return out.to(q.dtype)

    # Fall back to plain SDPA — used for cross-attention always, and for
    # self-attention when cfg is None (the reference run).
    qb = q.transpose(1, 2)  # (B, N, L, C)
    kb = k.transpose(1, 2)
    vb = v.transpose(1, 2)
    out = F.scaled_dot_product_attention(qb, kb, vb, dropout_p=dropout_p, is_causal=causal, scale=softmax_scale)
    out = out.transpose(1, 2).contiguous()
    _log_call({"kind": "self-sdpa" if is_self_attn else "cross-sdpa", "Lq": Lq, "Lk": Lk, "D": C, "H": Nq})
    return out.to(q.dtype)


# ----- Install ---------------------------------------------------------------


_installed = False


def install_hook() -> None:
    """Patch wan.modules.attention so all self-attention calls go through us."""
    global _installed
    if _installed:
        return
    import wan.modules.attention as wma
    import wan.modules.model as wmm
    # Replace both names — model.py imports flash_attention by name.
    wma.attention = _fake_or_sdpa_attention
    wma.flash_attention = _fake_or_sdpa_attention
    wmm.flash_attention = _fake_or_sdpa_attention
    _installed = True
    print("[wan_attention_hook] Patched wan.modules.attention.{attention,flash_attention}")


__all__ = ["install_hook", "set_quant_cfg", "get_call_log", "NO_QUANT"]
