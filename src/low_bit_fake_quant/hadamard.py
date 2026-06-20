from __future__ import annotations

import math
from functools import lru_cache
from typing import Callable

import torch

try:  # pragma: no cover - exercised only when the optional CUDA op is installed
    from fast_hadamard_transform import hadamard_transform as _fast_hadamard_transform
except Exception:  # noqa: BLE001
    _fast_hadamard_transform = None


def fast_hadamard_available() -> bool:
    """Whether Dao-AILab fast-hadamard-transform is importable."""

    return _fast_hadamard_transform is not None


def _next_power_of_two(x: int) -> int:
    return 1 << (x - 1).bit_length()


def _torch_hadamard_transform(x: torch.Tensor, *, scale: float) -> torch.Tensor:
    """Torch FWHT fallback matching fast_hadamard_transform's last-dim API."""

    dim = x.shape[-1]
    padded_dim = _next_power_of_two(dim)
    if padded_dim != dim:
        x_work = torch.nn.functional.pad(x, (0, padded_dim - dim))
    else:
        x_work = x

    out_shape = x_work.shape
    y = x_work.reshape(-1, padded_dim).float().clone()
    stride = 1
    while stride < padded_dim:
        y = y.reshape(-1, padded_dim // (2 * stride), 2 * stride)
        left = y[..., :stride].clone()
        right = y[..., stride : 2 * stride].clone()
        y[..., :stride] = left + right
        y[..., stride : 2 * stride] = left - right
        y = y.reshape(-1, padded_dim)
        stride *= 2
    y = (y.reshape(out_shape)[..., :dim] * float(scale)).to(x.dtype)
    return y.reshape_as(x)


def hadamard_transform_last_dim(x: torch.Tensor, *, scale: float) -> torch.Tensor:
    """Apply an unstructured Hadamard transform over the final dimension."""

    if _fast_hadamard_transform is not None:
        return _fast_hadamard_transform(x, scale=scale)
    return _torch_hadamard_transform(x, scale=scale)


@lru_cache(maxsize=128)
def _cached_cpu_signs(h: int, d: int, seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    signs = torch.randint(0, 2, (h, d), generator=gen, dtype=torch.int8)
    return signs.mul_(2).sub_(1)


def _qk_signs(
    *,
    h: int,
    d: int,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    signs = _cached_cpu_signs(h, d, seed).to(device=device, dtype=dtype)
    return signs.view(1, 1, h, d)


def apply_qk_hadamard(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    random_sign: bool = True,
    seed: int = 0,
    transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate Q/K by the same orthonormal Hadamard basis along head dim.

    For every head, this computes ``x' = (x * sign) @ H / sqrt(D)`` for Q and K
    using the same signs. Dot products are unchanged before quantization:
    ``(Q R) @ (K R).T == Q @ K.T``. The benefit comes from applying FP8/MXFP8
    quantization after the rotation, where outliers are spread across D.
    """

    if q.shape != k.shape:
        raise ValueError(f"q/k shapes must match; got {tuple(q.shape)} and {tuple(k.shape)}")
    if q.ndim != 4:
        raise ValueError(f"q/k must be NHD tensors; got {tuple(q.shape)}")
    if not (q.is_cuda and k.is_cuda):
        raise ValueError("q/k Hadamard rotation expects CUDA tensors")

    _, _, h, d = q.shape
    if d <= 0:
        raise ValueError("head dimension must be positive")

    q_work = q.contiguous()
    k_work = k.contiguous()
    if random_sign:
        signs = _qk_signs(h=h, d=d, seed=seed, device=q.device, dtype=q.dtype)
        q_work = q_work * signs
        k_work = k_work * signs.to(dtype=k.dtype)

    scale = 1.0 / math.sqrt(float(_next_power_of_two(d)))
    if transform is not None:
        return transform(q_work), transform(k_work)
    return (
        hadamard_transform_last_dim(q_work, scale=scale).contiguous(),
        hadamard_transform_last_dim(k_work, scale=scale).contiguous(),
    )


__all__ = [
    "apply_qk_hadamard",
    "fast_hadamard_available",
    "hadamard_transform_last_dim",
]
