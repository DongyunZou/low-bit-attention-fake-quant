"""Token k-means reorder backed by flash-kmeans (Triton).

Two thin wrappers — :func:`q_kmeans_reorder` and :func:`v_kmeans_reorder` —
share a single implementation :func:`kmeans_reorder_tokens` that clusters
tokens along the S axis per ``(B, H)`` slice with Euclidean distance and
returns a permutation that bunches similar tokens together.

For Q the reorder permutes the *query* axis only; for V the reorder
permutes the *key/value* axis, which means K must be co-permuted by the
same permutation so attention output is unchanged (this co-permutation is
done at the call site in :mod:`attention.py`).

flash-kmeans (Apache-2.0, https://github.com/svg-project/flash-kmeans)
provides ``batch_kmeans_Euclid`` which fuses the assignment and centroid
update passes on GPU.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from flash_kmeans import batch_kmeans_Euclid


@dataclass
class KMeansReorderResult:
    tensor_reordered: torch.Tensor
    order: torch.Tensor
    inverse_order: torch.Tensor
    labels: torch.Tensor

    # Back-compat aliases — older code referred to q_reordered, new code may
    # use v_reordered. Both point at the same tensor.
    @property
    def q_reordered(self) -> torch.Tensor:  # pragma: no cover - trivial
        return self.tensor_reordered

    @property
    def v_reordered(self) -> torch.Tensor:  # pragma: no cover - trivial
        return self.tensor_reordered


def kmeans_reorder_tokens(
    t: torch.Tensor,
    *,
    n_clusters: int,
    max_iters: int = 10,
    seed: int = 0,
    chunk_tokens: int = 4096,  # kept for backward-compat; unused
) -> KMeansReorderResult:
    """Cluster tokens of ``t`` per (B, H) and return a reorder result.

    Parameters
    ----------
    t : (B, S, H, D) tensor on CUDA — the tokens to cluster along S.
    n_clusters : k
    max_iters : maximum Lloyd iterations
    seed : torch RNG seed used for centroid init.

    Returns
    -------
    KMeansReorderResult with
      - tensor_reordered : (B, S, H, D) — rows permuted by cluster
      - order            : (B*H, S) int64 — order[i, j] = original index of
                           the j-th row in batch i after reorder
      - inverse_order    : (B*H, S) int64 — gather indices to undo reorder
      - labels           : (B*H, S) int64 — final cluster assignment
    """
    del chunk_tokens  # flash-kmeans is single-pass
    if t.ndim != 4 or not t.is_cuda:
        raise ValueError(f"t must be 4-D NHD on CUDA; got {tuple(t.shape)} on {t.device}")
    b, s, h, d = t.shape
    if n_clusters <= 0 or n_clusters > s:
        raise ValueError(f"n_clusters must be in [1, S]; got {n_clusters} for S={s}")

    x = t.permute(0, 2, 1, 3).reshape(b * h, s, d).contiguous().float()
    torch.manual_seed(seed)
    labels_i32, _centroids, _n_iters = batch_kmeans_Euclid(
        x, n_clusters, max_iters=max_iters, tol=0.0, verbose=False,
    )
    labels = labels_i32.to(torch.int64)  # (B*H, S)

    order = torch.argsort(labels, dim=1, stable=True)
    pos = torch.arange(s, device=t.device).expand_as(order)
    inverse_order = torch.empty_like(order)
    inverse_order.scatter_(1, order, pos)
    x_reordered = torch.gather(x, 1, order.unsqueeze(-1).expand(-1, -1, d))
    t_reordered = x_reordered.reshape(b, h, s, d).permute(0, 2, 1, 3).to(t.dtype).contiguous()
    return KMeansReorderResult(
        tensor_reordered=t_reordered,
        order=order,
        inverse_order=inverse_order,
        labels=labels,
    )


def q_kmeans_reorder(
    q: torch.Tensor,
    *,
    n_clusters: int = 32,
    max_iters: int = 10,
    seed: int = 0,
    chunk_tokens: int = 4096,
) -> KMeansReorderResult:
    """Cluster Q tokens per ``(B, H)`` and reorder them. Output is in Q's S order;
    attention's output rows must be inverse-permuted to restore original Q ordering."""
    return kmeans_reorder_tokens(
        q, n_clusters=n_clusters, max_iters=max_iters, seed=seed, chunk_tokens=chunk_tokens,
    )


def v_kmeans_reorder(
    v: torch.Tensor,
    *,
    n_clusters: int = 32,
    max_iters: int = 10,
    seed: int = 0,
) -> KMeansReorderResult:
    """Cluster V tokens per ``(B, H)`` and reorder. Callers MUST co-permute K
    along S with the same ``order`` so attention output is unchanged."""
    return kmeans_reorder_tokens(
        v, n_clusters=n_clusters, max_iters=max_iters, seed=seed,
    )


def apply_kv_permutation(t: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
    """Apply a ``(B*H, S)`` permutation (e.g. from V kmeans) along S of a
    ``(B, S, H, D)`` tensor. Used to co-permute K with V's reorder."""
    if t.ndim != 4:
        raise ValueError(f"t must be (B,S,H,D); got {tuple(t.shape)}")
    b, s, h, d = t.shape
    t_bh = t.permute(0, 2, 1, 3).reshape(b * h, s, d).contiguous()
    t_re = torch.gather(t_bh, 1, order.unsqueeze(-1).expand(-1, -1, d))
    return t_re.reshape(b, h, s, d).permute(0, 2, 1, 3).contiguous().to(t.dtype)
