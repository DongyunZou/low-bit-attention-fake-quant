from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class KMeansReorderResult:
    q_reordered: torch.Tensor
    order: torch.Tensor
    inverse_order: torch.Tensor
    labels: torch.Tensor


def _assign_labels_chunked(x: torch.Tensor, centroids: torch.Tensor, chunk_tokens: int) -> torch.Tensor:
    n, s, _ = x.shape
    labels = torch.empty((n, s), dtype=torch.int64, device=x.device)
    c_norm = centroids.square().sum(dim=-1)
    for start in range(0, s, chunk_tokens):
        end = min(start + chunk_tokens, s)
        x_chunk = x[:, start:end, :]
        x_norm = x_chunk.square().sum(dim=-1, keepdim=True)
        dot = torch.einsum("nsd,nkd->nsk", x_chunk, centroids)
        dist = x_norm + c_norm[:, None, :] - 2.0 * dot
        labels[:, start:end] = dist.argmin(dim=-1)
    return labels


def q_kmeans_reorder(
    q: torch.Tensor,
    *,
    n_clusters: int = 32,
    max_iters: int = 10,
    seed: int = 0,
    chunk_tokens: int = 4096,
) -> KMeansReorderResult:
    """Cluster Q tokens per (B,H) and reorder by labels.

    This is a deterministic chunked torch implementation intended as the
    correctness helper for the first framework version. The plan document
    tracks the Triton split-kernel replacement: label assignment by tiled
    distance matmul plus centroid reductions.
    """
    if q.ndim != 4 or not q.is_cuda:
        raise ValueError(f"q must be 4-D NHD on CUDA; got {tuple(q.shape)} on {q.device}")
    b, s, h, d = q.shape
    if n_clusters <= 0 or n_clusters > s:
        raise ValueError(f"n_clusters must be in [1, S]; got {n_clusters} for S={s}")
    x = q.permute(0, 2, 1, 3).reshape(b * h, s, d).contiguous().float()
    n = x.shape[0]

    gen = torch.Generator(device=q.device)
    gen.manual_seed(seed)
    centroids = torch.empty((n, n_clusters, d), dtype=torch.float32, device=q.device)
    for row in range(n):
        idx = torch.randperm(s, device=q.device, generator=gen)[:n_clusters]
        centroids[row] = x[row, idx]

    labels = torch.empty((n, s), dtype=torch.int64, device=q.device)
    for _ in range(max_iters):
        labels = _assign_labels_chunked(x, centroids, chunk_tokens)
        new_centroids = torch.zeros_like(centroids)
        counts = torch.zeros((n, n_clusters), dtype=torch.float32, device=q.device)
        for start in range(0, s, chunk_tokens):
            end = min(start + chunk_tokens, s)
            x_chunk = x[:, start:end, :]
            lab_chunk = labels[:, start:end]
            for cluster in range(n_clusters):
                mask = lab_chunk == cluster
                counts[:, cluster] += mask.sum(dim=1)
                new_centroids[:, cluster, :] += (x_chunk * mask.unsqueeze(-1)).sum(dim=1)
        nonempty = counts > 0
        centroids = torch.where(
            nonempty.unsqueeze(-1),
            new_centroids / counts.clamp_min(1.0).unsqueeze(-1),
            centroids,
        )

    labels = _assign_labels_chunked(x, centroids, chunk_tokens)
    order = torch.argsort(labels, dim=1, stable=True)
    pos = torch.arange(s, device=q.device).expand_as(order)
    inverse_order = torch.empty_like(order)
    inverse_order.scatter_(1, order, pos)
    x_reordered = torch.gather(x, 1, order.unsqueeze(-1).expand(-1, -1, d))
    q_reordered = x_reordered.reshape(b, h, s, d).permute(0, 2, 1, 3).to(q.dtype).contiguous()
    return KMeansReorderResult(
        q_reordered=q_reordered,
        order=order,
        inverse_order=inverse_order,
        labels=labels,
    )
