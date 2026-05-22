"""Targeted sweep over Q kmeans k × V kmeans k.

Fixed knobs (best stack):
  qk_quant            = mxfp8
  v_quant             = fp8_channel
  smoothing           = full
  q_smooth_block_size = 256
  v_smooth_block_size = 256          (user-requested for this sweep)
  v_smooth_mode       = per_block
  p_requant           = on (Triton)

Swept:
  q_kmeans_k ∈ {None, 16, 32, 64, 128}
  v_kmeans_k ∈ {None, 16, 32, 64, 128}

5 × 5 = 25 configs × 6 wan21 timesteps = 150 evaluations. Output to
``bench/results_kmeans_k_sweep.json``.
"""

from __future__ import annotations

import dataclasses
import gc
import json
import math
import time
from pathlib import Path

import torch

from low_bit_fake_quant import (
    PreprocessCache,
    QuantConfig,
    fake_quant_attention,
    prepare_for_attention,
    reference_attention,
)


def build_matrix() -> dict[str, QuantConfig]:
    matrix: dict[str, QuantConfig] = {}
    qks = [None, 16, 32, 64, 128]
    vks = [None, 16, 32, 64, 128]
    for qk in qks:
        for vk in vks:
            name = f"qk_{qk if qk is not None else 'off'}__vk_{vk if vk is not None else 'off'}"
            matrix[name] = QuantConfig(
                qk_quant="mxfp8",
                v_quant="fp8_channel",
                smoothing="full",
                q_smooth_block_size=256,
                q_kmeans_k=qk,
                fp8_block_size=128,
                mxfp8_block_size=32,
                v_smooth_mode="per_block",
                v_smooth_block_size=256,   # user-requested
                v_kmeans_k=vk,
                p_requant=True,
                p_requant_block_m=64,
                p_requant_block_n=64,
            )
    return matrix


def load_qkv(path, device):
    data = torch.load(path, map_location="cpu", weights_only=False)
    return tuple(data[k].to(device=device, dtype=torch.bfloat16, non_blocking=True)
                 for k in ("query", "key", "value"))


def metrics(o, ref):
    a = o.float().flatten()
    b = ref.float().flatten()
    diff = a - b
    mse = float(diff.pow(2).mean().item())
    rmse = math.sqrt(mse)
    eps = 1e-12
    cos = float(torch.dot(a, b).item() / max(eps, a.norm().item()) / max(eps, b.norm().item()))
    return {"mse": mse, "rmse": rmse, "cosine": cos}


def main():
    data_root = Path("/home/dongyun/dataset/v-dit/wan21_p1")
    out_path = Path("bench/results_kmeans_k_sweep.json")
    device = torch.device("cuda")

    workloads = sorted((data_root / "layer_0").glob("timestep_*.pt"))
    print(f"Found {len(workloads)} workloads.")

    matrix = build_matrix()
    print(f"Configs: {len(matrix)}")

    results: dict = {"workloads": [], "configs": {n: dataclasses.asdict(c) for n, c in matrix.items()}}

    for w_idx, path in enumerate(workloads):
        t0 = time.time()
        q, k, v = load_qkv(path, device)
        load_s = time.time() - t0
        torch.cuda.synchronize()
        t_ref = time.time()
        ref = reference_attention(q, k, v)
        torch.cuda.synchronize()
        ref_s = time.time() - t_ref

        # Each (q_kmeans_k, v_kmeans_k) pair changes both q_work and v/k_work,
        # so the cache would have 25 distinct entries (~100GB at wan21 scale).
        # That OOMs an 80GB H100. Instead recompute preprocess per config —
        # flash-kmeans is fast (<1s).
        wl_record = {"path": str(path), "load_s": load_s, "ref_s": ref_s, "configs": {}}
        print(f"[{w_idx+1}/{len(workloads)}] {path.name}: load={load_s:.1f}s ref={ref_s:.2f}s")

        for name, cfg in matrix.items():
            torch.cuda.synchronize()
            t = time.time()
            o = fake_quant_attention(q, k, v, cfg)
            torch.cuda.synchronize()
            secs = time.time() - t
            m = metrics(o, ref)
            wl_record["configs"][name] = {**m, "seconds": secs}
            print(f"  {name:<32} MSE={m['mse']:.3e} RMSE={m['rmse']:.3e} cos={m['cosine']:.6f} ({secs:.2f}s)")
            del o
            torch.cuda.empty_cache()

        results["workloads"].append(wl_record)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            json.dump(results, f, indent=2, default=str)
        del q, k, v, ref
        gc.collect()
        torch.cuda.empty_cache()

    # Aggregate
    agg: dict[str, dict[str, float]] = {}
    for w in results["workloads"]:
        for name, m in w["configs"].items():
            slot = agg.setdefault(name, {"mse": [], "rmse": [], "cosine": []})
            for k_ in ("mse", "rmse", "cosine"):
                slot[k_].append(m[k_])
    summary = {name: {f"{k_}_mean": sum(vs) / len(vs) for k_, vs in slots.items()}
               for name, slots in agg.items()}
    results["summary"] = summary
    with out_path.open("w") as f:
        json.dump(results, f, indent=2, default=str)

    # Print pretty matrix
    qks = [None, 16, 32, 64, 128]
    vks = [None, 16, 32, 64, 128]
    print()
    print("Mean Cosine over 6 wan21 timesteps (rows=Q kmeans k, cols=V kmeans k):")
    print(f'{"":>10}', end="")
    for vk in vks:
        print(f'{("vk=" + (str(vk) if vk else "off")):>13}', end="")
    print()
    for qk in qks:
        print(f'{("qk=" + (str(qk) if qk else "off")):>10}', end="")
        for vk in vks:
            name = f"qk_{qk if qk is not None else 'off'}__vk_{vk if vk is not None else 'off'}"
            print(f'{summary[name]["cosine_mean"]:>13.6f}', end="")
        print()

    print()
    print("Mean MSE over 6 wan21 timesteps:")
    print(f'{"":>10}', end="")
    for vk in vks:
        print(f'{("vk=" + (str(vk) if vk else "off")):>13}', end="")
    print()
    for qk in qks:
        print(f'{("qk=" + (str(qk) if qk else "off")):>10}', end="")
        for vk in vks:
            name = f"qk_{qk if qk is not None else 'off'}__vk_{vk if vk is not None else 'off'}"
            print(f'{summary[name]["mse_mean"]:>13.3e}', end="")
        print()


if __name__ == "__main__":
    main()
