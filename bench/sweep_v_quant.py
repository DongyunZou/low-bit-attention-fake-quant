"""V quant variants × V smoothing sweep on wan21.

Fixed (best-stack) knobs:
  qk_quant            = mxfp8
  smoothing           = full
  q_smooth_block_size = 256
  q_kmeans_k          = 32
  v_smooth_block_size = 256
  v_kmeans_k          = 64
  p_requant           = on (Triton)
  P quant             = auto (matches V quant)

Swept:
  v_quant         ∈ {fp8_channel, fp8_block, mxfp8}
  v_smooth_mode   ∈ {off, per_block}
  block sizes     = 64 for v_fp8 / v_mxfp8 (so MX P aligns with block_n=64)

3 × 2 = 6 configs × 6 wan21 timesteps = 36 evaluations.
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
    QuantConfig,
    fake_quant_attention,
    reference_attention,
)


def build_matrix() -> dict[str, QuantConfig]:
    matrix: dict[str, QuantConfig] = {}
    for vq in ("fp8_channel", "fp8_block", "mxfp8"):
        for vsm in ("off", "per_block"):
            name = f"V_{vq}__vsmooth_{vsm}"
            matrix[name] = QuantConfig(
                qk_quant="mxfp8",
                v_quant=vq,
                smoothing="full",
                q_smooth_block_size=256,
                q_kmeans_k=32,
                fp8_block_size=128,
                mxfp8_block_size=32,
                v_fp8_block_size=64,
                v_mxfp8_block_size=64,
                v_smooth_mode=vsm,
                v_smooth_block_size=256,
                v_kmeans_k=64,
                p_quant="auto",
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
    a = o.float().flatten(); b = ref.float().flatten()
    diff = a - b
    mse = float(diff.pow(2).mean().item())
    rmse = math.sqrt(mse)
    cos = float(torch.dot(a, b).item() / max(1e-12, a.norm().item()) / max(1e-12, b.norm().item()))
    return {"mse": mse, "rmse": rmse, "cosine": cos}


def main():
    data_root = Path("/home/dongyun/dataset/v-dit/wan21_p1")
    out_path = Path("bench/results_v_quant_sweep.json")
    device = torch.device("cuda")
    workloads = sorted((data_root / "layer_0").glob("timestep_*.pt"))
    matrix = build_matrix()
    print(f"{len(matrix)} configs × {len(workloads)} workloads = {len(matrix)*len(workloads)} evaluations")

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
            print(f"  {name:<40} MSE={m['mse']:.3e} RMSE={m['rmse']:.3e} cos={m['cosine']:.6f}")
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
    agg: dict[str, dict[str, list[float]]] = {}
    for w in results["workloads"]:
        for name, m in w["configs"].items():
            slot = agg.setdefault(name, {"mse": [], "rmse": [], "cosine": []})
            for k_ in ("mse", "rmse", "cosine"):
                slot[k_].append(m[k_])
    summary = {name: {f"{k_}_mean": sum(vs)/len(vs) for k_, vs in slots.items()}
               for name, slots in agg.items()}
    results["summary"] = summary
    with out_path.open("w") as f:
        json.dump(results, f, indent=2, default=str)

    print()
    print("Summary (mean over 6 wan21 timesteps):")
    print(f'{"config":<40} {"MSE":>11} {"RMSE":>11} {"cos":>10}')
    print('-' * 80)
    for name, agg_m in sorted(summary.items(), key=lambda r: -r[1]["cosine_mean"]):
        print(f'{name:<40} {agg_m["mse_mean"]:>11.3e} {agg_m["rmse_mean"]:>11.3e} {agg_m["cosine_mean"]:>10.6f}')

    # Effect of V smooth per V quant kind
    print()
    print("V smoothing effect per V quant:")
    print(f'{"V quant":<15} {"MSE off":>11} {"MSE on":>11} {"ΔMSE_rel":>10} {"Δcos":>10}')
    print('-' * 70)
    for vq in ("fp8_channel", "fp8_block", "mxfp8"):
        off = summary[f"V_{vq}__vsmooth_off"]
        on = summary[f"V_{vq}__vsmooth_per_block"]
        dm = (on["mse_mean"] - off["mse_mean"]) / off["mse_mean"]
        dc = on["cosine_mean"] - off["cosine_mean"]
        print(f'{vq:<15} {off["mse_mean"]:>11.3e} {on["mse_mean"]:>11.3e} {dm:>+10.2%} {dc:>+10.6f}')


if __name__ == "__main__":
    main()
