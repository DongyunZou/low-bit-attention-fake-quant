"""Evaluate PSNR/SSIM/LPIPS between generated videos and references."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

try:
    from torchvision.io import read_video
except Exception:  # torchvision 0.27 no longer exposes read_video.
    read_video = None


VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


@dataclass
class VideoMetrics:
    frames: int
    psnr_sum: float
    ssim_sum: float
    lpips_sum: float


@torch.no_grad()
def _build_ssim_kernel(
    channels: int, device: torch.device, window_size: int = 11, sigma: float = 1.5
) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
    gauss = torch.exp(-(coords * coords) / (2 * sigma * sigma))
    gauss = gauss / gauss.sum()
    kernel_2d = gauss[:, None] * gauss[None, :]
    return kernel_2d[None, None, :, :].repeat(channels, 1, 1, 1)


@torch.no_grad()
def _ssim_batch(x: torch.Tensor, y: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    c1 = 0.01**2
    c2 = 0.03**2
    padding = kernel.shape[-1] // 2
    mu_x = F.conv2d(x, kernel, padding=padding, groups=x.shape[1])
    mu_y = F.conv2d(y, kernel, padding=padding, groups=x.shape[1])
    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)
    mu_xy = mu_x * mu_y
    sigma_x_sq = F.conv2d(x * x, kernel, padding=padding, groups=x.shape[1]) - mu_x_sq
    sigma_y_sq = F.conv2d(y * y, kernel, padding=padding, groups=x.shape[1]) - mu_y_sq
    sigma_xy = F.conv2d(x * y, kernel, padding=padding, groups=x.shape[1]) - mu_xy
    numerator = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
    return (numerator / (denominator + 1e-12)).mean(dim=(1, 2, 3))


@torch.no_grad()
def _read_video_frames(path: Path) -> torch.Tensor:
    if read_video is not None:
        try:
            frames, _, _ = read_video(str(path), pts_unit="sec")
            if frames.numel() == 0:
                raise RuntimeError(f"No frames in video: {path}")
            return frames
        except Exception:
            pass
    try:
        import imageio.v3 as iio

        arr = iio.imread(str(path), plugin="pyav")
        if arr.ndim != 4:
            raise RuntimeError(f"Unexpected decoded shape for {path}: {arr.shape}")
        if arr.shape[-1] > 3:
            arr = arr[..., :3]
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return torch.from_numpy(arr)
    except Exception as exc:
        raise RuntimeError(f"Failed to decode video: {path}\n{exc}") from exc


def _list_videos(root: Path, recursive: bool, match_mode: str, exts: set[str]) -> Dict[str, Path]:
    pattern = "**/*" if recursive else "*"
    videos = {}
    for p in root.glob(pattern):
        if not p.is_file() or p.suffix.lower() not in exts:
            continue
        key = p.stem if match_mode == "stem" else p.name
        if key in videos:
            raise RuntimeError(f"Duplicate key {key!r} in {root}")
        videos[key] = p
    return videos


@torch.no_grad()
def _compute_metrics_for_pair(
    pred_path: Path,
    ref_path: Path,
    device: torch.device,
    chunk_size: int,
    ssim_kernel: torch.Tensor,
    lpips_model,
) -> VideoMetrics:
    pred_frames = _read_video_frames(pred_path)
    ref_frames = _read_video_frames(ref_path)
    t = min(pred_frames.shape[0], ref_frames.shape[0])
    if t <= 0:
        raise RuntimeError(f"No overlapping frames for pair:\n- pred={pred_path}\n- ref={ref_path}")

    pred = pred_frames[:t, ..., :3].permute(0, 3, 1, 2).float() / 255.0
    ref = ref_frames[:t, ..., :3].permute(0, 3, 1, 2).float() / 255.0
    if pred.shape[-2:] != ref.shape[-2:]:
        pred = F.interpolate(pred, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    psnr_sum = 0.0
    ssim_sum = 0.0
    lpips_sum = 0.0
    for i in range(0, t, chunk_size):
        p = pred[i : i + chunk_size].to(device=device, dtype=torch.float32, non_blocking=True)
        r = ref[i : i + chunk_size].to(device=device, dtype=torch.float32, non_blocking=True)
        mse = (p - r).pow(2).mean(dim=(1, 2, 3))
        psnr = 10.0 * torch.log10(1.0 / (mse + 1e-10))
        ssim = _ssim_batch(p, r, ssim_kernel)
        psnr_sum += float(psnr.sum().item())
        ssim_sum += float(ssim.sum().item())
        if lpips_model is not None:
            lp = lpips_model(p * 2.0 - 1.0, r * 2.0 - 1.0).reshape(-1)
            lpips_sum += float(lp.sum().item())
    return VideoMetrics(frames=t, psnr_sum=psnr_sum, ssim_sum=ssim_sum, lpips_sum=lpips_sum)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PSNR/SSIM/LPIPS for video outputs")
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument("--ref-dir", type=Path, default=None)
    parser.add_argument("--ref-video", type=Path, default=None, help="Compare every pred video to one ref")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--match-mode", default="name", choices=["name", "stem"])
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--disable-lpips", action="store_true")
    parser.add_argument("--lpips-net", default="alex", choices=["alex", "vgg", "squeeze"])
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--extensions", default=",".join(sorted(VIDEO_EXTS)))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    pred_dir = args.pred_dir.resolve()
    if not pred_dir.is_dir():
        raise RuntimeError(f"Invalid pred-dir: {pred_dir}")
    if args.ref_dir is None and args.ref_video is None:
        raise RuntimeError("pass either --ref-dir or --ref-video")
    if args.ref_dir is not None and args.ref_video is not None:
        raise RuntimeError("pass only one of --ref-dir or --ref-video")

    exts = {x.strip().lower() for x in args.extensions.split(",") if x.strip()}
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but not available")

    pred_map = _list_videos(pred_dir, args.recursive, args.match_mode, exts)
    if args.ref_video is not None:
        ref_video = args.ref_video.resolve()
        keys = sorted(k for k, p in pred_map.items() if p.resolve() != ref_video)
        ref_map = {k: ref_video for k in keys}
    else:
        ref_dir = args.ref_dir.resolve()
        if not ref_dir.is_dir():
            raise RuntimeError(f"Invalid ref-dir: {ref_dir}")
        ref_map = _list_videos(ref_dir, args.recursive, args.match_mode, exts)
        keys = sorted(set(pred_map.keys()) & set(ref_map.keys()))
    if not keys:
        raise RuntimeError("No matched videos found")

    lpips_model = None
    if not args.disable_lpips:
        import lpips

        lpips_model = lpips.LPIPS(net=args.lpips_net).to(device)
        lpips_model.eval()
    ssim_kernel = _build_ssim_kernel(channels=3, device=device)

    total_frames = 0
    total_psnr = 0.0
    total_ssim = 0.0
    total_lpips = 0.0
    per_video = []
    for key in keys:
        m = _compute_metrics_for_pair(
            pred_path=pred_map[key],
            ref_path=ref_map[key],
            device=device,
            chunk_size=args.chunk_size,
            ssim_kernel=ssim_kernel,
            lpips_model=lpips_model,
        )
        total_frames += m.frames
        total_psnr += m.psnr_sum
        total_ssim += m.ssim_sum
        total_lpips += m.lpips_sum
        per_video.append(
            {
                "key": key,
                "pred_path": str(pred_map[key]),
                "ref_path": str(ref_map[key]),
                "frames": m.frames,
                "psnr": m.psnr_sum / m.frames,
                "ssim": m.ssim_sum / m.frames,
                "lpips": (m.lpips_sum / m.frames) if lpips_model is not None else None,
            }
        )

    summary = {
        "num_matched_videos": len(keys),
        "total_frames": total_frames,
        "frame_weighted_avg": {
            "psnr": total_psnr / total_frames,
            "ssim": total_ssim / total_frames,
            "lpips": (total_lpips / total_frames) if lpips_model is not None else None,
        },
        "video_weighted_avg": {
            "psnr": float(np.mean([x["psnr"] for x in per_video])),
            "ssim": float(np.mean([x["ssim"] for x in per_video])),
            "lpips": (
                float(np.mean([x["lpips"] for x in per_video]))
                if lpips_model is not None
                else None
            ),
        },
        "per_video": per_video,
    }
    print(json.dumps(summary["frame_weighted_avg"], indent=2))
    print(json.dumps(summary["video_weighted_avg"], indent=2))
    output_json = args.output_json or pred_dir / f"{pred_dir.name}_metrics.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Saved full report to: {output_json.resolve()}")


if __name__ == "__main__":
    main()
