#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Image-level parity layer for the MixFusion A/B endorsement run.

Decodes dumped final latents (packed 2-D layout, [1, P, 4*z_dim]) through the
same Qwen-Image VAE the engine serves, then compares the decoded images
pairwise (PSNR, SSIM, max pixel diff) and archives A/B/diff PNGs.

The decode mirrors QwenImagePipeline._decode_latents exactly: unpack to
[1, z_dim, 1, H/8, W/8], denormalize with the VAE's latents_mean/std, decode,
drop the temporal dim. Both sides decode through the same VAE module file and
weights, so image-level deltas reflect only the latent difference.

One invocation can compare several dump-dir pairs; the VAE is loaded once.

Usage:
  python mixfusion_ab_image_parity.py --model Qwen/Qwen-Image --device cuda \
      --pair A-W3:<parity>/A-W3:<parity>/B-W3-on:1024x1024,1024x1024,1024x768,... \
      --pair A-W3-floor:<parity>/A-W3-ctrl1:<parity>/A-W3-ctrl2:... \
      --out-dir <parity>/images

Each --pair is name:dirA:dirB:sizes (sizes = per-request HxW, comma-separated,
matching dump file order).
"""

from __future__ import annotations

import argparse
import importlib.util
import math
from pathlib import Path

import torch
import torch.nn.functional as F

# VAE module of this checkout; importing it from file avoids the heavy
# vllm_omni package __init__ (and any VLLM_OMNI_ROOT path isolation concerns).
DEFAULT_VAE_MODULE = (
    Path(__file__).resolve().parents[2]
    / "vllm_omni"
    / "diffusion"
    / "models"
    / "qwen_image"
    / "autoencoder_kl_qwenimage.py"
)


def load_vae(model_dir: str, device: str, dtype: torch.dtype, vae_module: Path):
    """Import AutoencoderKLQwenImage from the repo file (avoids the heavy
    vllm_omni package __init__) and load the served VAE weights."""
    spec = importlib.util.spec_from_file_location("bench_vae_qwenimage", vae_module)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    vae = mod.AutoencoderKLQwenImage.from_pretrained(model_dir, subfolder="vae", torch_dtype=dtype)
    return vae.to(device=device).eval()


def unpack_latents(latents: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Copy of QwenImagePipeline._unpack_latents (vae_scale_factor=8, 2x2 packed)."""
    if latents.dim() == 2:
        latents = latents.unsqueeze(0)
    batch_size, num_patches, channels = latents.shape
    h = 2 * (height // 16)
    w = 2 * (width // 16)
    assert num_patches == (h // 2) * (w // 2), (
        f"num_patches {num_patches} != {(h // 2) * (w // 2)} for {height}x{width}"
    )
    x = latents.view(batch_size, h // 2, w // 2, channels // 4, 2, 2)
    x = x.permute(0, 3, 1, 4, 2, 5)
    return x.reshape(batch_size, channels // 4, 1, h, w)


def decode_latent(vae, latents: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Mirror QwenImagePipeline._decode_latents (pil branch, before postprocess)."""
    x = unpack_latents(latents, height, width).to(device=vae.device, dtype=vae.dtype)
    cfg = vae.config
    z_dim = cfg.z_dim
    mean = torch.tensor(cfg.latents_mean, device=x.device, dtype=x.dtype).view(1, z_dim, 1, 1, 1)
    std = torch.tensor(cfg.latents_std, device=x.device, dtype=x.dtype).view(1, z_dim, 1, 1, 1)
    x = x / std + mean
    with torch.no_grad():
        image = vae.decode(x, return_dict=False)[0][:, :, 0]
    return image  # [B, 3, H, W], roughly [-1, 1]


def to_unit(image: torch.Tensor) -> torch.Tensor:
    return (image.float() / 2 + 0.5).clamp(0, 1)


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = F.mse_loss(a, b).item()
    return 10 * math.log10(1.0 / mse) if mse > 0 else float("inf")


def _gaussian_window(size: int = 11, sigma: float = 1.5, device: str = "cpu") -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32, device=device) - (size - 1) / 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    window = torch.outer(g, g)
    return window / window.sum()


def ssim(a: torch.Tensor, b: torch.Tensor, data_range: float = 1.0) -> float:
    """Standard SSIM (11x11 gaussian window) averaged over the image."""
    window = _gaussian_window(device=a.device).expand(1, 1, 11, 11).to(a.dtype)
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    pad = 11 // 2

    def mu(x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, window, padding=pad, groups=1)

    channels = a.shape[0]
    a1 = a.reshape(channels, 1, *a.shape[-2:])
    b1 = b.reshape(channels, 1, *b.shape[-2:])
    mu_a, mu_b = mu(a1), mu(b1)
    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    sigma_a2 = mu(a1 * a1) - mu_a2
    sigma_b2 = mu(b1 * b1) - mu_b2
    sigma_ab = mu(a1 * b1) - mu_ab
    ssim_map = ((2 * mu_ab + c1) * (2 * sigma_ab + c2)) / ((mu_a2 + mu_b2 + c1) * (sigma_a2 + sigma_b2 + c2))
    return ssim_map.mean().item()


def save_png(t: torch.Tensor, path: Path) -> None:
    from PIL import Image

    arr = (t.permute(1, 2, 0).cpu().numpy() * 255).round().astype("uint8")
    Image.fromarray(arr).save(path)


def compare_pair(name: str, dir_a: Path, dir_b: Path, sizes: list[str], vae, device: str, out_dir: Path | None):
    files = sorted(p.name for p in dir_a.glob("mixfusion_batch-*.pt"))
    files_b = sorted(p.name for p in dir_b.glob("mixfusion_batch-*.pt"))
    if files != files_b:
        raise SystemExit(f"{name}: dump file sets differ:\n  {dir_a}: {files}\n  {dir_b}: {files_b}")
    if len(files) != len(sizes):
        raise SystemExit(f"{name}: {len(files)} dumps but {len(sizes)} sizes given")
    worst = {"psnr": float("inf"), "ssim": 1.0, "max_abs": 0.0}
    for idx, fname in enumerate(files):
        h, w = (int(v) for v in sizes[idx].split("x"))
        la = torch.load(dir_a / fname, map_location="cpu", weights_only=True).to(device)
        lb = torch.load(dir_b / fname, map_location="cpu", weights_only=True).to(device)
        ia = to_unit(decode_latent(vae, la, h, w))[0]
        ib = to_unit(decode_latent(vae, lb, h, w))[0]
        if ia.shape != ib.shape:
            print(f"  {fname}: SHAPE MISMATCH {tuple(ia.shape)} vs {tuple(ib.shape)}")
            continue
        p = psnr(ia, ib)
        s = ssim(ia, ib)
        mx = (ia - ib).abs().max().item()
        worst["psnr"] = min(worst["psnr"], p)
        worst["ssim"] = min(worst["ssim"], s)
        worst["max_abs"] = max(worst["max_abs"], mx)
        print(f"  {fname}: {h}x{w} img_psnr={p:.2f}dB img_ssim={s:.4f} max_pixel_diff={mx:.4f}")
        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            stem = fname[: -len(".pt")]
            save_png(ia, out_dir / f"{name}-{stem}-A.png")
            save_png(ib, out_dir / f"{name}-{stem}-B.png")
            diff = (ia - ib).abs()
            save_png((diff * 10).clamp(0, 1), out_dir / f"{name}-{stem}-diff(x10).png")
    print(
        f"{name}: worst img_psnr={worst['psnr']:.2f}dB img_ssim={worst['ssim']:.4f} "
        f"max_pixel_diff={worst['max_abs']:.4f}"
    )
    return worst


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen-Image")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    ap.add_argument(
        "--vae-module",
        type=Path,
        default=DEFAULT_VAE_MODULE,
        help="path of autoencoder_kl_qwenimage.py in a vllm-omni checkout",
    )
    ap.add_argument(
        "--pair", action="append", required=True, help="name:dirA:dirB:sizes (sizes = comma-separated HxW per request)"
    )
    ap.add_argument("--out-dir", default=None, help="archive PNGs here (omit to skip)")
    args = ap.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    vae = load_vae(args.model, args.device, dtype, args.vae_module)
    out_dir = Path(args.out_dir) if args.out_dir else None
    for spec in args.pair:
        name, da, db, sizes = spec.split(":", 3)
        print(f"== image-level: {name} ==")
        compare_pair(name, Path(da), Path(db), sizes.split(","), vae, args.device, out_dir)


if __name__ == "__main__":
    main()
