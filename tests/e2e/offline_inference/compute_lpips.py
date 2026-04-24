#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compute LPIPS between pairs of BF16 (baseline) and FP8 (quantized) images.

Reads image pairs from a directory, computes LPIPS perceptual distance,
and prints a Markdown results table.

Usage:
    python compute_lpips.py --image-dir ./test_quant_outputs

Expects files named: <model>_bf16.png and <model>_fp8.png
e.g. zimage_bf16.png / zimage_fp8.png, qwen_bf16.png / qwen_fp8.png

Requirements:
    pip install lpips Pillow torchvision
"""

import argparse
import sys
from pathlib import Path


def compute_lpips(img_baseline, img_quantized, net="alex"):
    """Compute LPIPS between two PIL images."""
    import lpips
    import torch
    from torchvision import transforms

    loss_fn = lpips.LPIPS(net=net).eval()
    if torch.cuda.is_available():
        loss_fn = loss_fn.cuda()

    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    t_bl = transform(img_baseline.convert("RGB")).unsqueeze(0)
    t_qt = transform(img_quantized.convert("RGB")).unsqueeze(0)
    if torch.cuda.is_available():
        t_bl, t_qt = t_bl.cuda(), t_qt.cuda()
    with torch.no_grad():
        score = loss_fn(t_bl, t_qt).item()
    return score


def main():
    parser = argparse.ArgumentParser(description="Compute LPIPS for BF16 vs FP8 image pairs.")
    parser.add_argument(
        "--image-dir", type=str, required=True, help="Directory containing *_bf16.png and *_fp8.png pairs."
    )
    parser.add_argument("--threshold", type=float, default=0.1, help="LPIPS threshold for PASS/FAIL (default: 0.1).")
    parser.add_argument(
        "--net", type=str, default="alex", choices=["alex", "vgg", "squeeze"], help="LPIPS backbone (default: alex)."
    )
    args = parser.parse_args()

    from PIL import Image

    image_dir = Path(args.image_dir)
    if not image_dir.exists():
        print(f"ERROR: directory {image_dir} does not exist")
        sys.exit(1)

    # Find all *_bf16.png files and look for matching *_fp8.png
    bf16_files = sorted(image_dir.glob("*_bf16.png"))
    if not bf16_files:
        print(f"No *_bf16.png files found in {image_dir}")
        sys.exit(0)

    results = []
    all_pass = True

    for bf16_path in bf16_files:
        model_name = bf16_path.stem.replace("_bf16", "")
        fp8_path = image_dir / f"{model_name}_fp8.png"
        if not fp8_path.exists():
            print(f"  SKIP {model_name}: no matching {fp8_path.name}")
            continue

        img_bl = Image.open(bf16_path)
        img_fp8 = Image.open(fp8_path)
        score = compute_lpips(img_bl, img_fp8, net=args.net)

        status = "PASS" if score < args.threshold else "FAIL"
        if score >= args.threshold:
            all_pass = False
        results.append(
            (model_name, score, status, f"{img_bl.width}x{img_bl.height}", f"{img_fp8.width}x{img_fp8.height}")
        )

    # Print results table
    print("")
    print("=" * 70)
    print(f"  LPIPS Results (net={args.net}, threshold={args.threshold})")
    print("=" * 70)
    print(f"  {'Model':<20} {'LPIPS':>8} {'Status':>8}  {'BF16 size':>12} {'FP8 size':>12}")
    print(f"  {'-' * 20} {'-' * 8} {'-' * 8}  {'-' * 12} {'-' * 12}")
    for model_name, score, status, sz_bl, sz_fp8 in results:
        print(f"  {model_name:<20} {score:>8.4f} {status:>8}  {sz_bl:>12} {sz_fp8:>12}")
    print("")
    print("  LPIPS < 0.01 = imperceptible")
    print("  LPIPS < 0.05 = minor differences")
    print("  LPIPS < 0.10 = noticeable but acceptable")
    print("  LPIPS > 0.10 = clearly different")
    print("=" * 70)

    # Markdown table for PR
    md_path = image_dir / "lpips_results.md"
    with open(md_path, "w") as f:
        f.write("## LPIPS Quality Benchmark (BF16 vs FP8)\n\n")
        f.write(f"LPIPS backbone: `{args.net}` | threshold: `{args.threshold}`\n\n")
        f.write("| Model | LPIPS | Status | BF16 Size | FP8 Size |\n")
        f.write("|-------|-------|--------|-----------|----------|\n")
        for model_name, score, status, sz_bl, sz_fp8 in results:
            emoji = "✅" if status == "PASS" else "❌"
            f.write(f"| {model_name} | {score:.4f} | {emoji} {status} | {sz_bl} | {sz_fp8} |\n")
        f.write(f"\n> LPIPS < {args.threshold} = PASS\n")
    print(f"  Markdown saved to: {md_path}")

    if not all_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
