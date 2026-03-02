# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Offline inference example for DiNa-LRM reward scoring.

DiNa-LRM is a *discriminative* reward model that scores text-image alignment
in the SD3.5-M latent space.  It returns a scalar reward for each
(prompt, image) pair — higher is better.

Usage
-----
Score a prompt against a local image:

    python end2end.py \
        --model liuhuohuo/DiNa-LRM-SD35M-12layers \
        --prompt "A cat sitting on a wooden floor" \
        --image-path /path/to/image.png

Use a higher noise level (recommended for pipeline-generated latents):

    python end2end.py \
        --model liuhuohuo/DiNa-LRM-SD35M-12layers \
        --prompt "A red sports car" \
        --image-path /path/to/image.png \
        --noise-level 0.4

Fix the random seed for reproducible results:

    python end2end.py \
        --model liuhuohuo/DiNa-LRM-SD35M-12layers \
        --prompt "A cat sitting on a wooden floor" \
        --image-path /path/to/image.png \
        --seed 42
"""

import argparse
import os

import torch
from PIL import Image

from vllm_omni.entrypoints.omni_diffusion import OmniDiffusion
from vllm_omni.inputs.data import OmniDiffusionSamplingParams


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score (prompt, image) pairs with DiNa-LRM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default="liuhuohuo/DiNa-LRM-SD35M-12layers",
        help="DiNa-LRM HF repo ID or local checkpoint directory.",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Text prompt to score against the image.",
    )
    parser.add_argument(
        "--image-path",
        default=None,
        help="Path to a local image file (JPEG / PNG).",
    )
    parser.add_argument(
        "--noise-level",
        type=float,
        default=0.1,
        help=(
            "Noise sigma for the reward query.  "
            "0.1 → recommended for clean / disk images.  "
            "0.4 → recommended for pipeline-generated latents."
        ),
    )
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
        help="Model compute dtype.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Torch random seed for reproducibility.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    print(f"[Info] Torch random seed set to {args.seed}.")

    # ── Collect prompt ───────────────────────────────────────────────────────
    if not args.prompt:
        raise ValueError("No prompt provided. Use --prompt.")
    prompts = [args.prompt]

    # ── Load image ───────────────────────────────────────────────────────────
    if not args.image_path:
        raise ValueError("No image path provided. Use --image-path.")
    if not os.path.exists(args.image_path):
        raise FileNotFoundError(f"Image path not found: {args.image_path!r}")
    image = Image.open(args.image_path).convert("RGB")
    print(f"[Info] Loaded image from: {args.image_path}")

    # ── Build request prompts (one dict per prompt, sharing the same image) ─
    request_prompts = [{"prompt": p, "multi_modal_data": {"image": image}} for p in prompts]

    # ── Sampling params ──────────────────────────────────────────────────────
    sampling_params = OmniDiffusionSamplingParams(
        extra_args={"noise_level": args.noise_level},
    )

    # ── Run DiNa-LRM via OmniDiffusion ───────────────────────────────────────
    print(f"\n[Info] Loading DiNa-LRM: {args.model!r}  (dtype={args.dtype})")
    client = OmniDiffusion(model=args.model, dtype=args.dtype)

    print(f"[Info] Scoring {len(prompts)} prompt(s) …\n")
    outputs = client.generate(request_prompts, sampling_params)

    # ── Print results ────────────────────────────────────────────────────────
    print("=" * 60)
    print(f"{'Prompt':<45}  {'Raw score':>10}  {'Norm score':>10}")
    print("-" * 60)
    for prompt, out in zip(prompts, outputs):
        raw: torch.Tensor = out.latents[0][0]
        norm = (raw + 10.0) / 10.0
        raw_val = raw.item() if raw.numel() == 1 else raw.tolist()
        norm_val = norm.item() if norm.numel() == 1 else norm.tolist()
        truncated = prompt[:42] + "…" if len(prompt) > 45 else prompt
        print(f"{truncated:<45}  {raw_val:>10.4f}  {norm_val:>10.4f}")
    print("=" * 60)
    print("\nNormalised score = (raw + 10) / 10  — typically in [0, 2], centred ~1.")


if __name__ == "__main__":
    main()
