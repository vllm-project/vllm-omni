# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import time
from pathlib import Path

import torch

from vllm_omni.diffusion.data import DiffusionParallelConfig, OmniDiffusionConfig
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.utils.platform_utils import detect_device_type, is_npu


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an image with Qwen-Image.")
    parser.add_argument("--model", default="Qwen/Qwen-Image", help="Diffusion model name or local path.")
    parser.add_argument("--prompt", default="a cup of coffee on the table", help="Text prompt for image generation.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic results.")
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=4.0,
        help="True classifier-free guidance scale specific to Qwen-Image.",
    )
    parser.add_argument("--height", type=int, default=1024, help="Height of generated image.")
    parser.add_argument("--width", type=int, default=1024, help="Width of generated image.")
    parser.add_argument(
        "--output",
        type=str,
        default="qwen_image_output_sp.png",
        help="Path to save the generated image (PNG).",
    )
    parser.add_argument(
        "--num_images_per_prompt",
        type=int,
        default=1,
        help="Number of images to generate for the given prompt.",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of denoising steps for the diffusion sampler.",
    )
    parser.add_argument(
        "--ulysses_degree",
        type=int,
        default=2,
        help="Number of GPUs used for ulysses sequence parallelism.",
    )
    parser.add_argument(
        "--ring_degree",
        type=int,
        default=1,
        help="Number of GPUs used for ring sequence parallelism.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = detect_device_type()
    generator = torch.Generator(device=device).manual_seed(args.seed)
    # Enable VAE memory optimizations on NPU
    vae_use_slicing = is_npu()
    vae_use_tiling = is_npu()

    assert args.ring_degree == 1, "Ring attention is not supported yet"

    config_kwargs = {
        "model": args.model,
        "vae_use_slicing": vae_use_slicing,
        "vae_use_tiling": vae_use_tiling,
    }

    omni_diffusion_config = OmniDiffusionConfig(
        **config_kwargs, parallel_config=DiffusionParallelConfig(ulysses_degree=args.ulysses_degree)
    )

    omni = Omni(
        **config_kwargs,
        od_config=omni_diffusion_config,
    )

    start_time = time.time()
    images = omni.generate(
        args.prompt,
        height=args.height,
        width=args.width,
        generator=generator,
        true_cfg_scale=args.cfg_scale,
        num_inference_steps=args.num_inference_steps,
        num_images_per_prompt=args.num_images_per_prompt,
        num_outputs_per_prompt=args.num_images_per_prompt,
    )
    end_time = time.time()
    elapsed_time = end_time - start_time

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix or ".png"
    stem = output_path.stem or "qwen_image_output"
    if args.num_images_per_prompt <= 1:
        images[0].save(output_path)
        print(f"Saved generated image to {output_path}")
    else:
        for idx, img in enumerate(images):
            save_path = output_path.parent / f"{stem}_{idx}{suffix}"
            img.save(save_path)
            print(f"Saved generated image to {save_path}")

    print(
        f"inference time: {elapsed_time:.2f} sec, average time per image: {elapsed_time / args.num_images_per_prompt:.2f} sec"
    )


if __name__ == "__main__":
    main()
