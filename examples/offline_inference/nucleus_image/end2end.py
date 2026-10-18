# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
from pathlib import Path

import torch

from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.platforms import current_omni_platform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Nucleus-Image offline text-to-image example.")
    parser.add_argument("--model", default="NucleusAI/Nucleus-Image", help="Model name or local path.")
    parser.add_argument(
        "--prompt",
        default="A weathered lighthouse on a rocky coastline at golden hour.",
        help="Positive text prompt.",
    )
    parser.add_argument(
        "--negative-prompt",
        default="blurry, low quality",
        help="Negative prompt for classifier-free guidance.",
    )
    parser.add_argument("--height", type=int, default=1024, help="Output image height.")
    parser.add_argument("--width", type=int, default=1024, help="Output image width.")
    parser.add_argument("--num-inference-steps", type=int, default=50, help="Number of denoising steps.")
    parser.add_argument("--guidance-scale", type=float, default=4.0, help="Classifier-free guidance scale.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--num-images-per-prompt", type=int, default=1, help="Number of output images.")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="DiT tensor parallel size.")
    parser.add_argument("--ulysses-degree", type=int, default=1, help="Ulysses sequence parallel degree.")
    parser.add_argument("--ring-degree", type=int, default=1, help="Ring sequence parallel degree.")
    parser.add_argument("--enable-layerwise-offload", action="store_true", help="Enable layerwise CPU offload.")
    parser.add_argument("--enable-cpu-offload", action="store_true", help="Enable CPU offload.")
    parser.add_argument(
        "--quantization",
        type=str,
        default=None,
        choices=["fp8", "int8", "gguf"],
        help="Optional transformer quantization mode.",
    )
    parser.add_argument("--output", type=str, default="nucleus_image_output.png", help="Output image path.")
    parser.add_argument("--log-stats", action="store_true", help="Enable diffusion pipeline stats logging.")
    parser.add_argument("--init-timeout", type=int, default=600, help="Initialization timeout in seconds.")
    parser.add_argument("--stage-init-timeout", type=int, default=600, help="Single-stage init timeout in seconds.")

    current_omni_platform.pre_register_and_update(parser)
    from vllm_omni.engine.arg_utils import nullify_stage_engine_defaults

    nullify_stage_engine_defaults(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generator = torch.Generator(device=current_omni_platform.device_type).manual_seed(args.seed)

    omni = Omni(
        model=args.model,
        mode="text-to-image",
        tensor_parallel_size=args.tensor_parallel_size,
        ulysses_degree=args.ulysses_degree,
        ring_degree=args.ring_degree,
        enable_layerwise_offload=args.enable_layerwise_offload,
        enable_cpu_offload=args.enable_cpu_offload,
        quantization=args.quantization,
        log_stats=args.log_stats,
        init_timeout=args.init_timeout,
        stage_init_timeout=args.stage_init_timeout,
    )

    outputs = omni.generate(
        {
            "prompt": args.prompt,
            "negative_prompt": args.negative_prompt,
        },
        OmniDiffusionSamplingParams(
            height=args.height,
            width=args.width,
            generator=generator,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            num_outputs_per_prompt=args.num_images_per_prompt,
        ),
    )

    if not outputs:
        raise ValueError("No output generated from omni.generate().")

    first_output = outputs[0]
    req_out = getattr(first_output, "request_output", None)
    if req_out is None or not getattr(req_out, "images", None):
        raise ValueError("No images found in request output.")

    images = req_out.images
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if len(images) == 1:
        images[0].save(output_path)
        print(f"Saved generated image to {output_path}")
    else:
        stem = output_path.stem or "nucleus_image_output"
        suffix = output_path.suffix or ".png"
        for i, image in enumerate(images):
            image_path = output_path.with_name(f"{stem}_{i}{suffix}")
            image.save(image_path)
            print(f"Saved generated image to {image_path}")


if __name__ == "__main__":
    main()
