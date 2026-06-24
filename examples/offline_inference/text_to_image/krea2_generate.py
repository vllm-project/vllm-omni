# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generate images with Krea 2 (base or distilled/TDM checkpoint).

Usage:
    # Base model (28 steps, guidance 4.5)
    python krea2_generate.py --model krea-ai/krea-2-medium

    # Distilled / TDM model (8 steps, no CFG)
    python krea2_generate.py --model krea-ai/krea-2-medium-tdm \
        --steps 8 --guidance 0.0

    # Custom prompt and resolution
    python krea2_generate.py --model krea-ai/krea-2-large \
        --prompt "a serene Vermont mountain lake at dawn" \
        --height 768 --width 1360
"""

import argparse
import time
from pathlib import Path

import torch

from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.platforms import current_omni_platform


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Krea 2 text-to-image generation")
    p.add_argument("--model", default="krea-ai/krea-2-medium",
                   help="HF repo or local path to a Krea 2 checkpoint.")
    p.add_argument("--prompt", default="a cup of coffee on the table",
                   help="Text prompt for image generation.")
    p.add_argument("--negative-prompt", default=None,
                   help="Negative prompt for CFG (ignored when guidance=0).")
    p.add_argument("--steps", type=int, default=28,
                   help="Number of denoising steps (28 base, 8 distilled).")
    p.add_argument("--guidance", type=float, default=4.5,
                   help="Guidance scale (4.5 base, 0 distilled).")
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--seed", type=int, default=142)
    p.add_argument("--output", default="krea2_output.png",
                   help="Output image path.")
    p.add_argument("--num-images", type=int, default=1,
                   help="Number of images to generate per prompt.")
    p.add_argument("--quantization", type=str, default=None,
                   choices=["fp8", "int8"],
                   help="Quantization method for the transformer.")
    p.add_argument("--cfg-parallel-size", type=int, default=1,
                   choices=[1, 2],
                   help="GPUs used for CFG parallelism.")
    p.add_argument("--tensor-parallel-size", type=int, default=1,
                   help="GPUs used for tensor parallelism.")
    p.add_argument("--enforce-eager", action="store_true",
                   help="Disable torch.compile.")
    p.add_argument("--enable-cpu-offload", action="store_true")
    current_omni_platform.pre_register_and_update(p)
    return p.parse_args()


def main():
    args = parse_args()
    generator = torch.Generator(
        device=current_omni_platform.device_type
    ).manual_seed(args.seed)

    is_distilled = "tdm" in args.model.lower() or "distill" in args.model.lower()
    if is_distilled and args.steps == 28:
        args.steps = 8
    if is_distilled and args.guidance == 4.5:
        args.guidance = 0.0

    omni_kwargs = {
        "model": args.model,
        "mode": "text-to-image",
        "cfg_parallel_size": args.cfg_parallel_size,
        "tensor_parallel_size": args.tensor_parallel_size,
        "enforce_eager": args.enforce_eager,
        "enable_cpu_offload": args.enable_cpu_offload,
    }
    if is_distilled:
        omni_kwargs["model_config"] = {"is_distilled": True}
    if args.quantization:
        omni_kwargs["quantization"] = args.quantization

    omni = Omni(**omni_kwargs)

    prompt_dict = {"prompt": args.prompt}
    if args.negative_prompt:
        prompt_dict["negative_prompt"] = args.negative_prompt

    sampling_params = OmniDiffusionSamplingParams(
        height=args.height,
        width=args.width,
        seed=args.seed,
        generator=generator,
        guidance_scale=args.guidance,
        num_inference_steps=args.steps,
        num_outputs_per_prompt=args.num_images,
    )

    print(f"Generating with Krea 2 ({'distilled' if is_distilled else 'base'})")
    print(f"  Steps: {args.steps}  Guidance: {args.guidance}")
    print(f"  Size: {args.width}x{args.height}  Seed: {args.seed}")

    t0 = time.perf_counter()
    outputs = omni.generate(prompt_dict, sampling_params_list=[sampling_params])
    elapsed = time.perf_counter() - t0
    print(f"Generated in {elapsed:.2f}s")

    images = None
    for output in outputs:
        images = getattr(output, "images", None)
        if images:
            break
        req_out = getattr(output, "request_output", None)
        images = getattr(req_out, "images", None) if req_out else None
        if images:
            break

    if not images:
        raise ValueError("No images found in output")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if len(images) == 1:
        images[0].save(out)
        print(f"Saved to {out}")
    else:
        for i, img in enumerate(images):
            p = out.parent / f"{out.stem}_{i}{out.suffix or '.png'}"
            img.save(p)
            print(f"Saved to {p}")


if __name__ == "__main__":
    main()
