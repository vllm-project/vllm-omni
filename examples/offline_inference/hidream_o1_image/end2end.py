# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='HiDream-O1-Image end-to-end inference.')
    p.add_argument('--model', default='HiDream-ai/HiDream-O1-Image', help='HF repo or local path.')
    p.add_argument('--prompt', default="A cat holds a sign that says 'HiDream-O1 vLLM-Omni'", help='Text prompt.')
    p.add_argument('--height', type=int, default=1024, help='Image height.')
    p.add_argument('--width', type=int, default=1024, help='Image width.')
    p.add_argument('--seed', type=int, default=42, help='Random seed.')
    p.add_argument(
        '--steps',
        '--num-inference-steps',
        dest='steps',
        type=int,
        default=50,
        help='Denoising steps.',
    )
    p.add_argument(
        '--cfg-scale',
        '--guidance-scale',
        dest='guidance_scale',
        type=float,
        default=5.0,
        help='CFG scale.',
    )
    p.add_argument('--output', default='hidream_o1_output.png', help='Output PNG path.')
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print(
        f'[hidream_o1] model={args.model} size={args.width}x{args.height} '
        f'steps={args.steps} cfg={args.guidance_scale} seed={args.seed}'
    )

    omni = Omni(model=args.model, dtype=torch.bfloat16)
    try:
        outputs = omni.generate(
            prompts=[{'prompt': args.prompt}],
            sampling_params_list=[
                OmniDiffusionSamplingParams(
                    height=args.height,
                    width=args.width,
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance_scale,
                    seed=args.seed,
                )
            ],
        )
        result = OmniRequestOutput.unwrap_result(outputs)
    finally:
        omni.close()

    if result.final_output_type != 'image':
        raise ValueError(
            f"expected diffusion image output, got {result.final_output_type!r}"
        )
    if result.num_images != 1:
        raise ValueError(
            f'expected exactly one image, got num_images={result.num_images}'
        )

    img = result.images[0]
    if not isinstance(img, Image.Image):
        raise TypeError(f'expected PIL.Image.Image, got {type(img).__name__}')

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)
    print(f'[hidream_o1] saved {output_path} size={img.size}')

if __name__ == '__main__':
    main()
