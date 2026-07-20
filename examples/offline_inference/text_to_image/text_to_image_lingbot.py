# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.platforms import current_omni_platform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a text-to-image result with LingBot-Video.")
    parser.add_argument("--model", default="robbyant/lingbot-video-dense-1.3b")
    parser.add_argument("--prompt", default="a red fox standing in fresh snow")
    parser.add_argument("--negative-prompt", default=None)
    parser.add_argument("--output", default="lingbot_image_output.png")
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--num-inference-steps", type=int, default=2)
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--flow-shift", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _to_pil_image(value: Any) -> Image.Image:
    if isinstance(value, Image.Image):
        return value
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().float().numpy()
    array = np.asarray(value)
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 3 and array.shape[0] in (3, 4):
        array = np.transpose(array, (1, 2, 0))
    if array.ndim not in (2, 3):
        raise ValueError(f"Unexpected LingBot image shape {array.shape!r}.")
    if np.issubdtype(array.dtype, np.floating):
        array = array * 0.5 + 0.5 if float(array.min()) < 0.0 else array
        array = np.clip(array, 0.0, 1.0) * 255.0
    return Image.fromarray(array.astype(np.uint8))


def main() -> None:
    args = parse_args()
    omni = Omni(
        model=args.model,
        model_class_name="LingBotVideoPipeline",
        flow_shift=args.flow_shift,
        parallel_config=DiffusionParallelConfig(),
    )
    prompt: dict[str, object] = {
        "prompt": args.prompt,
        "modalities": ["image"],
    }
    if args.negative_prompt is not None:
        prompt["negative_prompt"] = args.negative_prompt

    generator = torch.Generator(device=current_omni_platform.device_type).manual_seed(args.seed)
    sampling_params = OmniDiffusionSamplingParams(
        height=args.height,
        width=args.width,
        num_frames=1,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
        output_type="np",
        extra_args={"shift": args.flow_shift},
    )

    try:
        outputs = omni.generate(prompt, sampling_params)
        if not outputs or not outputs[0].images:
            raise ValueError("LingBot text-to-image generation returned no image.")
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _to_pil_image(outputs[0].images[0]).save(output_path)
        print(f"Saved generated image to {output_path}")
    finally:
        omni.close()


if __name__ == "__main__":
    main()
