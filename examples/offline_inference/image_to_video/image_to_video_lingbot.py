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
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.platforms import current_omni_platform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text-image-to-video with LingBot-Video.")
    parser.add_argument("--model", default="robbyant/lingbot-video-dense-1.3b")
    parser.add_argument("--image", required=True, help="Path to the first-frame image.")
    parser.add_argument("--prompt", default="the fox turns its head toward the camera")
    parser.add_argument("--negative-prompt", default=None)
    parser.add_argument("--output", default="lingbot_ti2v_output.mp4")
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--num-frames", type=int, default=9)
    parser.add_argument("--num-inference-steps", type=int, default=2)
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--flow-shift", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=24)
    return parser.parse_args()


def _extract_video(output: Any) -> Any:
    if isinstance(output, list):
        output = output[0] if output else None
    if isinstance(output, OmniRequestOutput):
        if not output.images:
            raise ValueError("No video frames found in OmniRequestOutput.")
        return output.images[0] if len(output.images) == 1 else output.images
    return output


def _to_frame_array(video: Any) -> np.ndarray:
    if isinstance(video, torch.Tensor):
        tensor = video.detach().cpu()
        if tensor.dim() == 5:
            tensor = tensor[0]
        if tensor.dim() == 4 and tensor.shape[-1] not in (3, 4):
            if tensor.shape[0] in (3, 4):
                tensor = tensor.permute(1, 2, 3, 0)
            elif tensor.shape[1] in (3, 4):
                tensor = tensor.permute(0, 2, 3, 1)
        if tensor.is_floating_point():
            tensor = tensor.clamp(-1, 1) * 0.5 + 0.5 if float(tensor.min()) < 0.0 else tensor.clamp(0, 1)
        return tensor.float().numpy()
    if isinstance(video, np.ndarray):
        array = video[0] if video.ndim == 5 else video
        return array.astype(np.float32) / 255.0 if np.issubdtype(array.dtype, np.integer) else array
    if isinstance(video, list):
        return np.stack([np.asarray(frame, dtype=np.float32) / 255.0 for frame in video], axis=0)
    return np.asarray(video, dtype=np.float32)


def main() -> None:
    args = parse_args()
    image = Image.open(args.image).convert("RGB")
    omni = Omni(
        model=args.model,
        model_class_name="LingBotVideoPipeline",
        flow_shift=args.flow_shift,
        parallel_config=DiffusionParallelConfig(),
    )
    prompt: dict[str, Any] = {
        "prompt": args.prompt,
        "modalities": ["video"],
        "multi_modal_data": {"image": image},
    }
    if args.negative_prompt is not None:
        prompt["negative_prompt"] = args.negative_prompt

    generator = torch.Generator(device=current_omni_platform.device_type).manual_seed(args.seed)
    sampling_params = OmniDiffusionSamplingParams(
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
        output_type="np",
        extra_args={"shift": args.flow_shift},
    )

    try:
        video = _to_frame_array(_extract_video(omni.generate(prompt, sampling_params)))
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        from diffusers.utils import export_to_video

        export_to_video(list(video), str(output_path), fps=args.fps)
        print(f"Saved generated video to {output_path}")
    finally:
        omni.close()


if __name__ == "__main__":
    main()
