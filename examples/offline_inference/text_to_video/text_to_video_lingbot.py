# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.platforms import current_omni_platform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text-to-video with the LingBot dense preset.")
    parser.add_argument("--model", default="/home/models/lingbot-video-dense-1.3b")
    parser.add_argument("--prompt", default="a robotic arm picks up a red block")
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--output", default="lingbot_video_output.mp4")
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--num-frames", type=int, default=9)
    parser.add_argument("--num-inference-steps", type=int, default=2)
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--flow-shift", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--output-type", choices=["pt", "np"], default="pt")
    parser.add_argument("--batch-cfg", action="store_true")
    return parser.parse_args()


def _peak_memory_mb(output: Any) -> float:
    result = output[0] if isinstance(output, list) and output else output
    return float(getattr(result, "peak_memory_mb", 0.0) or 0.0)


def _extract_video(output: Any) -> Any:
    if isinstance(output, list):
        output = output[0] if output else None
    if isinstance(output, OmniRequestOutput):
        if output.final_output_type != "image":
            raise ValueError(f"Unexpected output type {output.final_output_type!r}.")
        if not output.images:
            raise ValueError("No video frames found in OmniRequestOutput.")
        return output.images[0] if len(output.images) == 1 else output.images
    return output


def _to_frame_array(video: Any) -> np.ndarray:
    if isinstance(video, torch.Tensor):
        tensor = video.detach().cpu()
        if tensor.dim() == 5:
            tensor = tensor[0] if tensor.shape[1] not in (3, 4) else tensor[0].permute(1, 2, 3, 0)
        elif tensor.dim() == 4 and tensor.shape[0] in (3, 4):
            tensor = tensor.permute(1, 2, 3, 0)
        if tensor.is_floating_point():
            if float(tensor.min()) < 0.0:
                tensor = tensor.clamp(-1, 1) * 0.5 + 0.5
            else:
                tensor = tensor.clamp(0, 1)
        return tensor.float().numpy()
    if isinstance(video, np.ndarray):
        array = video[0] if video.ndim == 5 else video
        if np.issubdtype(array.dtype, np.integer):
            return array.astype(np.float32) / 255.0
        return array.astype(np.float32, copy=False)
    if isinstance(video, list):
        return np.stack([_to_frame_array(frame) for frame in video], axis=0)
    return np.asarray(video, dtype=np.float32)


def main() -> None:
    args = parse_args()
    generator = torch.Generator(device=current_omni_platform.device_type).manual_seed(args.seed)
    omni = Omni(
        model=args.model,
        model_class_name="LingBotVideoPipeline",
        flow_shift=args.flow_shift,
        parallel_config=DiffusionParallelConfig(),
    )

    prompt: dict[str, str] = {"prompt": args.prompt}
    if args.negative_prompt:
        prompt["negative_prompt"] = args.negative_prompt

    sampling_params = OmniDiffusionSamplingParams(
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
        output_type=args.output_type,
        extra_args={"batch_cfg": args.batch_cfg, "shift": args.flow_shift},
    )

    start = time.perf_counter()
    output = omni.generate(prompt, sampling_params)
    elapsed = time.perf_counter() - start
    print(f"Total generation time: {elapsed:.4f} seconds ({elapsed * 1000:.2f} ms)")
    peak_mb = _peak_memory_mb(output)
    if peak_mb:
        print(f"Worker peak GPU memory (reserved): {peak_mb:.2f} MiB ({peak_mb / 1024:.2f} GiB)")

    video = _to_frame_array(_extract_video(output))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    from diffusers.utils import export_to_video

    export_to_video(list(video), str(output_path), fps=args.fps)
    print(f"Saved generated video to {output_path}")
    omni.close()


if __name__ == "__main__":
    main()
