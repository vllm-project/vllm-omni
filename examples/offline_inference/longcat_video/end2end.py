# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import imageio
import numpy as np
import torch

from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput


def parse_args():
    parser = argparse.ArgumentParser(description="Generate LongCat-Video-Avatar A2V/AI2V video.")
    parser.add_argument("--model", default="meituan-longcat/LongCat-Video-Avatar-1.5")
    parser.add_argument("--base-model-dir", default=None, help="Optional local LongCat-Video base model directory.")
    parser.add_argument("--stage", choices=["at2v", "ai2v"], default="at2v")
    parser.add_argument("--prompt", default="A person speaks calmly while facing the camera.")
    parser.add_argument("--negative-prompt", default="low quality, blurry, watermark, text")
    parser.add_argument("--audio", required=True, help="Input speech audio path.")
    parser.add_argument("--image", default=None, help="Input reference image path for AI2V.")
    parser.add_argument("--output", default="longcat_avatar_output.mp4")
    parser.add_argument("--resolution", choices=["480p", "720p"], default="480p")
    parser.add_argument("--num-frames", type=int, default=93)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--num-inference-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-distill", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-int8", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enforce-eager", action="store_true")
    return parser.parse_args()


def _extract_frames(output: Any):
    if isinstance(output, list):
        output = output[0] if output else None
    if isinstance(output, OmniRequestOutput):
        if not output.images:
            raise ValueError("No video frames found in OmniRequestOutput.")
        frames = output.images[0] if len(output.images) == 1 else output.images
        if isinstance(frames, tuple) and len(frames) == 2:
            frames = frames[0]
        if isinstance(frames, dict):
            frames = frames.get("frames") or frames.get("video")
    else:
        frames = output
    if frames is None:
        raise ValueError("No video frames found in output.")
    return frames


def _frames_to_uint8(frames):
    frames_np = np.asarray([np.asarray(frame) for frame in frames])
    if frames_np.ndim == 5:
        frames_np = frames_np[0]
    if np.issubdtype(frames_np.dtype, np.floating):
        frames_np = (np.clip(frames_np, 0.0, 1.0) * 255).round()
    return np.clip(frames_np, 0, 255).astype(np.uint8)


def _save_video_with_audio(frames, output_path: Path, audio_path: str, fps: int) -> None:
    frames_np = _frames_to_uint8(frames)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="longcat_avatar_export_") as tmp:
        tmp_dir = Path(tmp)
        temp_video = tmp_dir / "video.mp4"
        crop_audio = tmp_dir / "audio.wav"
        writer = imageio.get_writer(str(temp_video), fps=fps, quality=5)
        try:
            for frame in frames_np:
                writer.append_data(frame)
        finally:
            writer.close()

        duration = len(frames_np) / fps
        subprocess.run(["ffmpeg", "-y", "-i", audio_path, "-t", f"{duration}", str(crop_audio)], check=True)
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(temp_video),
                "-i",
                str(crop_audio),
                "-c:v",
                "libx264",
                "-c:a",
                "aac",
                "-shortest",
                str(output_path),
            ],
            check=True,
        )


def main():
    args = parse_args()
    if args.stage == "ai2v" and not args.image:
        raise ValueError("--image is required when --stage=ai2v.")

    additional_config = {
        "model_type": "avatar-v1.5",
        "resolution": args.resolution,
        "use_distill": args.use_distill,
        "use_int8": args.use_int8,
    }
    if args.base_model_dir:
        additional_config["base_model_dir"] = args.base_model_dir

    omni = Omni(
        model=args.model,
        model_class_name="LongCatVideoAvatarPipeline",
        dtype="bfloat16",
        enforce_eager=args.enforce_eager,
        additional_config=additional_config,
        parallel_config=DiffusionParallelConfig(
            ulysses_degree=1,
            ring_degree=1,
            cfg_parallel_size=1,
            tensor_parallel_size=1,
            vae_patch_parallel_size=1,
            pipeline_parallel_size=1,
        ),
    )

    prompt = {
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "multi_modal_data": {"audio": args.audio},
        "additional_information": {"stage": args.stage, "resolution": args.resolution},
    }
    if args.stage == "ai2v":
        prompt["multi_modal_data"]["image"] = args.image

    try:
        output = omni.generate(
            prompt,
            OmniDiffusionSamplingParams(
                generator=torch.Generator(device="cuda").manual_seed(args.seed),
                seed=args.seed,
                guidance_scale=1.0 if args.use_distill else 4.0,
                guidance_scale_2=1.0 if args.use_distill else 4.0,
                num_inference_steps=args.num_inference_steps,
                num_frames=args.num_frames,
                fps=args.fps,
                extra_args={
                    "stage": args.stage,
                    "resolution": args.resolution,
                    "save_fps": args.fps,
                    "use_distill": args.use_distill,
                    "use_int8": args.use_int8,
                },
            ),
            use_tqdm=True,
        )
    finally:
        omni.close()

    _save_video_with_audio(_extract_frames(output), Path(args.output), args.audio, args.fps)
    print(f"Saved video to {args.output}")


if __name__ == "__main__":
    main()
