# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch

from vllm_omni.entrypoints.omni_diffusion import OmniDiffusion
from vllm_omni.inputs.data import OmniDiffusionSamplingParams


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline inference for DreamID-Omni (video + audio).")
    parser.add_argument("--ckpt-dir", required=True, help="DreamID ckpt root directory.")
    parser.add_argument("--prompt", required=True, help="Text prompt.")
    parser.add_argument(
        "--image-path",
        action="append",
        default=[],
        help="Reference image path. Can be specified multiple times (max 2).",
    )
    parser.add_argument(
        "--audio-path",
        action="append",
        default=[],
        help="Reference audio path. Can be specified multiple times (max 2, required).",
    )
    parser.add_argument("--height", type=int, default=720, help="Video height.")
    parser.add_argument("--width", type=int, default=720, help="Video width.")
    parser.add_argument("--num-inference-steps", type=int, default=45, help="Sampling steps.")
    parser.add_argument("--seed", type=int, default=103, help="Random seed.")
    parser.add_argument("--solver-name", default="unipc", help="Solver name: unipc|dpm++|euler.")
    parser.add_argument("--shift", type=float, default=5.0, help="Scheduler shift.")
    parser.add_argument("--video-guidance-scale", type=float, default=4.0, help="CFG scale for video.")
    parser.add_argument("--audio-guidance-scale", type=float, default=3.0, help="CFG scale for audio.")
    parser.add_argument("--slg-layer", type=int, default=9, help="SLG layer index.")
    parser.add_argument(
        "--video-negative-prompt",
        default="jitter, bad hands, blur, distortion",
        help="Negative prompt for video.",
    )
    parser.add_argument(
        "--audio-negative-prompt",
        default="robotic, muffled, echo, distorted",
        help="Negative prompt for audio.",
    )
    parser.add_argument("--model-name", default=None, help="DreamID model name, e.g. 960x960_5s.")
    parser.add_argument("--config-path", default=None, help="Path to inference_r2av.yaml.")
    parser.add_argument("--output", default="dreamid_output.mp4", help="Output video path.")
    parser.add_argument("--fps", type=int, default=24, help="FPS for output video.")
    parser.add_argument("--audio-sample-rate", type=int, default=16000, help="Audio sample rate for muxing.")
    parser.add_argument("--disable-dummy-run", action="store_true", help="Disable engine warmup dummy run.")
    return parser.parse_args()


def _normalize_frames(frames):
    if isinstance(frames, torch.Tensor):
        tensor = frames.detach().cpu()
        if tensor.dim() == 5:
            tensor = tensor[0]
        if tensor.dim() == 4 and tensor.shape[0] in (3, 4):
            tensor = tensor.permute(1, 2, 3, 0)
        if tensor.is_floating_point():
            tensor = tensor.clamp(-1, 1) * 0.5 + 0.5
        return tensor.float().numpy()
    if isinstance(frames, np.ndarray):
        array = frames
        if array.ndim == 5:
            array = array[0]
        if np.issubdtype(array.dtype, np.integer):
            array = array.astype(np.float32) / 255.0
        return array
    if isinstance(frames, list):
        normalized = []
        for frame in frames:
            normalized.append(_normalize_frames(frame))
        return normalized
    return frames


def _ensure_frame_list(video_array):
    if isinstance(video_array, list):
        if not video_array:
            return video_array
        first_item = video_array[0]
        if isinstance(first_item, np.ndarray):
            if first_item.ndim == 5:
                return list(first_item[0])
            if first_item.ndim == 4:
                return list(first_item)
            if first_item.ndim == 3:
                return video_array
        return video_array
    if isinstance(video_array, np.ndarray):
        if video_array.ndim == 5:
            return list(video_array[0])
        if video_array.ndim == 4:
            return list(video_array)
        if video_array.ndim == 3:
            return [video_array]
    return video_array


def _expand_paths(paths, kind: str):
    if kind == "image":
        exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    elif kind == "audio":
        exts = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}
    else:
        raise ValueError(f"Unknown kind: {kind}")

    expanded = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            candidates = [
                p for p in sorted(path.iterdir())
                if p.is_file() and p.suffix.lower() in exts
            ]
            if not candidates:
                raise ValueError(f"No {kind} files found in directory: {path}")
            expanded.append(str(candidates[0]))
        else:
            expanded.append(str(path))
    return expanded


def main() -> None:
    args = parse_args()

    if not args.audio_path:
        raise ValueError("DreamID requires at least one --audio-path reference.")

    image_paths = _expand_paths(args.image_path, "image") if args.image_path else []
    audio_paths = _expand_paths(args.audio_path, "audio") if args.audio_path else []

    prompt = {
        "prompt": args.prompt,
        "image_paths": image_paths[:2],
        "audio_paths": audio_paths[:2],
        "video_frame_height_width": [args.height, args.width],
    }

    sampling_params = OmniDiffusionSamplingParams(
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.video_guidance_scale,
        guidance_scale_2=args.audio_guidance_scale,
        seed=args.seed,
        extra_args={
            "solver_name": args.solver_name,
            "shift": args.shift,
            "slg_layer": args.slg_layer,
            "video_negative_prompt": args.video_negative_prompt,
            "audio_negative_prompt": args.audio_negative_prompt,
            "video_frame_height_width": [args.height, args.width],
        },
    )

    custom_pipeline_args = {}
    if args.model_name:
        custom_pipeline_args["model_name"] = args.model_name
    if args.config_path:
        custom_pipeline_args["config_path"] = args.config_path

    start = time.perf_counter()
    engine = OmniDiffusion(
        model=args.ckpt_dir,
        model_class_name="DreamIDOmniPipeline",
        custom_pipeline_args=custom_pipeline_args or None,
        disable_dummy_run=args.disable_dummy_run,
    )
    outputs = engine.generate(prompt, sampling_params)
    elapsed = time.perf_counter() - start

    if not outputs:
        raise RuntimeError("No output returned from DreamID-Omni.")

    output = outputs[0]
    video = output.images[0] if output.images else None
    audio = output.multimodal_output.get("audio") if output.multimodal_output else None
    audio_sample_rate = (
        output.multimodal_output.get("audio_sample_rate")
        if output.multimodal_output
        else None
    )

    if video is None:
        raise RuntimeError("No video output found.")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from diffusers.utils import export_to_video
    except ImportError as exc:
        raise ImportError("diffusers is required for export_to_video.") from exc

    video_array = _normalize_frames(video)
    video_array = _ensure_frame_list(video_array)

    use_audio_mux = False
    encode_video = None
    try:
        from diffusers.pipelines.ltx2.export_utils import encode_video

        use_audio_mux = True
    except Exception:
        use_audio_mux = False

    if use_audio_mux and encode_video is not None:
        if isinstance(video_array, list):
            frames_np = np.stack(video_array, axis=0)
        else:
            frames_np = np.asarray(video_array)
        if frames_np.ndim == 4 and frames_np.shape[-1] == 4:
            frames_np = frames_np[..., :3]
        frames_np = np.clip(frames_np, 0.0, 1.0)
        frames_u8 = (frames_np * 255).round().clip(0, 255).astype("uint8")
        video_tensor = torch.from_numpy(frames_u8)

        audio_out = None
        if audio is not None:
            if isinstance(audio, list):
                audio = audio[0] if audio else None
            if isinstance(audio, np.ndarray):
                audio = torch.from_numpy(audio)
            if isinstance(audio, torch.Tensor):
                audio_out = audio
                if audio_out.dim() > 1:
                    audio_out = audio_out[0]
                audio_out = audio_out.float().cpu()

        encode_video(
            video_tensor,
            fps=args.fps,
            audio=audio_out,
            audio_sample_rate=(
                audio_sample_rate or args.audio_sample_rate if audio_out is not None else None
            ),
            output_path=str(output_path),
        )
    else:
        export_to_video(video_array, str(output_path), fps=args.fps)

    print(f"Saved generated video to {output_path}")
    print(f"Total time: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
