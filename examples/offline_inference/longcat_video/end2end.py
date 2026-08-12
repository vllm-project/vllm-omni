# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import json
import re
import subprocess
import tempfile
import wave
from functools import lru_cache
from pathlib import Path
from typing import Any

import imageio
import numpy as np
import torch

from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.diffusion.models.longcat_video.pipeline_longcat_video_avatar import (
    prepare_longcat_video_avatar_model_for_omni,
)
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.platforms import current_omni_platform


def parse_args():
    parser = argparse.ArgumentParser(description="Generate LongCat-Video-Avatar A2V/AI2V video.")
    parser.add_argument("--model", default="meituan-longcat/LongCat-Video-Avatar-1.5")
    parser.add_argument("--base-model-dir", default=None, help="Optional local LongCat-Video base model directory.")
    parser.add_argument("--stage", choices=["at2v", "ai2v"], default="at2v")
    parser.add_argument("--prompt", default="A person speaks calmly while facing the camera.")
    parser.add_argument("--negative-prompt", default="low quality, blurry, watermark, text")
    parser.add_argument("--audio", default=None, help="Input speech audio path.")
    parser.add_argument("--image", default=None, help="Input reference image path for AI2V.")
    parser.add_argument("--input-json", default=None, help="Official LongCat Avatar JSON case for AI2V.")
    parser.add_argument("--output", default="longcat_avatar_output.mp4")
    parser.add_argument("--resolution", choices=["480p", "720p"], default="480p")
    parser.add_argument("--num-frames", type=int, default=93)
    parser.add_argument("--num-segments", default="1", help="Number of AVC segments, or 'auto' to cover full audio.")
    parser.add_argument("--num-cond-frames", type=int, default=13)
    parser.add_argument("--ref-img-index", type=int, default=10)
    parser.add_argument("--mask-frame-range", type=int, default=3)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--num-inference-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-distill", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-int8", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--build-components-on-gpu",
        action="store_true",
        default=False,
        help="Build large Avatar components directly on GPU for faster startup. Requires more peak VRAM.",
    )
    parser.add_argument("--use-kv-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--offload-kv-cache", action=argparse.BooleanOptionalAction, default=False)
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


@lru_cache(maxsize=1)
def _ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        raise RuntimeError("LongCat Avatar example requires the Python imageio[ffmpeg] dependency.") from exc


def _wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as audio:
        return audio.getnframes() / float(audio.getframerate())


def _save_video_with_audio(frames, output_path: Path, audio_path: str, fps: int) -> None:
    frames_np = _frames_to_uint8(frames)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="longcat_avatar_export_") as tmp:
        tmp_dir = Path(tmp)
        temp_video = tmp_dir / "video.mp4"
        crop_audio = tmp_dir / "audio.wav"
        crop_video = tmp_dir / "video_cropped.mp4"
        writer = imageio.get_writer(str(temp_video), fps=fps, quality=5)
        try:
            for frame in frames_np:
                writer.append_data(frame)
        finally:
            writer.close()

        duration = len(frames_np) / fps
        ffmpeg = _ffmpeg_exe()
        subprocess.run([ffmpeg, "-y", "-i", audio_path, "-t", f"{duration}", str(crop_audio)], check=True)
        audio_duration = _wav_duration(crop_audio)
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-i",
                str(temp_video),
                "-t",
                f"{audio_duration}",
                "-c:v",
                "copy",
                str(crop_video),
            ],
            check=True,
        )
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-i",
                str(crop_video),
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


def _infer_asset_root_from_input_json(input_json: Path) -> Path:
    resolved = input_json.resolve()
    parts = resolved.parts
    if "assets" not in parts:
        return resolved.parent
    asset_idx = parts.index("assets")
    return Path(*parts[:asset_idx])


def _resolve_case_path(path: str, asset_root: Path) -> str:
    raw_path = Path(path)
    if raw_path.is_absolute():
        return str(raw_path)
    candidate = asset_root / raw_path
    return str(candidate if candidate.exists() else raw_path)


def _speaker_sort_key(name: str) -> tuple[int, str]:
    match = re.search(r"(\d+)$", str(name))
    return (int(match.group(1)) if match else 10_000, str(name))


def _load_input_case(input_json: str) -> tuple[dict[str, Any], Path]:
    input_json_path = Path(input_json)
    with input_json_path.open("r", encoding="utf-8") as f:
        case = json.load(f)
    return case, _infer_asset_root_from_input_json(input_json_path)


def _create_merged_audio_for_case(
    audio_paths: dict[str, str],
    audio_type: str,
    output_path: Path,
) -> str:
    sorted_paths = [path for _, path in sorted(audio_paths.items(), key=lambda item: _speaker_sort_key(str(item[0])))]
    if len(sorted_paths) == 1:
        return sorted_paths[0]
    inputs = [arg for path in sorted_paths for arg in ("-i", path)]
    audio_type = audio_type.lower()
    if audio_type == "para":
        filter_complex = f"amix=inputs={len(sorted_paths)}:duration=longest:normalize=0"
    elif audio_type == "add":
        filter_complex = "".join(f"[{idx}:a]" for idx in range(len(sorted_paths)))
        filter_complex += f"concat=n={len(sorted_paths)}:v=0:a=1"
    else:
        raise NotImplementedError(f"Unsupported LongCat Avatar audio_type {audio_type!r}.")
    subprocess.run([_ffmpeg_exe(), "-y", *inputs, "-filter_complex", filter_complex, str(output_path)], check=True)
    return str(output_path)


def main():
    args = parse_args()
    input_case = None
    input_asset_root = None
    audio_for_mux = args.audio
    if args.input_json:
        input_case, input_asset_root = _load_input_case(args.input_json)
        args.stage = "ai2v"
        args.prompt = input_case.get("prompt") or args.prompt
        args.image = _resolve_case_path(input_case["cond_image"], input_asset_root)
        args.audio = None
    if not args.input_json and not args.audio:
        raise ValueError("--audio is required unless --input-json is provided.")
    if args.stage == "ai2v" and not args.image:
        raise ValueError("--image is required when --stage=ai2v.")
    model_for_omni = prepare_longcat_video_avatar_model_for_omni(args.model, args.use_int8)

    additional_config = {
        "model_type": "avatar-v1.5",
        "resolution": args.resolution,
        "use_distill": args.use_distill,
        "use_int8": args.use_int8,
        "build_components_on_gpu": args.build_components_on_gpu,
    }
    if args.base_model_dir:
        additional_config["base_model_dir"] = args.base_model_dir

    omni = Omni(
        model=model_for_omni,
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

    audio_input: str | dict[str, str] | None = args.audio
    additional_information = {"stage": args.stage, "resolution": args.resolution}
    if input_case is not None and input_asset_root is not None:
        audio_input = {
            speaker: _resolve_case_path(path, input_asset_root)
            for speaker, path in sorted(
                input_case["cond_audio"].items(),
                key=lambda item: _speaker_sort_key(str(item[0])),
            )
        }
        additional_information.update(
            {
                "input_json": args.input_json,
                "audio_type": input_case.get("audio_type", "para"),
                "bbox": input_case.get("bbox", {}),
            }
        )

    prompt = {
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "multi_modal_data": {"audio": audio_input},
        "additional_information": additional_information,
    }
    if args.stage == "ai2v":
        prompt["multi_modal_data"]["image"] = args.image

    try:
        output = omni.generate(
            prompt,
            OmniDiffusionSamplingParams(
                generator=torch.Generator(device=current_omni_platform.device_type).manual_seed(args.seed),
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
                    "input_json": args.input_json,
                    "num_segments": args.num_segments,
                    "num_cond_frames": args.num_cond_frames,
                    "ref_img_index": args.ref_img_index,
                    "mask_frame_range": args.mask_frame_range,
                    "use_kv_cache": args.use_kv_cache,
                    "offload_kv_cache": args.offload_kv_cache,
                },
            ),
            use_tqdm=True,
        )
    finally:
        omni.close()

    with tempfile.TemporaryDirectory(prefix="longcat_avatar_audio_mux_") as tmp:
        if input_case is not None and isinstance(audio_input, dict):
            audio_for_mux = _create_merged_audio_for_case(
                audio_input,
                input_case.get("audio_type", "para"),
                Path(tmp) / "merged_audio.wav",
            )
        _save_video_with_audio(_extract_frames(output), Path(args.output), audio_for_mux, args.fps)
    print(f"Saved video to {args.output}")


if __name__ == "__main__":
    main()
