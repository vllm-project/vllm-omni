# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from diffusers.utils import export_to_video

from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput


def _safe_camera_name(camera: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", camera).strip("_") or "camera"


def _extract_payload(value: Any) -> tuple[Any, dict[str, Any]]:
    if isinstance(value, list) and len(value) == 1:
        return _extract_payload(value[0])
    if isinstance(value, OmniRequestOutput):
        if value.images:
            return _extract_payload(value.images[0] if len(value.images) == 1 else value.images)
        raise ValueError("Cosmos3 multiview inference returned no video output.")
    if isinstance(value, dict):
        metadata = value.get("metadata") if isinstance(value.get("metadata"), dict) else {}
        payload = value.get("payload") if isinstance(value.get("payload"), dict) else value
        if "video" in payload:
            return payload["video"], metadata
    return value, {}


def _frame_list(video: Any) -> list[Any]:
    if isinstance(video, torch.Tensor):
        tensor = video.detach().cpu()
        if tensor.ndim == 5:
            tensor = tensor[0]
        if tensor.ndim == 4 and tensor.shape[0] in (3, 4):
            tensor = tensor.permute(1, 2, 3, 0)
        if tensor.is_floating_point() and tensor.numel() and tensor.min() < 0:
            tensor = tensor.mul(0.5).add(0.5)
        return list(tensor.clamp(0, 1).numpy())
    if isinstance(video, np.ndarray):
        array = video[0] if video.ndim == 5 else video
        if np.issubdtype(array.dtype, np.integer):
            array = array.astype(np.float32) / 255.0
        return list(array)
    if isinstance(video, list):
        if len(video) == 1 and isinstance(video[0], list):
            return video[0]
        if len(video) == 1 and isinstance(video[0], np.ndarray) and video[0].ndim == 4:
            return list(video[0])
        return video
    raise TypeError(f"Unsupported multiview video output type: {type(video).__name__}.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Exported Cosmos3 Multiview-AV Diffusers directory")
    parser.add_argument("--input", required=True, type=Path, help="Multiview JSON file")
    parser.add_argument("--output-dir", type=Path, default=Path("cosmos3_multiview_output"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--negative-prompt-json",
        type=Path,
        help=(
            "Structured negative prompt to serialize with json.dumps defaults. The pipeline ships no default "
            "negative prompt, so reference-parity runs must supply the reference one here."
        ),
    )
    args = parser.parse_args()

    request = json.loads(args.input.read_text())
    multiview = request.get("multiview")
    if not isinstance(multiview, dict):
        raise ValueError("Input JSON must contain a multiview object.")
    views = multiview.get("views")
    if not isinstance(views, list) or not views:
        raise ValueError("Input JSON must contain multiview.views.")

    extra_args = {"multiview": multiview, "wsm": request.get("wsm", {})}
    sampling_params = OmniDiffusionSamplingParams(
        height=480,
        width=832,
        num_frames=int(request.get("num_frames", 93)),
        fps=int(request.get("fps", 10)),
        num_inference_steps=int(request.get("num_inference_steps", 35)),
        guidance_scale=float(request.get("guidance_scale", 6.0)),
        seed=args.seed,
        extra_args=extra_args,
    )
    prompt = {
        "prompt": str(request.get("prompt", "")),
        "modalities": ["video"],
    }
    if request.get("negative_prompt") is not None:
        prompt["negative_prompt"] = request["negative_prompt"]
    elif args.negative_prompt_json is not None:
        # Default separators (", " and ": ") and the file's key order are part
        # of the reference's serialization, so keep json.dumps unconfigured.
        prompt["negative_prompt"] = json.dumps(json.loads(args.negative_prompt_json.read_text()))

    omni = Omni(
        model=args.model,
        dtype="bfloat16",
        model_class_name="Cosmos3MultiviewPipeline",
        enforce_eager=False,
        ulysses_degree=1,
        ring_degree=1,
        cfg_parallel_size=1,
        diffusion_compile_granularity="regional",
        diffusion_compile_dynamic=False,
    )
    result = omni.generate(prompt, sampling_params)
    video, metadata = _extract_payload(result)
    frames = _frame_list(video)

    frames_per_view = int(metadata.get("multiview", {}).get("frames_per_view", sampling_params.num_frames))
    cameras = metadata.get("multiview", {}).get("cameras") or [view["camera_key"] for view in views]
    fps = float(metadata.get("multiview", {}).get("fps", sampling_params.fps or 10))
    if len(frames) != len(cameras) * frames_per_view:
        raise ValueError(f"Expected {len(cameras) * frames_per_view} camera-major frames, got {len(frames)}.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    files_by_camera = {}
    for index, camera in enumerate(cameras):
        camera_frames = frames[index * frames_per_view : (index + 1) * frames_per_view]
        output_path = args.output_dir / f"vision_view{index:02d}_{_safe_camera_name(camera)}.mp4"
        export_to_video(camera_frames, str(output_path), fps=fps)
        files_by_camera[camera] = [str(output_path)]

    manifest = {
        "prompt": prompt["prompt"],
        "multiview_cameras": cameras,
        "frames_per_view": frames_per_view,
        "fps": fps,
        "files_by_camera": files_by_camera,
    }
    (args.output_dir / "sample_outputs.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
