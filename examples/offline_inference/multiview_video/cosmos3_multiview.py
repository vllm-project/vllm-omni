# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

r"""Cosmos3 Multiview-AV inference with an exported checkpoint and request JSON/JSONL.

Usage examples (run from the repository root):

    # Single GPU
    python examples/offline_inference/multiview_video/cosmos3_multiview.py \
        --model /models/cosmos3-multiview --input request.json

    # Four GPUs: CFGP2 x Ulysses CP2
    python examples/offline_inference/multiview_video/cosmos3_multiview.py \
        --model /models/cosmos3-multiview --input request.json \
        --cfg-parallel-size 2 --ulysses-degree 2

    # Two GPUs: HSDP weight sharding
    python examples/offline_inference/multiview_video/cosmos3_multiview.py \
        --model /models/cosmos3-multiview --input request.json \
        --use-hsdp --hsdp-shard-size 2

    # Two GPUs: tensor parallelism
    python examples/offline_inference/multiview_video/cosmos3_multiview.py \
        --model /models/cosmos3-multiview --input request.json \
        --tensor-parallel-size 2

CP uses strict Ulysses. HSDP and TP cannot be combined.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.diffusion.models.cosmos3.utils import VIDEO_RES_SIZE_INFO
from vllm_omni.diffusion.utils.video_encoding import run_ordered_encoding_jobs, write_imageio_video
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.model_extras.cosmos3 import normalize_multiview_aspect_ratio
from vllm_omni.model_extras.cosmos3_lidar import lidar_output_requested, serialize_lidar_output
from vllm_omni.outputs import OmniRequestOutput

SUPPORTED_MODEL_MODES = {"image2video", "text2video"}
SUPPORTED_RESOLUTIONS = {key: VIDEO_RES_SIZE_INFO[key] for key in ("480", "720")}


def _safe_camera_name(camera: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", camera).strip("_") or "camera"


def _safe_sample_name(name: str, sample_index: int) -> str:
    safe_name = _safe_camera_name(name)
    if safe_name in {".", ".."}:
        return f"sample_{sample_index:04d}"
    return safe_name


def _resolve_input_paths(request: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    # Match the HTTP client's path base for both direct and extra_params manifests.
    for container in (request, request.get("extra_params", {})):
        media = list(container.get("multiview", {}).get("views", []))
        if container.get("lidar") is not None:
            media.append(container["lidar"])
        for item in media:
            for field in ("vision_path", "control_path", "vision", "control"):
                value = item.get(field)
                if isinstance(value, str) and "://" not in value:
                    path = Path(value).expanduser()
                    item[field] = str((base_dir / path).resolve())
    return request


def _load_requests(input_path: Path) -> list[dict[str, Any]]:
    if input_path.suffix.lower() == ".jsonl":
        requests = []
        for line_number, line in enumerate(input_path.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            try:
                request = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {input_path} line {line_number}: {exc}") from exc
            if not isinstance(request, dict):
                raise TypeError(f"{input_path} line {line_number} must contain a JSON object.")
            requests.append(_resolve_input_paths(request, input_path.resolve().parent))
        if not requests:
            raise ValueError(f"Input JSONL file is empty: {input_path}")
        return requests

    request = json.loads(input_path.read_text())
    if not isinstance(request, dict):
        raise TypeError(f"Input JSON must contain one object, got {type(request).__name__}.")
    return [_resolve_input_paths(request, input_path.resolve().parent)]


def _resolve_model_mode(request: dict[str, Any], views: list[dict[str, Any]]) -> str:
    vision_present = [view.get("vision_path", view.get("vision")) is not None for view in views]
    inferred_mode = "image2video" if any(vision_present) else "text2video"
    model_mode = str(request.get("model_mode", inferred_mode)).strip().lower()
    if model_mode not in SUPPORTED_MODEL_MODES:
        raise ValueError(f"Unsupported model_mode {model_mode!r}; expected one of {sorted(SUPPORTED_MODEL_MODES)}.")
    if model_mode == "image2video" and not any(vision_present):
        raise ValueError("model_mode='image2video' requires at least one camera vision input.")
    if model_mode == "text2video" and any(vision_present):
        raise ValueError("model_mode='text2video' must not include per-camera vision inputs.")
    return model_mode


def _resolve_resolution(request: dict[str, Any], multiview: dict[str, Any], override: str | None = None) -> str | None:
    top_level = override if override is not None else request.get("resolution")
    nested = override if override is not None else multiview.get("resolution")
    if top_level is not None and nested is not None and str(top_level) != str(nested):
        raise ValueError(
            "Conflicting Cosmos3 multiview resolutions: "
            f"top-level resolution={top_level!r}, multiview.resolution={nested!r}."
        )
    if top_level is None and nested is None:
        return None
    resolution = str(nested if nested is not None else top_level)
    if resolution not in SUPPORTED_RESOLUTIONS:
        raise ValueError(
            f"Unsupported Cosmos3 multiview resolution {resolution!r}; expected one of {sorted(SUPPORTED_RESOLUTIONS)}."
        )
    return resolution


def _resolve_aspect_ratio(request: dict[str, Any], multiview: dict[str, Any], override: str | None = None) -> str:
    if override is not None:
        return normalize_multiview_aspect_ratio(override)
    declarations = {
        normalize_multiview_aspect_ratio(value)
        for value in (request.get("aspect_ratio"), multiview.get("aspect_ratio"))
        if value is not None
    }
    if len(declarations) > 1:
        raise ValueError(f"Conflicting Cosmos3 multiview aspect ratios: {sorted(declarations)}.")
    return next(iter(declarations), "auto")


def _resolve_seed(request: dict[str, Any], base_seed: int, sample_index: int) -> int:
    value = request.get("seed", base_seed + sample_index)
    if isinstance(value, bool):
        raise TypeError("seed must be an integer, not a boolean.")
    return int(value)


def _request_output_metadata(value: Any) -> dict[str, Any]:
    """Return the pipeline metadata the engine attaches to a request output.

    The formatter puts decoded frames in ``images`` and the pipeline's
    ``metadata`` envelope under ``multimodal_output["metadata"]``.
    """
    multimodal_output = getattr(value, "multimodal_output", None)
    if isinstance(multimodal_output, dict):
        metadata = multimodal_output.get("metadata")
    else:
        metadata = getattr(multimodal_output, "metadata", None)
    return metadata if isinstance(metadata, dict) else {}


def _extract_payload(value: Any) -> tuple[Any, dict[str, Any]]:
    if isinstance(value, list) and len(value) == 1:
        return _extract_payload(value[0])
    if isinstance(value, OmniRequestOutput) or (
        not isinstance(value, dict | list | np.ndarray | torch.Tensor) and hasattr(value, "images")
    ):
        if not value.images:
            raise ValueError("Cosmos3 multiview inference returned no video output.")
        video, metadata = _extract_payload(value.images[0] if len(value.images) == 1 else value.images)
        return video, {**_request_output_metadata(value), **metadata}
    if isinstance(value, dict):
        metadata = value.get("metadata") if isinstance(value.get("metadata"), dict) else {}
        payload = value.get("payload") if isinstance(value.get("payload"), dict) else value
        if "video" in payload:
            return payload["video"], metadata
    return value, {}


def _extract_lidar_payload(value: Any) -> torch.Tensor | None:
    if isinstance(value, list) and len(value) == 1:
        return _extract_lidar_payload(value[0])
    if hasattr(value, "multimodal_output"):
        return _extract_lidar_payload(value.multimodal_output)
    if isinstance(value, dict):
        return value.get("payload", value).get("lidar")
    return None


def _resolve_frames_per_view(frames: list[Any], cameras: list[str], metadata: dict[str, Any]) -> int:
    """Per-camera frame count of the camera-major output.

    The pipeline reports it in its metadata; without that, it is derived from
    the frame count. The requested ``num_frames`` is not a valid fallback: the
    pipeline rounds it up to the VAE grid.
    """
    reported = metadata.get("multiview", {}).get("frames_per_view")
    if reported is not None:
        frames_per_view = int(reported)
    elif cameras and len(frames) % len(cameras) == 0:
        frames_per_view = len(frames) // len(cameras)
    else:
        raise ValueError(f"Cannot split {len(frames)} camera-major frames evenly across {len(cameras)} cameras.")
    if frames_per_view <= 0 or len(frames) != len(cameras) * frames_per_view:
        raise ValueError(f"Expected {len(cameras) * frames_per_view} camera-major frames, got {len(frames)}.")
    return frames_per_view


def _frame_list(video: Any) -> list[Any]:
    if isinstance(video, torch.Tensor):
        tensor = video.detach().cpu()
        if tensor.ndim == 5:
            tensor = tensor[0]
        if tensor.ndim == 4 and tensor.shape[0] in (3, 4):
            tensor = tensor.permute(1, 2, 3, 0)
        # Keep normalized floats and uint8 storage as frame views. Clipping and
        # quantization happen one frame at a time in the writer.
        return list(tensor.numpy())
    if isinstance(video, np.ndarray):
        array = video[0] if video.ndim == 5 else video
        return list(array)
    if isinstance(video, list):
        if len(video) == 1 and isinstance(video[0], list):
            return video[0]
        if len(video) == 1 and isinstance(video[0], np.ndarray) and video[0].ndim == 4:
            return list(video[0])
        return video
    raise TypeError(f"Unsupported multiview video output type: {type(video).__name__}.")


def _export_combined_views(
    frames: list[Any], num_views: int, frames_per_view: int, path: Path, fps: float
) -> dict[str, Any]:
    """Stream synchronized camera-major frames into a grid, leaving unused tiles black."""
    import imageio.v2 as imageio

    columns = math.ceil(math.sqrt(num_views))
    rows = math.ceil(num_views / columns)
    height, width = np.asarray(frames[0]).shape[:2]
    normalize_negative = any(
        np.issubdtype(np.asarray(frame).dtype, np.floating) and np.asarray(frame).size and np.asarray(frame).min() < 0
        for frame in frames
    )
    with imageio.get_writer(str(path), fps=fps, macro_block_size=1, output_params=["-threads", "1"]) as writer:
        for frame_index in range(frames_per_view):
            grid = np.zeros((rows * height, columns * width, 3), dtype=np.uint8)
            for view_index in range(num_views):
                frame = np.asarray(frames[view_index * frames_per_view + frame_index])
                if np.issubdtype(frame.dtype, np.floating):
                    if normalize_negative:
                        frame = np.clip(frame, -1, 1) * 0.5 + 0.5
                    frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
                row, column = divmod(view_index, columns)
                grid[row * height : (row + 1) * height, column * width : (column + 1) * width] = frame[..., :3]
            writer.append_data(grid)
    return {"file_name": str(path), "rows": rows, "columns": columns, "width": columns * width, "height": rows * height}


@dataclass(frozen=True)
class _CameraEncodingJob:
    index: int
    camera: str
    frames: list[Any]
    destination: Path
    temporary: Path
    normalize_negative: bool


def _new_video_temporary_path(destination: Path) -> Path:
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{destination.stem}.", suffix=".tmp.mp4", dir=destination.parent, delete=False
    )
    handle.close()
    return Path(handle.name)


def _save_camera_videos(
    frames: list[Any],
    cameras: list[str],
    frames_per_view: int,
    output_dir: Path,
    fps: float,
    *,
    parallel: bool,
) -> dict[str, list[str]]:
    """Encode camera views transactionally and preserve camera order."""
    jobs: list[_CameraEncodingJob] = []
    try:
        normalize_negative = any(
            np.issubdtype(np.asarray(frame).dtype, np.floating)
            and np.asarray(frame).size
            and np.asarray(frame).min() < 0
            for frame in frames
        )
        for index, camera in enumerate(cameras):
            destination = output_dir / f"vision_view{index:02d}_{_safe_camera_name(camera)}.mp4"
            jobs.append(
                _CameraEncodingJob(
                    index=index,
                    camera=camera,
                    frames=frames[index * frames_per_view : (index + 1) * frames_per_view],
                    destination=destination,
                    temporary=_new_video_temporary_path(destination),
                    normalize_negative=normalize_negative,
                )
            )

        def encode(job: _CameraEncodingJob, encoder_threads: int) -> _CameraEncodingJob:
            write_imageio_video(
                job.frames,
                job.temporary,
                fps=fps,
                encoder_threads=encoder_threads,
                # Match Diffusers export_to_video's dimension handling.
                macro_block_size=16,
                normalize_negative=job.normalize_negative,
            )
            return job

        completed, _ = run_ordered_encoding_jobs(jobs, encode, parallel=parallel)
        for job in completed:
            os.replace(job.temporary, job.destination)
        return {job.camera: [str(job.destination)] for job in completed}
    except BaseException:
        for job in jobs:
            try:
                job.temporary.unlink(missing_ok=True)
            except OSError:
                pass
        raise


def _run_request(
    omni: Omni,
    request: dict[str, Any],
    *,
    output_dir: Path,
    seed: int,
    fallback_negative_prompt: str | None,
    fps_override: float | None = None,
    num_frames_override: int | None = None,
    resolution_override: str | None = None,
    aspect_ratio_override: str | None = None,
    combine_views: bool = False,
    video_encoding_mode: str = "parallel",
) -> dict[str, Any]:
    request = {**request.get("extra_params", {}), **request}
    multiview_value = request.get("multiview")
    if not isinstance(multiview_value, dict):
        raise ValueError("Input JSON must contain a multiview object.")
    multiview = dict(multiview_value)
    views_value = multiview.get("views")
    if not isinstance(views_value, list) or not views_value:
        raise ValueError("Input JSON must contain multiview.views.")
    if not all(isinstance(view, dict) for view in views_value):
        raise TypeError("Every multiview.views entry must be an object.")
    views: list[dict[str, Any]] = views_value

    model_mode = _resolve_model_mode(request, views)
    resolution = _resolve_resolution(request, multiview, resolution_override)
    aspect_ratio = _resolve_aspect_ratio(request, multiview, aspect_ratio_override)
    geometry_override = resolution_override is not None or aspect_ratio_override is not None
    width = None if geometry_override else request.get("width")
    height = None if geometry_override else request.get("height")
    if aspect_ratio != "auto" and resolution is not None:
        target_width, target_height = SUPPORTED_RESOLUTIONS[resolution][aspect_ratio]
        for key, value, expected in (("width", width, target_width), ("height", height, target_height)):
            if value is not None and int(value) != expected:
                raise ValueError(
                    f"Cosmos3 multiview resolution={resolution!r} requires {key}={expected}, got {value} "
                    f"for aspect_ratio={aspect_ratio!r}."
                )
        width, height = target_width, target_height
    # Keep the resolved value with the variant-owned multiview parameters so
    # top-level Imaginaire inputs and native vLLM-Omni inputs behave identically.
    if resolution is not None:
        multiview["resolution"] = resolution
    multiview["aspect_ratio"] = aspect_ratio

    # Frame rate and per-camera frame count are pipeline-owned: when neither the
    # CLI nor the record sets them, the pipeline applies its defaults (checkpoint FPS,
    # 201 frames) and rounds frame counts up to the VAE's 4k+1 grid. CLI
    # overrides win over record values.
    num_frames = num_frames_override
    if num_frames is None:
        num_frames = _first_present(multiview, "num_frames")
    if num_frames is None:
        num_frames = _first_present(request, "num_frames")
    fps = fps_override if fps_override is not None else _first_present(request, "fps")

    extra_args = {
        "multiview": multiview,
        "aspect_ratio": aspect_ratio,
    }
    if resolution is not None:
        extra_args["resolution"] = resolution
    for key in (
        "wsm",
        "edge",
        "blur",
        "depth",
        "seg",
        "lidar",
        "emphasize_control_in_prompt",
        "guidance_interval",
        "control_guidance",
        "control_guidance_interval",
        "sigma_max",
        "normalize_cfg",
    ):
        if key in request:
            extra_args[key] = request[key]
    # Records may also use the field names ``guidance``, ``num_steps``, and
    # ``shift``; the vLLM-Omni names win when both are present.
    flow_shift = _first_present(request, "flow_shift", "shift")
    if flow_shift is not None:
        extra_args["flow_shift"] = float(flow_shift)
    sampling_kwargs: dict[str, Any] = {}
    if num_frames is not None:
        num_frames = int(num_frames)
        # Keep the record's variant-owned copy in step with the resolved value.
        multiview["num_frames"] = num_frames
        sampling_kwargs["num_frames"] = num_frames
    if fps is not None:
        sampling_kwargs["fps"] = float(fps)
    sampling_params = OmniDiffusionSamplingParams(
        height=height,
        width=width,
        num_inference_steps=int(_first_present(request, "num_inference_steps", "num_steps", default=35)),
        guidance_scale=float(_first_present(request, "guidance_scale", "guidance", default=6.0)),
        seed=seed,
        extra_args=extra_args,
        **sampling_kwargs,
    )
    prompt = {
        "prompt": str(request.get("prompt", "")),
        "modalities": ["video"],
    }
    if request.get("negative_prompt") is not None:
        prompt["negative_prompt"] = request["negative_prompt"]
    elif fallback_negative_prompt is not None:
        prompt["negative_prompt"] = fallback_negative_prompt

    started = time.perf_counter()
    result = omni.generate(prompt, sampling_params)
    generation_seconds = time.perf_counter() - started
    video, metadata = _extract_payload(result)
    lidar = _extract_lidar_payload(result)
    if lidar_output_requested(extra_args) and lidar is None:
        raise ValueError("The requested LiDAR output was not returned by the model.")
    frames = _frame_list(video)
    if not frames:
        raise ValueError("Cosmos3 multiview output contains no video frames.")
    output_geometry = metadata.get("multiview", {})
    output_height, output_width = np.asarray(frames[0]).shape[:2]
    output_resolution = output_geometry.get("resolution", resolution or "480")
    output_aspect_ratio = output_geometry.get("aspect_ratio")
    if output_aspect_ratio is None:
        output_aspect_ratio = next(
            (
                ratio
                for ratio, size in SUPPORTED_RESOLUTIONS[output_resolution].items()
                if size == (output_width, output_height)
            ),
            None,
        )
    if output_aspect_ratio is None or output_aspect_ratio == "auto":
        raise ValueError("Cosmos3 multiview output is missing a resolved aspect ratio and canonical dimensions.")

    cameras = metadata.get("multiview", {}).get("cameras") or [view["camera_key"] for view in views]
    frames_per_view = _resolve_frames_per_view(frames, cameras, metadata)
    output_fps = float(metadata.get("multiview", {}).get("fps", sampling_params.fps or 30))

    output_dir.mkdir(parents=True, exist_ok=True)
    video_save_started = time.perf_counter()
    files_by_camera = _save_camera_videos(
        frames,
        list(cameras),
        frames_per_view,
        output_dir,
        output_fps,
        parallel=video_encoding_mode == "parallel",
    )

    manifest = {
        "name": request.get("name"),
        "model_mode": model_mode,
        "resolution": output_resolution,
        "aspect_ratio": output_aspect_ratio,
        "width": output_width,
        "height": output_height,
        "seed": seed,
        "prompt": prompt["prompt"],
        "multiview_cameras": cameras,
        "frames_per_view": frames_per_view,
        "fps": output_fps,
        "files_by_camera": files_by_camera,
        "generation_seconds": generation_seconds,
    }
    if combine_views:
        combined_path = output_dir / "combined_views.mp4"
        combined_temporary = _new_video_temporary_path(combined_path)
        try:
            manifest["combined_video"] = _export_combined_views(
                frames, len(cameras), frames_per_view, combined_temporary, output_fps
            )
            os.replace(combined_temporary, combined_path)
            manifest["combined_video"]["file_name"] = str(combined_path)
        except BaseException:
            combined_temporary.unlink(missing_ok=True)
            raise
    manifest["video_save_seconds"] = time.perf_counter() - video_save_started
    if lidar is not None:
        data, details = serialize_lidar_output(lidar, metadata.get("lidar", {}))
        lidar_path = output_dir / "lidar.safetensors"
        lidar_path.write_bytes(data)
        manifest["lidar"] = {**details, "file_name": str(lidar_path), "format": "safetensors"}
    # This timestamp intentionally precedes the manifest write itself.
    manifest["total_seconds"] = time.perf_counter() - started
    (output_dir / "sample_outputs.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def _first_present(record: dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Return the first non-null value among ``keys``, in priority order."""
    for key in keys:
        value = record.get(key)
        if value is not None:
            return value
    return default


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Exported Cosmos3 Multiview-AV Diffusers directory")
    parser.add_argument("--input", required=True, type=Path, help="Multiview JSON or JSONL file")
    parser.add_argument("--output-dir", type=Path, default=Path("cosmos3_multiview_output"))
    parser.add_argument(
        "--combine-views",
        action="store_true",
        help="Also save combined_views.mp4 as a synchronized grid in camera order, with unused tiles black",
    )
    parser.add_argument(
        "--video-encoding-mode",
        choices=("parallel", "serial"),
        default="parallel",
        help="Encode camera videos concurrently (default) or one camera at a time for diagnostics",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base seed; JSONL records use seed + record index unless the record provides seed",
    )
    parser.add_argument(
        "--negative-prompt-json",
        type=Path,
        help=(
            "Structured negative prompt to serialize with json.dumps defaults. The pipeline ships no default "
            "negative prompt, so reference-parity runs must supply the reference one here."
        ),
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help=(
            "Frame rate for every record, overriding any record value. "
            "Unset: the record's fps, else the checkpoint default (30 FPS for unversioned artifacts)."
        ),
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=None,
        help=(
            "Per-camera frame count for every record, overriding any record value. "
            "The pipeline rounds it up to the VAE's 4k+1 grid; unset defaults to 201."
        ),
    )
    parser.add_argument(
        "--resolution",
        choices=tuple(SUPPORTED_RESOLUTIONS),
        help=(
            "Video resolution bucket (480 or 720) for every record, overriding any record value. "
            "Unset: the record's resolution, else the checkpoint default (480 for unversioned artifacts)."
        ),
    )
    parser.add_argument(
        "--aspect-ratio",
        type=normalize_multiview_aspect_ratio,
        help="auto, 1:1, 4:3, 3:4, 16:9, or 9:16; overrides every record (default: detect from first WSM input)",
    )
    parser.add_argument("--cfg-parallel-size", type=int, choices=(1, 2), default=1)
    parser.add_argument("--ulysses-degree", type=int, default=1)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--use-hsdp", action="store_true")
    parser.add_argument("--hsdp-shard-size", type=int, default=-1)
    parser.add_argument("--hsdp-replicate-size", type=int, default=1)
    parser.add_argument("--enforce-eager", action="store_true", help="Disable regional transformer compilation")
    args = parser.parse_args()

    requests = _load_requests(args.input)
    fallback_negative_prompt = None
    if args.negative_prompt_json is not None:
        # Default separators (", " and ": ") and the file's key order are part
        # of the reference's serialization, so keep json.dumps unconfigured.
        fallback_negative_prompt = json.dumps(json.loads(args.negative_prompt_json.read_text()))

    parallel_config = DiffusionParallelConfig(
        ulysses_degree=args.ulysses_degree,
        cfg_parallel_size=args.cfg_parallel_size,
        tensor_parallel_size=args.tensor_parallel_size,
        use_hsdp=args.use_hsdp,
        hsdp_shard_size=args.hsdp_shard_size,
        hsdp_replicate_size=args.hsdp_replicate_size,
    )
    omni = Omni(
        model=args.model,
        dtype="bfloat16",
        model_class_name="Cosmos3MultiviewPipeline",
        enforce_eager=args.enforce_eager,
        parallel_config=parallel_config,
        diffusion_compile_granularity="regional",
        diffusion_compile_dynamic=False,
    )

    sample_names = [
        _safe_sample_name(str(request.get("name") or f"sample_{sample_index:04d}"), sample_index)
        for sample_index, request in enumerate(requests)
    ]
    if len(sample_names) != len(set(sample_names)):
        duplicates = sorted({name for name in sample_names if sample_names.count(name) > 1})
        raise ValueError(f"Duplicate output sample names after sanitization: {duplicates}.")

    is_batch = len(requests) > 1
    manifests = []
    for sample_index, (request, sample_name) in enumerate(zip(requests, sample_names, strict=True)):
        output_dir = args.output_dir / sample_name if is_batch else args.output_dir
        seed = _resolve_seed(request, args.seed, sample_index)
        print(f"[{sample_index + 1}/{len(requests)}] Generating {sample_name!r} with seed {seed}...")
        manifest = _run_request(
            omni,
            request,
            output_dir=output_dir,
            seed=seed,
            fallback_negative_prompt=fallback_negative_prompt,
            fps_override=args.fps,
            num_frames_override=args.num_frames,
            resolution_override=args.resolution,
            aspect_ratio_override=args.aspect_ratio,
            combine_views=args.combine_views,
            video_encoding_mode=args.video_encoding_mode,
        )
        manifests.append({**manifest, "output_dir": str(output_dir)})

    if is_batch:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "sample_outputs.jsonl").write_text(
            "".join(json.dumps(manifest) + "\n" for manifest in manifests)
        )


if __name__ == "__main__":
    main()
