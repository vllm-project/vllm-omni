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
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from diffusers.utils import export_to_video

from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput

SUPPORTED_MODEL_MODES = {"image2video", "text2video"}
SUPPORTED_RESOLUTIONS = {"480": (832, 480)}


def _safe_camera_name(camera: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", camera).strip("_") or "camera"


def _safe_sample_name(name: str, sample_index: int) -> str:
    safe_name = _safe_camera_name(name)
    if safe_name in {".", ".."}:
        return f"sample_{sample_index:04d}"
    return safe_name


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
            requests.append(request)
        if not requests:
            raise ValueError(f"Input JSONL file is empty: {input_path}")
        return requests

    request = json.loads(input_path.read_text())
    if not isinstance(request, dict):
        raise TypeError(f"Input JSON must contain one object, got {type(request).__name__}.")
    return [request]


def _resolve_model_mode(request: dict[str, Any], views: list[dict[str, Any]]) -> str:
    vision_present = [view.get("vision_path", view.get("vision")) is not None for view in views]
    inferred_mode = "image2video" if any(vision_present) else "text2video"
    model_mode = str(request.get("model_mode", inferred_mode)).strip().lower()
    if model_mode not in SUPPORTED_MODEL_MODES:
        raise ValueError(f"Unsupported model_mode {model_mode!r}; expected one of {sorted(SUPPORTED_MODEL_MODES)}.")
    if model_mode == "image2video" and not all(vision_present):
        raise ValueError("model_mode='image2video' requires vision input for every camera view.")
    if model_mode == "text2video" and any(vision_present):
        raise ValueError("model_mode='text2video' must not include per-camera vision inputs.")
    return model_mode


def _resolve_resolution(request: dict[str, Any], multiview: dict[str, Any]) -> tuple[str, int, int]:
    top_level = request.get("resolution")
    nested = multiview.get("resolution")
    if top_level is not None and nested is not None and str(top_level) != str(nested):
        raise ValueError(
            "Conflicting Cosmos3 multiview resolutions: "
            f"top-level resolution={top_level!r}, multiview.resolution={nested!r}."
        )
    resolution = str(nested if nested is not None else top_level if top_level is not None else "480")
    if resolution not in SUPPORTED_RESOLUTIONS:
        raise ValueError(
            f"Unsupported Cosmos3 multiview resolution {resolution!r}; expected one of {sorted(SUPPORTED_RESOLUTIONS)}."
        )
    width, height = SUPPORTED_RESOLUTIONS[resolution]
    return resolution, width, height


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


def _run_request(
    omni: Omni,
    request: dict[str, Any],
    *,
    output_dir: Path,
    seed: int,
    fallback_negative_prompt: str | None,
    fps_override: float | None = None,
    num_frames_override: int | None = None,
) -> dict[str, Any]:
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
    resolution, width, height = _resolve_resolution(request, multiview)
    # Keep the resolved value with the variant-owned multiview parameters so
    # top-level Imaginaire inputs and native vLLM-Omni inputs behave identically.
    multiview["resolution"] = resolution

    # Frame rate and per-camera frame count are pipeline-owned: when neither the
    # CLI nor the record sets them, the pipeline applies its defaults (30 FPS,
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
        "resolution": resolution,
        "wsm": request.get("wsm", {}),
    }
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

    result = omni.generate(prompt, sampling_params)
    video, metadata = _extract_payload(result)
    frames = _frame_list(video)

    cameras = metadata.get("multiview", {}).get("cameras") or [view["camera_key"] for view in views]
    frames_per_view = _resolve_frames_per_view(frames, cameras, metadata)
    output_fps = float(metadata.get("multiview", {}).get("fps", sampling_params.fps or 30))

    output_dir.mkdir(parents=True, exist_ok=True)
    files_by_camera = {}
    for index, camera in enumerate(cameras):
        camera_frames = frames[index * frames_per_view : (index + 1) * frames_per_view]
        output_path = output_dir / f"vision_view{index:02d}_{_safe_camera_name(camera)}.mp4"
        export_to_video(camera_frames, str(output_path), fps=output_fps)
        files_by_camera[camera] = [str(output_path)]

    manifest = {
        "name": request.get("name"),
        "model_mode": model_mode,
        "resolution": resolution,
        "seed": seed,
        "prompt": prompt["prompt"],
        "multiview_cameras": cameras,
        "frames_per_view": frames_per_view,
        "fps": output_fps,
        "files_by_camera": files_by_camera,
    }
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
            "Unset: the record's fps, else the pipeline default of 30 FPS."
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
        )
        manifests.append({**manifest, "output_dir": str(output_dir)})

    if is_batch:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "sample_outputs.jsonl").write_text(
            "".join(json.dumps(manifest) + "\n" for manifest in manifests)
        )


if __name__ == "__main__":
    main()
