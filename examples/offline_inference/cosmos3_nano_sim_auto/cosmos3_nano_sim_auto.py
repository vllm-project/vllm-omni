# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
r"""Cosmos3-Nano-Sim-Auto offline rollouts from prepared request JSON/JSONL.

Prepare inputs inside imaginaire4 first (caption discovery, HD-map conversion):

    PYTHONPATH="$PWD/packages/cosmos3:$PWD" python -m cosmos3.scripts.prepare_multiview_lidar \
        packages/cosmos3/inputs/omni_multiview/wsm_transfer_i2v.jsonl --output /data/sim_auto_rgb \
        --rolling --num-views 7 --num-frames 297 --fps 30

then run every request as one causal rollout (RGB-only or joint RGB+LiDAR):

    python examples/offline_inference/cosmos3_nano_sim_auto/cosmos3_nano_sim_auto.py \
        --model /models/Cosmos3-Nano-Sim-Auto --input /data/sim_auto_rgb/requests.jsonl \
        --output-dir outputs/sim_auto --tensor-parallel-size 4

Each sample writes one MP4 per camera (``vision_view{NN}_{camera}.mp4``), the
numeric LiDAR output as ``lidar.safetensors`` when requested, and
``sample_outputs.json``. The student runs its exported four-step schedule with
guidance 1; request fields that contradict it are rejected by the pipeline.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

PASSTHROUGH_EXTRAS = ("wsm", "lidar", "aspect_ratio", "resolution")


def _safe_name(value: str, fallback: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return fallback if safe in {"", ".", ".."} else safe


def _resolve_input_paths(request: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    for container in (request, request.get("extra_params", {})):
        media = list(container.get("multiview", {}).get("views", []))
        if container.get("lidar") is not None:
            media.append(container["lidar"])
        for item in media:
            for field in ("vision_path", "control_path", "condition_path", "vision", "control"):
                value = item.get(field)
                if isinstance(value, str) and "://" not in value:
                    path = Path(value).expanduser()
                    item[field] = str((base_dir / path).resolve())
    return request


def _load_requests(input_path: Path) -> list[dict[str, Any]]:
    base_dir = input_path.resolve().parent
    if input_path.suffix.lower() == ".jsonl":
        requests = [json.loads(line) for line in input_path.read_text().splitlines() if line.strip()]
    else:
        requests = [json.loads(input_path.read_text())]
    if not requests or any(not isinstance(request, dict) for request in requests):
        raise ValueError(f"{input_path} must hold one JSON object per request.")
    return [_resolve_input_paths(request, base_dir) for request in requests]


def _request_metadata(value: Any) -> dict[str, Any]:
    multimodal_output = getattr(value, "multimodal_output", None)
    metadata = (
        multimodal_output.get("metadata")
        if isinstance(multimodal_output, dict)
        else getattr(multimodal_output, "metadata", None)
    )
    return metadata if isinstance(metadata, dict) else {}


def _extract_payload(value: Any) -> tuple[Any, dict[str, Any]]:
    if isinstance(value, list) and len(value) == 1:
        return _extract_payload(value[0])
    if not isinstance(value, dict | list | np.ndarray | torch.Tensor) and hasattr(value, "images"):
        if not value.images:
            raise ValueError("Cosmos3-Nano-Sim-Auto returned no video output.")
        video, metadata = _extract_payload(value.images[0] if len(value.images) == 1 else value.images)
        return video, {**_request_metadata(value), **metadata}
    if isinstance(value, dict):
        metadata = value.get("metadata") if isinstance(value.get("metadata"), dict) else {}
        payload = value.get("payload") if isinstance(value.get("payload"), dict) else value
        if "video" in payload:
            return payload["video"], metadata
    return value, {}


def _extract_lidar(value: Any) -> torch.Tensor | None:
    if isinstance(value, list) and len(value) == 1:
        return _extract_lidar(value[0])
    if hasattr(value, "multimodal_output"):
        return _extract_lidar(value.multimodal_output)
    if isinstance(value, dict):
        return value.get("payload", value).get("lidar")
    return None


def _extract_latents(value: Any) -> dict[str, torch.Tensor] | None:
    if isinstance(value, list) and len(value) == 1:
        return _extract_latents(value[0])
    if hasattr(value, "multimodal_output"):
        return _extract_latents(value.multimodal_output)
    if isinstance(value, dict):
        return value.get("payload", value).get("latents")
    return None


def _save_latents(value: Any, output_dir: Path, *, joint: bool) -> Path:
    latents = _extract_latents(value)
    required = {"vision_latent", "lidar_latent"} if joint else {"vision_latent"}
    if not isinstance(latents, dict) or not required.issubset(latents):
        raise ValueError(f"The model did not return the requested latents: {sorted(required)}.")
    if any(not isinstance(tensor, torch.Tensor) or tensor.ndim != 5 for tensor in latents.values()):
        raise ValueError("Expected latent tensors with shape [1,C,T,H,W].")
    path = output_dir / "latents.pt"
    # Match imaginaire4's one-element list for a full rolling rollout.
    torch.save([{key: tensor.detach().cpu().contiguous() for key, tensor in latents.items()}], path)
    return path


def _frame_list(video: Any) -> list[Any]:
    if isinstance(video, torch.Tensor):
        tensor = video.detach().cpu()
        if tensor.ndim == 5:
            tensor = tensor[0]
        if tensor.ndim == 4 and tensor.shape[0] in (3, 4):
            tensor = tensor.permute(1, 2, 3, 0)
        return list(tensor.numpy())
    if isinstance(video, np.ndarray):
        return list(video[0] if video.ndim == 5 else video)
    if isinstance(video, list):
        if len(video) == 1 and isinstance(video[0], list):
            return video[0]
        if len(video) == 1 and isinstance(video[0], np.ndarray) and video[0].ndim == 4:
            return list(video[0])
        return video
    raise TypeError(f"Unsupported video output type: {type(video).__name__}.")


@dataclass(frozen=True)
class _CameraJob:
    camera: str
    frames: list[Any]
    destination: Path
    temporary: Path
    normalize_negative: bool


def _save_camera_videos(
    frames: list[Any], cameras: list[str], frames_per_view: int, output_dir: Path, fps: float
) -> dict[str, str]:
    import imageio.v2 as imageio

    if not cameras or frames_per_view <= 0:
        raise ValueError("Camera video output requires cameras and a positive frames_per_view.")
    if len(frames) != len(cameras) * frames_per_view:
        raise ValueError(f"Expected {len(cameras) * frames_per_view} camera-major frames, got {len(frames)}.")
    normalize_negative = any(
        np.issubdtype(np.asarray(frame).dtype, np.floating) and np.asarray(frame).size and np.asarray(frame).min() < 0
        for frame in frames
    )
    jobs: list[_CameraJob] = []
    try:
        for index, camera in enumerate(cameras):
            destination = output_dir / f"vision_view{index:02d}_{_safe_name(camera, 'camera')}.mp4"
            handle = tempfile.NamedTemporaryFile(
                prefix=f".{destination.stem}.", suffix=".tmp.mp4", dir=destination.parent, delete=False
            )
            handle.close()
            jobs.append(
                _CameraJob(
                    camera,
                    frames[index * frames_per_view : (index + 1) * frames_per_view],
                    destination,
                    Path(handle.name),
                    normalize_negative,
                )
            )

        def encode(job: _CameraJob) -> _CameraJob:
            with imageio.get_writer(
                str(job.temporary),
                fps=fps,
                quality=5.0,
                macro_block_size=16,
                output_params=["-threads", "1"],
            ) as writer:
                for source in job.frames:
                    frame = np.asarray(source)
                    if np.issubdtype(frame.dtype, np.floating):
                        if job.normalize_negative:
                            frame = np.clip(frame, -1.0, 1.0) * 0.5 + 0.5
                        frame = (np.clip(frame, 0.0, 1.0) * 255.0).astype(np.uint8)
                    elif frame.dtype != np.uint8:
                        frame = frame.astype(np.uint8)
                    writer.append_data(frame[..., :3])
            return job

        # Bound independent camera encodes and keep one codec thread per worker.
        # Join all writers before publishing files or cleaning up after a failure.
        with ThreadPoolExecutor(max_workers=min(len(jobs), os.cpu_count() or 1)) as executor:
            completed = list(executor.map(encode, jobs))
        for job in completed:
            os.replace(job.temporary, job.destination)
        return {job.camera: str(job.destination) for job in completed}
    except BaseException:
        for job in jobs:
            job.temporary.unlink(missing_ok=True)
        raise


def _sampling_params(
    request: dict[str, Any], *, seed: int, fps_override: float | None, num_frames_override: int | None
) -> OmniDiffusionSamplingParams:
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    multiview = dict(request.get("multiview") or request.get("extra_params", {}).get("multiview") or {})
    if not multiview.get("views"):
        raise ValueError("Each request needs multiview.views with per-camera prompt and control_path.")
    extra_args: dict[str, Any] = {"multiview": multiview}
    for key in PASSTHROUGH_EXTRAS:
        if key in request:
            extra_args[key] = request[key]
    if "resolution" not in extra_args and multiview.get("resolution") is not None:
        extra_args["resolution"] = multiview["resolution"]
    extra_args.update(session_id=f"offline-{seed}", reset=True, close_session=True)
    num_frames = (
        num_frames_override
        if num_frames_override is not None
        else multiview.get("num_frames", request.get("num_frames"))
    )
    fps = fps_override if fps_override is not None else request.get("fps")
    kwargs: dict[str, Any] = {}
    if num_frames is not None:
        multiview["num_frames"] = int(num_frames)
        kwargs["num_frames"] = int(num_frames)
    if fps is not None:
        kwargs["fps"] = float(fps)
    return OmniDiffusionSamplingParams(
        seed=seed,
        guidance_scale=1.0,
        num_inference_steps=None,
        extra_args=extra_args,
        generator=torch.Generator(device="cpu").manual_seed(seed),
        **kwargs,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Converted Cosmos3-Nano-Sim-Auto checkpoint directory.")
    parser.add_argument("--input", type=Path, required=True, help="Prepared request JSON or JSONL.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/cosmos3_nano_sim_auto"))
    parser.add_argument(
        "--deploy-config", default=None, help="Deploy YAML (defaults to vllm_omni/deploy/cosmos3_nano_sim_auto.yaml)."
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0, help="Base seed; request seeds win when present.")
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--save-latents",
        action="store_true",
        help="Save RGB and optional LiDAR latents in reference-compatible latents.pt.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    # Keep output helpers and --help usable without initializing the inference runtime.
    from vllm_omni.diffusion.data import DiffusionParallelConfig
    from vllm_omni.entrypoints.omni import Omni
    from vllm_omni.model_extras.cosmos3_lidar import lidar_output_requested, serialize_lidar_output

    requests = _load_requests(args.input)[: args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    omni = Omni(
        model=args.model,
        model_class_name="Cosmos3NanoSimAutoPipeline",
        deploy_config=args.deploy_config,
        enforce_eager=True,
        parallel_config=DiffusionParallelConfig(tensor_parallel_size=args.tensor_parallel_size),
    )
    for index, request in enumerate(requests):
        seed = int(request.get("seed", args.seed + index))
        name = _safe_name(str(request.get("name", f"sample_{index:04d}")), f"sample_{index:04d}")
        sample_dir = args.output_dir / name / "0"
        sample_dir.mkdir(parents=True, exist_ok=True)
        sampling_params = _sampling_params(
            request, seed=seed, fps_override=args.fps, num_frames_override=args.num_frames
        )
        if args.save_latents:
            sampling_params.extra_args["return_latents"] = True
        prompt = {"prompt": str(request.get("prompt", "")), "modalities": ["video"]}
        started = time.perf_counter()
        result = omni.generate(prompt, sampling_params)
        generation_seconds = time.perf_counter() - started
        video, metadata = _extract_payload(result)
        lidar = _extract_lidar(result)
        if lidar_output_requested(sampling_params.extra_args) and lidar is None:
            raise ValueError("The requested LiDAR output was not returned by the model.")
        frames = _frame_list(video)
        multiview_meta = metadata.get("multiview", {})
        cameras = list(
            multiview_meta.get("cameras")
            or [view["camera_key"] for view in sampling_params.extra_args["multiview"]["views"]]
        )
        frames_per_view = int(multiview_meta.get("frames_per_view") or len(frames) // len(cameras))
        fps = float(multiview_meta.get("fps") or sampling_params.fps or 30.0)
        files = _save_camera_videos(frames, cameras, frames_per_view, sample_dir, fps)
        outputs: dict[str, Any] = {
            "name": name,
            "seed": seed,
            "cameras": cameras,
            "frames_per_view": frames_per_view,
            "fps": fps,
            "files": files,
            "generation_seconds": generation_seconds,
            "metadata": metadata,
        }
        if args.save_latents:
            outputs["latents_file"] = str(
                _save_latents(result, sample_dir, joint=sampling_params.extra_args.get("lidar") is not None)
            )
        if lidar is not None:
            payload, details = serialize_lidar_output(lidar, metadata.get("lidar", {"fps": 10.0}))
            lidar_path = sample_dir / "lidar.safetensors"
            lidar_path.write_bytes(payload)
            outputs["lidar_file"] = str(lidar_path)
            outputs["lidar"] = details
        (sample_dir / "sample_outputs.json").write_text(json.dumps(outputs, indent=2, default=str) + "\n")
        print(
            json.dumps(
                {
                    "sample": name,
                    "seed": seed,
                    "cameras": len(cameras),
                    "frames_per_view": frames_per_view,
                    "seconds": round(generation_seconds, 2),
                }
            )
        )


if __name__ == "__main__":
    main()
