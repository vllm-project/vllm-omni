#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Sequence

import numpy as np

try:
    from examples.offline_inference.dreamzero import benchmark_utils
except ImportError:  # pragma: no cover - direct script execution fallback
    import benchmark_utils  # type: ignore[no-redef]

WORKER_EXTENSION = "vllm_omni.diffusion.models.dreamzero.video_export_worker.DreamZeroVideoExportWorkerExtension"
ASSET_REPO_ID = "YangshenDeng/vllm-omni-dreamzero-assets"
DEFAULT_SESSION_PREFIX = "dreamzero-benchmark"
DEFAULT_MODEL = "GEAR-Dreams/DreamZero-DROID"
DEFAULT_PROMPT = "Move the pan forward and use the brush in the middle of the plates to brush the inside of the pan"
RELATIVE_OFFSETS = [-23, -16, -8, 0]
ACTION_HORIZON = benchmark_utils.ACTION_HORIZON
CAMERA_FILES = {
    "observation/exterior_image_0_left": "exterior_image_1_left.mp4",
    "observation/exterior_image_1_left": "exterior_image_2_left.mp4",
    "observation/wrist_image_left": "wrist_image_left.mp4",
}
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_VIDEO_DIR = REPO_ROOT / "outputs" / "dreamzero" / "assets"
DEFAULT_OUTPUT_DIR = Path("outputs") / "dreamzero" / "benchmark"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark DreamZero offline generation latency while exporting "
            "prediction video and action artifacts."
        )
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--deploy-config", type=Path, required=True)
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=DEFAULT_VIDEO_DIR,
        help="Directory containing the three camera MP4 files.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-stem", default="dreamzero_benchmark")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--session-id", default=None)
    parser.add_argument(
        "--num-requests",
        type=_positive_int,
        default=2,
        help="Total action-producing generate calls, including the initial single-frame request.",
    )
    parser.add_argument("--fps", type=_positive_int, default=5, help="Playback FPS for written videos.")
    parser.add_argument(
        "--profiler-config",
        type=parse_profiler_config,
        default=None,
        help='JSON profiler config, e.g. \'{"profiler":"torch","torch_profiler_dir":"./perf"}\'.',
    )
    parser.add_argument(
        "--profile-request-index",
        type=_non_negative_int,
        default=1,
        help="Generate request index to profile when --profiler-config is set. Default profiles chunk_0.",
    )
    parser.add_argument("--save-gif", action="store_true")
    parser.add_argument("--save-input-video", action="store_true")
    parser.add_argument("--save-side-by-side", action="store_true")
    parser.add_argument(
        "--reference-actions",
        type=Path,
        default=None,
        help="Optional NPZ produced by this script. Compares stepN actions against the current run.",
    )
    parser.add_argument(
        "--accuracy-atol",
        type=float,
        default=1e-3,
        help="Max absolute action error tolerated when --reference-actions is provided.",
    )
    return parser.parse_args(argv)


def parse_profiler_config(value: str) -> dict[str, Any]:
    try:
        config = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"--profiler-config must be valid JSON: {exc}") from exc
    if not isinstance(config, dict):
        raise argparse.ArgumentTypeError("--profiler-config must be a JSON object")
    return config


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be a non-negative integer")
    return parsed


def _load_all_frames(video_path: Path) -> np.ndarray:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames loaded from {video_path}")
    return np.stack(frames, axis=0)


def _load_camera_frames(video_dir: Path) -> dict[str, np.ndarray]:
    camera_frames: dict[str, np.ndarray] = {}
    for camera_key, file_name in CAMERA_FILES.items():
        video_path = video_dir / file_name
        if not video_path.exists():
            raise FileNotFoundError(
                f"Missing DreamZero example asset: {video_path}. "
                "Download the example videos with: "
                f"`hf download {ASSET_REPO_ID} --repo-type dataset --local-dir {video_dir}`"
            )
        camera_frames[camera_key] = _load_all_frames(video_path)
    return camera_frames


def _build_frame_schedule(total_frames: int, num_chunks: int) -> list[list[int]]:
    chunks: list[list[int]] = []
    current_frame = 23
    for _ in range(num_chunks):
        indices = [max(current_frame + offset, 0) for offset in RELATIVE_OFFSETS]
        if indices[-1] >= total_frames:
            break
        chunks.append(indices)
        current_frame += ACTION_HORIZON
    return chunks


def _make_obs_from_video(
    camera_frames: dict[str, np.ndarray],
    frame_indices: list[int],
    *,
    prompt: str,
    session_id: str,
) -> dict:
    obs: dict = {}
    for camera_key, all_frames in camera_frames.items():
        selected = all_frames[frame_indices]
        obs[camera_key] = selected[0] if len(frame_indices) == 1 else selected

    obs["observation/joint_position"] = np.zeros(7, dtype=np.float32)
    obs["observation/cartesian_position"] = np.zeros(6, dtype=np.float32)
    obs["observation/gripper_position"] = np.zeros(1, dtype=np.float32)
    obs["prompt"] = prompt
    obs["session_id"] = session_id
    return obs


def _build_observations(
    video_dir: Path,
    prompt: str,
    session_id: str,
    *,
    num_requests: int,
) -> tuple[dict[str, np.ndarray], list[dict]]:
    camera_frames = _load_camera_frames(video_dir)
    total_frames = min(frames.shape[0] for frames in camera_frames.values())
    observations = [
        _make_obs_from_video(camera_frames, [0], prompt=prompt, session_id=session_id),
    ]
    for frame_indices in _build_frame_schedule(total_frames, num_requests - 1):
        observations.append(
            _make_obs_from_video(
                camera_frames,
                frame_indices,
                prompt=prompt,
                session_id=session_id,
            )
        )
    if len(observations) < num_requests:
        raise RuntimeError(
            f"Only built {len(observations)} requests from {total_frames} frames; "
            f"requested {num_requests}."
        )
    return camera_frames, observations


def _extract_latents(output) -> object:
    import torch

    if not output.images:
        raise RuntimeError("DreamZero output does not contain video latents in `images`.")

    latents = output.images[0]
    if not isinstance(latents, torch.Tensor):
        raise TypeError(f"Expected tensor latents, got {type(latents)!r}")

    latents = latents.detach().cpu()
    if latents.dim() == 4:
        latents = latents.unsqueeze(0)
    if latents.dim() != 5:
        raise ValueError(f"Unexpected latent shape: {tuple(latents.shape)}")

    if latents.shape[1] < latents.shape[2]:
        latents = latents.transpose(1, 2).contiguous()
    return latents


def _extract_actions(output, *, request_index: int) -> np.ndarray:
    actions = np.asarray(output.multimodal_output.get("actions"), dtype=np.float32)
    while actions.ndim > 2:
        actions = actions[0]
    benchmark_utils.validate_action_array(actions, request_index=request_index)
    return actions


def _run_generation(
    model: str,
    deploy_config_path: Path,
    observations: Sequence[dict],
    *,
    profiler_config: dict[str, Any] | None = None,
    profile_request_index: int | None = None,
):
    from vllm_omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    if (
        profiler_config is not None
        and profile_request_index is not None
        and profile_request_index >= len(observations)
    ):
        raise ValueError(
            f"--profile-request-index={profile_request_index} is outside "
            f"the {len(observations)} generated requests."
        )

    omni = Omni(
        model=model,
        deploy_config=str(deploy_config_path),
        enforce_eager=True,
        worker_extension_cls=WORKER_EXTENSION,
        profiler_config=profiler_config,
    )

    actions: list[np.ndarray] = []
    latent_steps = []
    request_timings: list[benchmark_utils.RequestTiming] = []
    for index, obs in enumerate(observations):
        profile_this_request = profiler_config is not None and index == profile_request_index
        sampling_params = OmniDiffusionSamplingParams(
            extra_args={
                "reset": index == 0,
                "session_id": obs["session_id"],
                "robot_obs": obs,
            }
        )
        if profile_this_request:
            print(f"[Profiler] Starting profiling before request {index}...", flush=True)
            omni.start_profile()
        start = time.perf_counter()
        try:
            result = omni.generate(obs["prompt"], sampling_params_list=[sampling_params])
        finally:
            latency_s = time.perf_counter() - start
            if profile_this_request:
                print(f"[Profiler] Stopping profiler after request {index}...", flush=True)
                profile_results = omni.stop_profile()
                _print_profile_results(profile_results)
        if not result:
            raise RuntimeError(f"No output returned for DreamZero request {index}")

        output = result[0]
        action = _extract_actions(output, request_index=index)
        latents = _extract_latents(output)
        actions.append(action)
        latent_steps.append(latents)
        request_timings.append(
            benchmark_utils.RequestTiming(
                index=index,
                label="initial" if index == 0 else f"chunk_{index - 1}",
                latency_s=latency_s,
                action_count=int(action.shape[0]),
                action_shape=tuple(int(dim) for dim in action.shape),
                video_latent_frames=int(latents.shape[2]),
            )
        )
        print(
            f"REQUEST {index} label={request_timings[-1].label} "
            f"latency_s={latency_s:.6f} action_shape={tuple(action.shape)} "
            f"video_latent_shape={tuple(latents.shape)}",
            flush=True,
        )

    return omni, actions, latent_steps, request_timings


def _print_profile_results(profile_results: object) -> None:
    if profile_results and isinstance(profile_results, dict):
        traces = profile_results.get("traces", [])
        print("PROFILING_RESULTS:", flush=True)
        for rank, trace in enumerate(traces):
            if trace:
                print(f"  rank{rank}: {trace}", flush=True)
        if not traces:
            print("  no traces collected", flush=True)
    else:
        print("[Profiler] No valid profiling data returned.", flush=True)


def _decode_with_worker(omni, full_latents) -> np.ndarray:
    stage_client = omni.engine.stage_clients[0]
    engine = getattr(stage_client, "_engine", None)
    if engine is None:
        raise RuntimeError("DreamZero benchmark requires inline diffusion stage access.")

    decoded = engine.executor.collective_rpc(
        "decode_video_latents_to_uint8",
        args=(full_latents,),
        unique_reply_rank=0,
        exec_all_ranks=True,
    )
    if not isinstance(decoded, np.ndarray):
        decoded = decoded.numpy()
    return decoded


def _write_mp4(path: Path, frames: np.ndarray, fps: int) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames.shape[1:3]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {path}")
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def _write_gif(path: Path, frames: np.ndarray, fps: int) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    images = [Image.fromarray(frame) for frame in frames]
    duration_ms = max(int(round(1000 / max(fps, 1))), 1)
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
    )


def _stitch_input_frames(camera_frames: dict[str, np.ndarray]) -> np.ndarray:
    total_frames = min(frames.shape[0] for frames in camera_frames.values())
    stitched = []
    for frame_index in range(total_frames):
        left = camera_frames["observation/exterior_image_0_left"][frame_index]
        right = camera_frames["observation/exterior_image_1_left"][frame_index]
        wrist = camera_frames["observation/wrist_image_left"][frame_index]
        pad = np.zeros((left.shape[0], left.shape[1], 3), dtype=np.uint8)
        canvas = np.concatenate([left, right], axis=1)
        bottom = np.concatenate([wrist, pad], axis=1)
        stitched.append(np.concatenate([canvas, bottom], axis=0))
    return np.stack(stitched, axis=0)


def _make_side_by_side(input_frames: np.ndarray, output_frames: np.ndarray) -> np.ndarray:
    import cv2

    frame_count = min(len(input_frames), len(output_frames))
    if frame_count == 0:
        raise RuntimeError("Cannot build side-by-side video without frames.")
    output_h, output_w = output_frames.shape[1:3]
    resized_input = [
        cv2.resize(frame, (output_w, output_h), interpolation=cv2.INTER_AREA)
        for frame in input_frames[:frame_count]
    ]
    return np.concatenate([np.stack(resized_input, axis=0), output_frames[:frame_count]], axis=2)


def _save_actions(path: Path, actions: Sequence[np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{f"step{index}": action for index, action in enumerate(actions)})


def _load_reference_actions(path: Path) -> list[np.ndarray]:
    with np.load(path) as payload:
        keys = sorted(payload.files, key=_action_key_sort_value)
        return [np.asarray(payload[key], dtype=np.float32) for key in keys]


def _action_key_sort_value(key: str) -> tuple[int, str]:
    if key.startswith("step") and key[4:].isdigit():
        return int(key[4:]), key
    return 10**9, key


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    os.environ.setdefault("DIFFUSION_ATTENTION_BACKEND", "TORCH_SDPA")

    session_id = args.session_id or f"{DEFAULT_SESSION_PREFIX}-{uuid.uuid4()}"
    camera_frames, observations = _build_observations(
        video_dir=args.video_dir,
        prompt=args.prompt,
        session_id=session_id,
        num_requests=args.num_requests,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_video_path = args.output_dir / f"{args.output_stem}.mp4"
    actions_path = args.output_dir / f"{args.output_stem}_actions.npz"
    summary_path = args.output_dir / f"{args.output_stem}_summary.json"

    if args.save_input_video:
        input_frames = _stitch_input_frames(camera_frames)
        _write_mp4(args.output_dir / f"{args.output_stem}_input.mp4", input_frames, fps=15)

    omni = None
    try:
        omni, actions, latent_steps, request_timings = _run_generation(
            model=args.model,
            deploy_config_path=args.deploy_config,
            observations=observations,
            profiler_config=args.profiler_config,
            profile_request_index=args.profile_request_index,
        )
        import torch

        full_latents = torch.cat(latent_steps, dim=2)
        decode_start = time.perf_counter()
        frames = _decode_with_worker(omni, full_latents)
        decode_latency_s = time.perf_counter() - decode_start
    finally:
        if omni is not None:
            omni.close()

    accuracy: dict[str, object] = {
        "mode": "shape_finite_only",
        "passed": True,
    }
    if args.reference_actions is not None:
        reference_actions = _load_reference_actions(args.reference_actions)
        accuracy = benchmark_utils.compare_action_sequences(
            actions,
            reference_actions,
            atol=args.accuracy_atol,
        )

    write_start = time.perf_counter()
    _write_mp4(output_video_path, frames, fps=args.fps)
    _save_actions(actions_path, actions)
    if args.save_gif:
        _write_gif(args.output_dir / f"{args.output_stem}.gif", frames, fps=args.fps)
    if args.save_side_by_side:
        input_frames = _stitch_input_frames(camera_frames)
        side_by_side = _make_side_by_side(input_frames, frames)
        _write_mp4(args.output_dir / f"{args.output_stem}_side_by_side.mp4", side_by_side, fps=args.fps)
    write_latency_s = time.perf_counter() - write_start

    summary = benchmark_utils.summarize_benchmark(
        request_timings,
        decoded_video_frames=int(frames.shape[0]),
        decode_latency_s=decode_latency_s,
        write_latency_s=write_latency_s,
        output_video_path=str(output_video_path),
        actions_path=str(actions_path),
        accuracy=accuracy,
    )
    _write_json(summary_path, summary)

    print(f"SAVED_MP4={output_video_path}")
    print(f"SAVED_ACTIONS={actions_path}")
    print(f"SAVED_SUMMARY={summary_path}")
    print(f"MODEL_GENERATE_TOTAL_S={summary['latency_s']['model_generate_total']:.6f}")
    print(f"MODEL_VIDEO_FPS={summary['throughput']['model_video_fps']:.6f}")
    print(f"MODEL_ACTION_HZ={summary['throughput']['model_action_hz']:.6f}")
    print(f"MODEL_PLUS_DECODE_VIDEO_FPS={summary['throughput']['model_plus_decode_video_fps']:.6f}")
    print(f"ACCURACY_PASSED={summary['accuracy']['passed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
