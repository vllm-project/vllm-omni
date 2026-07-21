# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vllm_omni.entrypoints.omni import Omni
from vllm_omni.outputs import OmniRequestOutput

DREAMZERO_EXTRA_BODY_PARAMS: frozenset[str] = frozenset(
    {
        "robot_obs",
        "reset",
        "session_id",
    }
)
DREAMZERO_EXTRA_OUTPUT_PARAMS: frozenset[str] = frozenset()

ACTION_HORIZON = 24
RELATIVE_OFFSETS = (-23, -16, -8, 0)
CAMERA_FILES = {
    "observation/exterior_image_0_left": "exterior_image_1_left.mp4",
    "observation/exterior_image_1_left": "exterior_image_2_left.mp4",
    "observation/wrist_image_left": "wrist_image_left.mp4",
}
DEFAULT_NUM_CHUNKS = 15
DEFAULT_EXPORT_FPS = 5
DREAMZERO_WORKER_EXTENSION_CLS = (
    "vllm_omni.diffusion.models.dreamzero.video_export_worker.DreamZeroVideoExportWorkerExtension"
)


def _write_mp4(video_path: str, frames: np.ndarray, fps: int) -> None:
    import cv2

    height, width = frames.shape[1:3]
    writer = cv2.VideoWriter(
        video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {video_path}")
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def _load_video_frames(video_path: Path) -> np.ndarray:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover
        raise ImportError("DreamZero observation loading requires opencv-python.") from exc

    cap = cv2.VideoCapture(str(video_path))
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames loaded from {video_path}")
    return np.stack(frames, axis=0)


def _build_frame_schedule(total_frames: int, num_chunks: int) -> list[list[int]]:
    chunks: list[list[int]] = []
    current = 23
    for _ in range(num_chunks):
        indices = [max(current + off, 0) for off in RELATIVE_OFFSETS]
        if indices[-1] >= total_frames:
            break
        chunks.append(indices)
        current += ACTION_HORIZON
    return chunks


def _make_obs(
    camera_frames: dict[str, np.ndarray],
    frame_indices: list[int],
    *,
    prompt: str,
) -> dict[str, Any]:
    obs: dict[str, Any] = {}
    for key, all_frames in camera_frames.items():
        selected = all_frames[frame_indices]
        obs[key] = selected[0] if len(frame_indices) == 1 else selected
    obs["observation/joint_position"] = np.zeros(7, dtype=np.float32)
    obs["observation/cartesian_position"] = np.zeros(6, dtype=np.float32)
    obs["observation/gripper_position"] = np.zeros(1, dtype=np.float32)
    obs["prompt"] = prompt
    return obs


def build_observations(model_dir, task, data_dir, **extra_params) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read 3 camera MP4s and yield DreamZero AR extra_args dicts.

    Yields a sequence (autoregressive mode). The first item carries
    ``reset=True``; all share one ``session_id``.
    """
    if data_dir is None:
        raise ValueError("DreamZero requires source['data_dir'] with 3 camera MP4 files.")
    data_dir = Path(data_dir)
    session_id = extra_params.get("session_id") or str(uuid.uuid4())
    num_chunks = extra_params.get("num_chunks", DEFAULT_NUM_CHUNKS)
    repeat_chunk_observations = bool(extra_params.get("repeat_chunk_observations"))

    camera_frames: dict[str, np.ndarray] = {}
    for camera_key, file_name in CAMERA_FILES.items():
        path = data_dir / file_name
        if not path.exists():
            raise FileNotFoundError(f"Camera video not found: {path}")
        camera_frames[camera_key] = _load_video_frames(path)

    total = min(f.shape[0] for f in camera_frames.values())
    schedule = [[0]] + _build_frame_schedule(total, num_chunks)

    if repeat_chunk_observations and len(schedule) <= num_chunks:
        # <= num_chunks because schedule already contains the initial frame at index 0
        while len(schedule) < num_chunks:
            schedule.append(schedule[-1])

    observations = []
    for index, frame_indices in enumerate(schedule):
        obs = _make_obs(camera_frames, frame_indices, prompt=task)
        obs["session_id"] = session_id
        observations.append(
            {
                "reset": index == 0,
                "robot_obs": obs,
                "session_id": session_id,
                "prompt": task,
            }
        )
    return observations, {}


def process_robot_actions(
    output: OmniRequestOutput,
    **kwargs,
) -> dict[str, Any]:
    """Extract actions from DreamZero's DiffusionOutput.

    DreamZero returns ``DiffusionOutput(output={"actions": np.ndarray, ...})``.
    This processor extracts the actions array and passes through any extra metadata.
    """
    action_output = output.multimodal_output.get("actions")
    action_array = np.asarray(action_output)

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
    return {"actions": action_array, "metadata": {"video_latents": latents}}


def finalize(omni: Omni, results: list[dict], output_path) -> None:
    """Decode accumulated video latents into an mp4. Calling `decode_video_latents_to_uint8`
    provided by `vllm_omni/diffusion/models/dreamzero/video_export_worker.py`.
    """
    video_latents = [r["metadata"].get("video_latents") for r in results]
    print(f"[Robot Policy - dreamzero] Decoding {len(video_latents)} steps...")

    if video_latents[0] is None:
        raise RuntimeError("[Robot Policy - dreamzero] Video latents not found in output.")

    full_latents = torch.cat(video_latents, dim=2)
    stage_client = omni.engine.stage_clients[0]
    engine = getattr(stage_client, "_engine", None)
    if engine is None:
        raise RuntimeError("[Robot Policy - dreamzero] Video export requires inline diffusion stage access.")

    decoded = engine.executor.collective_rpc(
        "decode_video_latents_to_uint8",
        args=(full_latents,),
        unique_reply_rank=0,
        exec_all_ranks=True,
    )
    if isinstance(decoded, torch.Tensor):
        decoded = decoded.numpy()
    if not isinstance(decoded, np.ndarray):
        raise TypeError(f"Unexpected decoded output type: {type(decoded)!r}")

    _write_mp4(output_path.with_suffix(".mp4"), decoded, fps=DEFAULT_EXPORT_FPS)
