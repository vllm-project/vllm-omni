#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover - runtime dependency guard
    raise ImportError("DreamZero OpenPI example requires `opencv-python`.") from exc

try:
    example_dir = str(Path(__file__).resolve().parent)
    removed_path = False
    if sys.path and sys.path[0] == example_dir:
        sys.path.pop(0)
        removed_path = True
    try:
        from openpi_client.websocket_client_policy import WebsocketClientPolicy
    finally:
        if removed_path:
            sys.path.insert(0, example_dir)
except ImportError as exc:  # pragma: no cover - runtime dependency guard
    raise ImportError("DreamZero OpenPI example requires `openpi-client`.") from exc

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
DEFAULT_PATH = "/v1/realtime/robot/openpi"
DEFAULT_PROMPT = "Move the pan forward and use the brush in the middle of the plates to brush the inside of the pan"
ACTION_HORIZON = 24
DEFAULT_ACTION_DIM = 8
RELATIVE_OFFSETS = [-23, -16, -8, 0]
REPO_ROOT = Path(__file__).resolve().parents[3]
ASSET_REPO_ID = "YangshenDeng/vllm-omni-dreamzero-assets"
DEFAULT_VIDEO_DIR = REPO_ROOT / "outputs" / "dreamzero" / "assets"
CAMERA_FILES = {
    "observation/exterior_image_0_left": "exterior_image_1_left.mp4",
    "observation/exterior_image_1_left": "exterior_image_2_left.mp4",
    "observation/wrist_image_left": "wrist_image_left.mp4",
}


def _as_action_array(response: Any) -> np.ndarray:
    if isinstance(response, dict) and response.get("type") == "error":
        message = response.get("message", response)
        raise RuntimeError(f"Inference failed: {message}")
    return np.asarray(response, dtype=np.float32)


@dataclass(frozen=True)
class DreamZeroServerMetadata:
    image_resolution: tuple[int, int]
    n_external_cameras: int
    needs_wrist_camera: bool
    needs_stereo_camera: bool
    needs_session_id: bool
    action_space: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> DreamZeroServerMetadata:
        required_keys = (
            "image_resolution",
            "n_external_cameras",
            "needs_wrist_camera",
            "needs_stereo_camera",
            "needs_session_id",
            "action_space",
        )
        missing_keys = [key for key in required_keys if key not in payload]
        if missing_keys:
            raise ValueError(f"Missing DreamZero metadata keys: {missing_keys}")

        image_resolution = payload["image_resolution"]
        if not isinstance(image_resolution, (list, tuple)) or len(image_resolution) != 2:
            raise ValueError(f"Invalid image_resolution: {image_resolution!r}")

        return cls(
            image_resolution=(int(image_resolution[0]), int(image_resolution[1])),
            n_external_cameras=int(payload["n_external_cameras"]),
            needs_wrist_camera=bool(payload["needs_wrist_camera"]),
            needs_stereo_camera=bool(payload["needs_stereo_camera"]),
            needs_session_id=bool(payload["needs_session_id"]),
            action_space=str(payload["action_space"]),
        )


def _openpi_uri(host: str, port: int, path: str) -> str:
    if host.startswith(("ws://", "wss://")):
        return host
    return f"ws://{host}:{port}{path}"


def _close_policy_client(client: WebsocketClientPolicy) -> None:
    ws = getattr(client, "_ws", None)
    if ws is not None:
        ws.close()


def load_all_frames(video_path: Path) -> np.ndarray:
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


def load_camera_frames(video_dir: Path) -> dict[str, np.ndarray]:
    camera_frames: dict[str, np.ndarray] = {}
    for camera_key, file_name in CAMERA_FILES.items():
        video_path = video_dir / file_name
        if not video_path.exists():
            raise FileNotFoundError(
                f"Missing DreamZero example asset: {video_path}. "
                "Download the example videos with: "
                f"`hf download {ASSET_REPO_ID} --repo-type dataset --local-dir {video_dir}`"
            )
        camera_frames[camera_key] = load_all_frames(video_path)
    return camera_frames


def build_frame_schedule(total_frames: int, num_chunks: int) -> list[list[int]]:
    chunks: list[list[int]] = []
    current_frame = 23
    for _ in range(num_chunks):
        indices = [max(current_frame + offset, 0) for offset in RELATIVE_OFFSETS]
        if indices[-1] >= total_frames:
            break
        chunks.append(indices)
        current_frame += ACTION_HORIZON
    return chunks


def make_obs_from_video(
    camera_frames: dict[str, np.ndarray],
    frame_indices: list[int],
    *,
    prompt: str,
    session_id: str,
) -> dict[str, Any]:
    obs: dict[str, Any] = {}
    for camera_key, all_frames in camera_frames.items():
        selected = all_frames[frame_indices]
        obs[camera_key] = selected[0] if len(frame_indices) == 1 else selected

    obs["observation/joint_position"] = np.zeros(7, dtype=np.float32)
    obs["observation/cartesian_position"] = np.zeros(6, dtype=np.float32)
    obs["observation/gripper_position"] = np.zeros(1, dtype=np.float32)
    obs["prompt"] = prompt
    obs["session_id"] = session_id
    return obs


def build_demo_observations(
    camera_frames: dict[str, np.ndarray],
    *,
    prompt: str,
    session_id: str,
    num_chunks: int = 2,
) -> list[dict[str, Any]]:
    if num_chunks < 1:
        raise ValueError("num_chunks must be at least 1")

    total_frames = min(frames.shape[0] for frames in camera_frames.values())
    observations = [
        make_obs_from_video(
            camera_frames,
            [0],
            prompt=prompt,
            session_id=session_id,
        )
    ]
    for indices in build_frame_schedule(total_frames, num_chunks - 1):
        observations.append(
            make_obs_from_video(
                camera_frames,
                indices,
                prompt=prompt,
                session_id=session_id,
            )
        )
    return observations


def validate_session_result(
    result: dict[str, Any],
    *,
    expected_action_horizon: int = ACTION_HORIZON,
    expected_action_dim: int = DEFAULT_ACTION_DIM,
    expected_action_count: int | None = None,
) -> None:
    metadata = DreamZeroServerMetadata.from_dict(result["metadata"])
    if metadata.image_resolution != (180, 320):
        raise AssertionError(f"Unexpected image_resolution: {metadata.image_resolution}")
    if metadata.n_external_cameras != 2:
        raise AssertionError(f"Unexpected n_external_cameras: {metadata.n_external_cameras}")
    if not metadata.needs_wrist_camera:
        raise AssertionError("DreamZero example expects wrist camera metadata")
    if metadata.action_space != "joint_position":
        raise AssertionError(f"Unexpected action_space: {metadata.action_space}")

    actions = result["actions"]
    if expected_action_count is not None and len(actions) != expected_action_count:
        raise AssertionError(f"Expected {expected_action_count} action tensors, got {len(actions)}")
    if not actions:
        raise AssertionError("Expected at least one action tensor")
    for index, action in enumerate(actions):
        if action.shape != (expected_action_horizon, expected_action_dim):
            raise AssertionError(
                f"Action {index} shape mismatch: expected "
                f"{(expected_action_horizon, expected_action_dim)}, got {action.shape}"
            )
        if not np.isfinite(action).all():
            raise AssertionError(f"Action {index} contains non-finite values")


def run_policy_session(
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    path: str = DEFAULT_PATH,
    video_dir: Path = DEFAULT_VIDEO_DIR,
    prompt: str = DEFAULT_PROMPT,
    session_id: str | None = None,
    num_chunks: int = 2,
) -> dict[str, Any]:
    session_id = session_id or str(uuid.uuid4())
    camera_frames = load_camera_frames(video_dir)
    observations = build_demo_observations(
        camera_frames,
        prompt=prompt,
        session_id=session_id,
        num_chunks=num_chunks,
    )

    uri = _openpi_uri(host, port, path)
    logging.info("Connecting to %s", uri)
    client = WebsocketClientPolicy(host=uri)
    try:
        metadata = dict(client.get_server_metadata())
        actions = [_as_action_array(client.infer(obs)) for obs in observations]
        return {
            "metadata": metadata,
            "actions": actions,
            "session_id": session_id,
        }
    finally:
        _close_policy_client(client)


def format_action_summary(index: int, action: np.ndarray) -> str:
    return (
        f"Action {index}: shape={tuple(action.shape)} dtype={action.dtype} "
        f"min={action.min():.6f} max={action.max():.6f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DreamZero OpenPI client example with downloaded real videos.")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--path", default=DEFAULT_PATH)
    parser.add_argument(
        "--video-dir", type=Path, default=DEFAULT_VIDEO_DIR, help="Directory containing the three camera MP4 files."
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--num-chunks", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    result = run_policy_session(
        host=args.host,
        port=args.port,
        path=args.path,
        video_dir=args.video_dir,
        prompt=args.prompt,
        session_id=args.session_id,
        num_chunks=args.num_chunks,
    )
    validate_session_result(result, expected_action_count=args.num_chunks)

    print("Server metadata:", json.dumps(result["metadata"], sort_keys=True))
    for index, action in enumerate(result["actions"]):
        print(format_action_summary(index, action))
    print("Session ID:", result["session_id"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
