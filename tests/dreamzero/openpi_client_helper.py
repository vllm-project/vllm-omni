# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - optional e2e dependency
    cv2 = None

try:
    import websockets.sync.client as websockets_client
except ImportError:  # pragma: no cover - optional e2e dependency
    websockets_client = None

try:
    from openpi_client import msgpack_numpy
except ImportError:  # pragma: no cover - optional e2e dependency
    msgpack_numpy = None

PING_INTERVAL_SECS = 300
PING_TIMEOUT_SECS = 3600
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
DEFAULT_PATH = "/v1/realtime/robot/openpi"
DEFAULT_PROMPT = "Move the pan forward and use the brush in the middle of the plates to brush the inside of the pan"
ACTION_HORIZON = 24
DEFAULT_ACTION_DIM = 8
RELATIVE_OFFSETS = [-23, -16, -8, 0]
CAMERA_FILES = {
    "observation/exterior_image_0_left": "exterior_image_1_left.mp4",
    "observation/exterior_image_1_left": "exterior_image_2_left.mp4",
    "observation/wrist_image_left": "wrist_image_left.mp4",
}


def _decode_action_response(response: bytes | str) -> np.ndarray:
    if isinstance(response, str):
        raise RuntimeError(f"Inference failed: {response}")
    decoded = msgpack_numpy.unpackb(response)
    if isinstance(decoded, dict) and decoded.get("type") == "error":
        message = decoded.get("message", decoded)
        raise RuntimeError(f"Inference failed: {message}")
    return np.asarray(decoded, dtype=np.float32)


def require_dependencies() -> None:
    missing = []
    if cv2 is None:
        missing.append("opencv-python")
    if websockets_client is None:
        missing.append("websockets")
    if msgpack_numpy is None:
        missing.append("openpi-client")
    if missing:
        raise ModuleNotFoundError(f"DreamZero OpenPI test dependencies are missing: {', '.join(missing)}")


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


class OpenPIWebsocketClient:
    def __init__(
        self,
        *,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        path: str = DEFAULT_PATH,
    ) -> None:
        require_dependencies()
        self._uri = f"ws://{host}:{port}{path}"
        self._packer = msgpack_numpy.Packer()
        self._ws, self._server_metadata = self._connect()

    def _connect(self):
        conn = websockets_client.connect(
            self._uri,
            compression=None,
            max_size=None,
            ping_interval=PING_INTERVAL_SECS,
            ping_timeout=PING_TIMEOUT_SECS,
        )
        metadata = msgpack_numpy.unpackb(conn.recv())
        if not isinstance(metadata, dict):
            raise TypeError(f"Expected dict metadata from server, got {type(metadata)!r}")
        return conn, metadata

    def get_server_metadata(self) -> dict[str, Any]:
        return dict(self._server_metadata)

    def infer(self, obs: dict[str, Any]) -> np.ndarray:
        payload = dict(obs)
        payload["endpoint"] = "infer"
        self._ws.send(self._packer.pack(payload))
        response = self._ws.recv()
        return _decode_action_response(response)

    def reset(self, reset_info: dict[str, Any] | None = None) -> str:
        payload = dict(reset_info or {})
        payload["endpoint"] = "reset"
        self._ws.send(self._packer.pack(payload))
        response = self._ws.recv()
        if isinstance(response, str):
            return response
        decoded = msgpack_numpy.unpackb(response)
        if not isinstance(decoded, dict) or decoded.get("status") != "reset successful":
            raise RuntimeError(f"Unexpected reset response: {decoded!r}")
        return str(decoded["status"])

    def close(self) -> None:
        self._ws.close()


def load_all_frames(video_path: Path) -> np.ndarray:
    require_dependencies()
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
            raise FileNotFoundError(f"Missing DreamZero test asset: {video_path}")
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
) -> None:
    metadata = DreamZeroServerMetadata.from_dict(result["metadata"])
    if metadata.image_resolution != (180, 320):
        raise AssertionError(f"Unexpected image_resolution: {metadata.image_resolution}")
    if metadata.n_external_cameras != 2:
        raise AssertionError(f"Unexpected n_external_cameras: {metadata.n_external_cameras}")
    if not metadata.needs_wrist_camera:
        raise AssertionError("DreamZero test expects wrist camera metadata")
    if metadata.action_space != "joint_position":
        raise AssertionError(f"Unexpected action_space: {metadata.action_space}")

    actions = result["actions"]
    if len(actions) != 3:
        raise AssertionError(f"Expected 3 action tensors, got {len(actions)}")
    for index, action in enumerate(actions):
        if action.shape != (expected_action_horizon, expected_action_dim):
            raise AssertionError(
                f"Action {index} shape mismatch: expected "
                f"{(expected_action_horizon, expected_action_dim)}, got {action.shape}"
            )
        if not np.isfinite(action).all():
            raise AssertionError(f"Action {index} contains non-finite values")

    if result["reset_status"] != "reset successful":
        raise AssertionError(f"Unexpected reset status: {result['reset_status']!r}")


def run_policy_session(
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    path: str = DEFAULT_PATH,
    video_dir: Path,
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

    client = OpenPIWebsocketClient(host=host, port=port, path=path)
    try:
        metadata = client.get_server_metadata()
        actions = [client.infer(obs) for obs in observations]
        reset_status = client.reset({})
        actions.append(client.infer(observations[0]))
        return {
            "metadata": metadata,
            "actions": actions,
            "reset_status": reset_status,
            "session_id": session_id,
        }
    finally:
        client.close()
