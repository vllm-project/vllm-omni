# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import Any

import numpy as np

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
DEFAULT_SESSION_ID = "gr00t-smoke"
ACTION_KEYS = {"eef_9d", "gripper_position", "joint_position"}
LANGUAGE_KEY = "annotation.language.language_instruction"


def _identity_eef_9d_state() -> np.ndarray:
    state = np.zeros((1, 1, 9), dtype=np.float32)
    state[..., 3:] = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)
    return state


def require_dependencies() -> None:
    missing = []
    if websockets_client is None:
        missing.append("websockets")
    if msgpack_numpy is None:
        missing.append("openpi-client")
    if missing:
        raise ModuleNotFoundError(f"GR00T OpenPI test dependencies are missing: {', '.join(missing)}")


@dataclass(frozen=True)
class Gr00tServerMetadata:
    action_horizon: int
    action_keys: set[str]
    embodiment_tag: str
    needs_session_id: bool

    @classmethod
    def from_dict(cls, payload: dict[str, Any]):
        required_keys = ("action_horizon", "action_keys", "embodiment_tag", "needs_session_id")
        missing_keys = [key for key in required_keys if key not in payload]
        if missing_keys:
            raise ValueError(f"Missing GR00T metadata keys: {missing_keys}")

        return cls(
            action_horizon=int(payload["action_horizon"]),
            action_keys={str(key) for key in payload["action_keys"]},
            embodiment_tag=str(payload["embodiment_tag"]),
            needs_session_id=bool(payload["needs_session_id"]),
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

    def infer(self, obs: dict[str, Any]) -> dict[str, np.ndarray]:
        payload = dict(obs)
        payload["endpoint"] = "infer"
        self._ws.send(self._packer.pack(payload))
        response = msgpack_numpy.unpackb(self._ws.recv())
        if isinstance(response, dict) and response.get("type") == "error":
            raise RuntimeError(f"Inference failed: {response['message']}")
        if not isinstance(response, dict):
            raise RuntimeError(f"Expected dict actions from GR00T OpenPI endpoint, got {type(response)!r}")
        return {str(key): np.asarray(value, dtype=np.float32) for key, value in response.items()}

    def reset(self, reset_info: dict[str, Any] | None = None) -> str:
        payload = dict(reset_info or {})
        payload["endpoint"] = "reset"
        self._ws.send(self._packer.pack(payload))
        response = msgpack_numpy.unpackb(self._ws.recv())
        if not isinstance(response, dict) or response.get("status") != "reset successful":
            raise RuntimeError(f"Unexpected reset response: {response!r}")
        return str(response["status"])

    def close(self) -> None:
        self._ws.close()


def build_droid_observation(*, session_id: str = DEFAULT_SESSION_ID) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "video": {
            "exterior_image_1_left": np.zeros((1, 2, 256, 256, 3), dtype=np.uint8),
            "wrist_image_left": np.zeros((1, 2, 256, 256, 3), dtype=np.uint8),
        },
        "state": {
            "eef_9d": _identity_eef_9d_state(),
            "gripper_position": np.zeros((1, 1, 1), dtype=np.float32),
            "joint_position": np.zeros((1, 1, 7), dtype=np.float32),
        },
        "language": {LANGUAGE_KEY: [["pick up the object"]]},
    }


def run_policy_session(
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    path: str = DEFAULT_PATH,
    session_id: str = DEFAULT_SESSION_ID,
) -> dict[str, Any]:
    client = OpenPIWebsocketClient(host=host, port=port, path=path)
    try:
        metadata = client.get_server_metadata()
        actions = client.infer(build_droid_observation(session_id=session_id))
        reset_status = client.reset({})
        return {
            "metadata": metadata,
            "actions": actions,
            "reset_status": reset_status,
            "session_id": session_id,
        }
    finally:
        client.close()


def validate_session_result(result: dict[str, Any]) -> None:
    metadata = Gr00tServerMetadata.from_dict(result["metadata"])
    if metadata.embodiment_tag != "OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT":
        raise AssertionError(f"Unexpected embodiment_tag: {metadata.embodiment_tag}")
    if not metadata.needs_session_id:
        raise AssertionError("GR00T test expects needs_session_id metadata")
    if metadata.action_keys != ACTION_KEYS:
        raise AssertionError(f"Unexpected action keys: {metadata.action_keys}")
    if result["reset_status"] != "reset successful":
        raise AssertionError(f"Unexpected reset status: {result['reset_status']!r}")

    actions = result["actions"]
    if set(actions) != ACTION_KEYS:
        raise AssertionError(f"Unexpected action keys: {set(actions)}")
    expected_shapes = {
        "eef_9d": (1, metadata.action_horizon, 9),
        "gripper_position": (1, metadata.action_horizon, 1),
        "joint_position": (1, metadata.action_horizon, 7),
    }
    for key, expected_shape in expected_shapes.items():
        if actions[key].shape != expected_shape:
            raise AssertionError(f"Action {key} shape mismatch: expected {expected_shape}, got {actions[key].shape}")
        if actions[key].dtype != np.float32:
            raise AssertionError(f"Action {key} dtype mismatch: expected float32, got {actions[key].dtype}")
        if not np.isfinite(actions[key]).all():
            raise AssertionError(f"Action {key} contains non-finite values")
