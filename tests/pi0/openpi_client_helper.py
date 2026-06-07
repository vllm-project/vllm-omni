# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Minimal OpenPI websocket client for the π0 online-serving e2e test.

Mirrors ``tests/dreamzero/openpi_client_helper.py`` but for π0: π0's observation
is just 3 RGB cameras + proprioceptive state + a language prompt (no video), so
this builds synthetic numpy frames directly. Used by
``tests/pi0/test_pi0_e2e.py::test_pi0_openpi_online``.
"""

from __future__ import annotations

import uuid
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
DEFAULT_PROMPT = "pick up the red block and place it in the bin"

# π0 / pi0_base camera identities (must match the server's image_feature_keys,
# i.e. vllm_omni/deploy/pi0.yaml). The deploy yaml's image_key_map is empty, so
# the obs keys are used verbatim.
CAMERA_KEYS = (
    "observation.images.base_0_rgb",
    "observation.images.left_wrist_0_rgb",
    "observation.images.right_wrist_0_rgb",
)
ACTION_HORIZON = 50
ACTION_DIM = 32
STATE_DIM = 32
IMAGE_SIZE = 224


def require_dependencies() -> None:
    missing = []
    if websockets_client is None:
        missing.append("websockets")
    if msgpack_numpy is None:
        missing.append("openpi-client")
    if missing:
        raise ModuleNotFoundError(f"π0 OpenPI test dependencies are missing: {', '.join(missing)}")


def _decode_action_response(response: bytes | str) -> np.ndarray:
    if isinstance(response, str):
        raise RuntimeError(f"Inference failed: {response}")
    decoded = msgpack_numpy.unpackb(response)
    if isinstance(decoded, dict) and decoded.get("type") == "error":
        message = decoded.get("message", decoded)
        raise RuntimeError(f"Inference failed: {message}")
    return np.asarray(decoded, dtype=np.float32)


class OpenPIWebsocketClient:
    """Raw OpenPI websocket client (connect → metadata → infer)."""

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
        return _decode_action_response(self._ws.recv())

    def close(self) -> None:
        self._ws.close()


def make_dummy_obs(*, prompt: str, session_id: str, image_size: int = IMAGE_SIZE) -> dict[str, Any]:
    """A single π0 observation: 3 blank cameras (HWC uint8) + zero state + prompt."""
    obs: dict[str, Any] = {
        cam: np.zeros((image_size, image_size, 3), dtype=np.uint8) for cam in CAMERA_KEYS
    }
    obs["state"] = np.zeros(STATE_DIM, dtype=np.float32)
    obs["prompt"] = prompt
    obs["session_id"] = session_id
    return obs


def run_policy_session(
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    path: str = DEFAULT_PATH,
    prompt: str = DEFAULT_PROMPT,
    session_id: str | None = None,
    num_steps: int = 2,
) -> dict[str, Any]:
    """Connect, read handshake metadata, send ``num_steps`` observations."""
    session_id = session_id or str(uuid.uuid4())
    client = OpenPIWebsocketClient(host=host, port=port, path=path)
    try:
        metadata = client.get_server_metadata()
        actions = [
            client.infer(make_dummy_obs(prompt=prompt, session_id=session_id))
            for _ in range(num_steps)
        ]
        return {"metadata": metadata, "actions": actions, "session_id": session_id}
    finally:
        client.close()


def validate_session_result(
    result: dict[str, Any],
    *,
    expected_action_horizon: int = ACTION_HORIZON,
    expected_action_dim: int = ACTION_DIM,
) -> None:
    """Assert the handshake metadata + every returned action chunk for π0."""
    metadata = result["metadata"]
    required_keys = (
        "image_resolution",
        "needs_wrist_camera",
        "needs_session_id",
        "action_space",
        "action_horizon",
        "action_dim",
    )
    missing = [key for key in required_keys if key not in metadata]
    if missing:
        raise AssertionError(f"Missing π0 metadata keys: {missing}")

    if tuple(metadata["image_resolution"]) != (IMAGE_SIZE, IMAGE_SIZE):
        raise AssertionError(f"Unexpected image_resolution: {metadata['image_resolution']!r}")
    if not metadata["needs_wrist_camera"]:
        raise AssertionError("π0 test expects needs_wrist_camera=True")
    if metadata["needs_session_id"]:
        raise AssertionError("π0 is stateless; needs_session_id must be False")
    if metadata["action_space"] != "joint_position":
        raise AssertionError(f"Unexpected action_space: {metadata['action_space']!r}")
    if int(metadata["action_horizon"]) != expected_action_horizon:
        raise AssertionError(f"Unexpected action_horizon: {metadata['action_horizon']!r}")
    if int(metadata["action_dim"]) != expected_action_dim:
        raise AssertionError(f"Unexpected action_dim: {metadata['action_dim']!r}")

    actions = result["actions"]
    if not actions:
        raise AssertionError("No actions returned from the server")
    for index, action in enumerate(actions):
        if action.shape != (expected_action_horizon, expected_action_dim):
            raise AssertionError(
                f"Action {index} shape mismatch: expected "
                f"{(expected_action_horizon, expected_action_dim)}, got {action.shape}"
            )
        if not np.isfinite(action).all():
            raise AssertionError(f"Action {index} contains non-finite values")
