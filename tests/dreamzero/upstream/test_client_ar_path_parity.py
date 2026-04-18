# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Checks that the copied DreamZero client keeps identical logic across paths.

The only difference between talking to the upstream DreamZero websocket server
and the vLLM OpenPI websocket server should be the websocket URI suffix:

- upstream DreamZero: ``ws://host:port``
- vLLM OpenPI: ``ws://host:port/v1/realtime/robot/openpi``

This file verifies that `tests/dreamzero/upstream/openpi_test_client_ar.py` preserves the same
observation / infer / reset flow for both cases.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import uuid
from pathlib import Path

import numpy as np
import pytest

DREAMZERO_REPO = Path(os.environ.get("DREAMZERO_REPO", "~/code/dreamzero")).expanduser()
CLIENT_SCRIPT = Path(__file__).resolve().with_name("openpi_test_client_ar.py")

pytestmark = pytest.mark.skipif(
    not DREAMZERO_REPO.exists(),
    reason="DreamZero source repo is required",
)

if str(DREAMZERO_REPO) not in sys.path:
    sys.path.insert(0, str(DREAMZERO_REPO))


def _load_client_module():
    spec = importlib.util.spec_from_file_location(
        "dreamzero_test_client_ar_module",
        CLIENT_SCRIPT,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError as exc:
        pytest.skip(f"DreamZero client dependency is missing: {exc.name}")
    return module


def _snapshot_obs(obs: dict) -> dict:
    snapshot = {}
    for key, value in obs.items():
        if isinstance(value, np.ndarray):
            snapshot[key] = value.copy()
        else:
            snapshot[key] = value
    return snapshot


def _assert_obs_sequence_equal(actual: list[dict], expected: list[dict]) -> None:
    assert len(actual) == len(expected)
    for actual_obs, expected_obs in zip(actual, expected, strict=True):
        assert set(actual_obs) == set(expected_obs)
        for key in actual_obs:
            actual_value = actual_obs[key]
            expected_value = expected_obs[key]
            if isinstance(actual_value, np.ndarray):
                assert isinstance(expected_value, np.ndarray)
                assert actual_value.dtype == expected_value.dtype
                assert actual_value.shape == expected_value.shape
                assert np.array_equal(actual_value, expected_value)
            else:
                assert actual_value == expected_value


def test_websocket_uri_differs_only_by_path(monkeypatch) -> None:
    client_mod = _load_client_module()
    monkeypatch.setattr(
        client_mod.OpenPIWebsocketClientPolicy,
        "_wait_for_server",
        lambda self: (object(), {}),
        raising=False,
    )

    upstream = client_mod.OpenPIWebsocketClientPolicy(
        host="127.0.0.1",
        port=8000,
        path="",
    )
    vllm = client_mod.OpenPIWebsocketClientPolicy(
        host="127.0.0.1",
        port=8000,
        path="/v1/realtime/robot/openpi",
    )

    assert upstream._uri == "ws://127.0.0.1:8000"
    assert vllm._uri == "ws://127.0.0.1:8000/v1/realtime/robot/openpi"


def test_zero_image_client_flow_is_identical_across_server_paths(monkeypatch) -> None:
    client_mod = _load_client_module()
    fixed_session_id = uuid.UUID("12345678-1234-5678-1234-567812345678")
    monkeypatch.setattr(uuid, "uuid4", lambda: fixed_session_id)
    monkeypatch.setattr(client_mod, "_log_action", lambda actions, dt: None)

    instances = []

    class FakeClient:
        def __init__(self, host: str, port: int, path: str) -> None:
            self.host = host
            self.port = port
            self.path = path
            self.metadata_calls = 0
            self.infer_obs = []
            self.reset_payloads = []
            instances.append(self)

        def get_server_metadata(self) -> dict:
            self.metadata_calls += 1
            return {
                "image_resolution": [180, 320],
                "n_external_cameras": 2,
                "needs_wrist_camera": True,
                "needs_stereo_camera": False,
                "needs_session_id": True,
                "action_space": "joint_position",
            }

        def infer(self, obs: dict) -> np.ndarray:
            self.infer_obs.append(_snapshot_obs(obs))
            return np.zeros((24, 8), dtype=np.float32)

        def reset(self, payload: dict) -> str:
            self.reset_payloads.append(dict(payload))
            return "reset successful"

    monkeypatch.setattr(client_mod, "OpenPIWebsocketClientPolicy", FakeClient)

    client_mod.test_ar_droid_policy_server(
        host="127.0.0.1",
        port=8000,
        path="",
        num_chunks=2,
        prompt="pick up the object",
        use_zero_images=True,
    )
    client_mod.test_ar_droid_policy_server(
        host="127.0.0.1",
        port=8000,
        path="/v1/realtime/robot/openpi",
        num_chunks=2,
        prompt="pick up the object",
        use_zero_images=True,
    )

    assert len(instances) == 2
    upstream, vllm = instances

    assert upstream.path == ""
    assert vllm.path == "/v1/realtime/robot/openpi"
    assert upstream.metadata_calls == 1
    assert vllm.metadata_calls == 1
    assert len(upstream.infer_obs) == 2
    assert len(vllm.infer_obs) == 2
    _assert_obs_sequence_equal(upstream.infer_obs, vllm.infer_obs)
    assert upstream.reset_payloads == [{}]
    assert vllm.reset_payloads == [{}]
