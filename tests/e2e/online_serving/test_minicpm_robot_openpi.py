# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end online serving test for MiniCPM-RobotManip through OpenPI."""

import os
from typing import Any

import numpy as np
import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServerParams
from tests.helpers.stage_config import get_deploy_config_path

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

MODEL = "openbmb/MiniCPM-RobotManip"

pytest.importorskip("websockets")
pytest.importorskip("openpi_client.msgpack_numpy")

test_params = [
    pytest.param(
        OmniServerParams(
            model=MODEL,
            stage_config_path=get_deploy_config_path("MiniCPMRobotManip.yaml"),
            server_args=[
                "--disable-log-stats",
                "--trust-remote-code",
            ],
            init_timeout=600,
            stage_init_timeout=450,
        ),
        id="minicpm-robot-openpi",
    )
]


def _openpi_connect(host: str, port: int) -> tuple[Any, Any]:
    import websockets.sync.client as ws_client
    from openpi_client import msgpack_numpy

    uri = f"ws://{host}:{port}/v1/realtime/robot/openpi"
    ws = ws_client.connect(uri)
    metadata = msgpack_numpy.unpackb(ws.recv())
    return ws, metadata


def _openpi_infer(ws: Any, obs: dict[str, Any]) -> dict[str, np.ndarray]:
    from openpi_client import msgpack_numpy

    payload = dict(obs)
    payload["endpoint"] = "infer"
    ws.send(msgpack_numpy.packb(payload))
    response = msgpack_numpy.unpackb(ws.recv())
    if isinstance(response, dict) and response.get("type") == "error":
        raise RuntimeError(f"Inference failed: {response['message']}")
    if not isinstance(response, dict):
        raise RuntimeError(f"Expected dict, got {type(response)!r}")
    return {str(k): np.asarray(v, dtype=np.float32) for k, v in response.items()}


def _build_observation() -> dict[str, Any]:
    return {
        "images": [
            np.zeros((448, 448, 3), dtype=np.uint8),
            np.zeros((448, 448, 3), dtype=np.uint8),
        ],
        "state": np.zeros(80, dtype=np.float32),
        "language": "pick up the black bowl",
    }


@pytest.mark.full_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_minicpm_robot_openpi_infer(omni_server) -> None:
    ws, metadata = _openpi_connect(omni_server.host, omni_server.port)

    try:
        assert isinstance(metadata, dict), f"Expected dict metadata, got {type(metadata)!r}"

        actions = _openpi_infer(ws, _build_observation())

        assert "default" in actions, f"Missing 'default' key in actions; got {list(actions)}"
        assert actions["default"].dtype == np.float32
        assert actions["default"].shape == (1, 30, 80), f"Expected (1, 30, 80), got {actions['default'].shape}"
        assert np.isfinite(actions["default"]).all(), "Actions contain non-finite values"
    finally:
        ws.close()
