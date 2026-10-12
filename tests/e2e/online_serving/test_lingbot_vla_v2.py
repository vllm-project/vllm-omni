# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Real-checkpoint OpenPI smoke test; requires one CUDA GPU with 40 GiB free.

Set LINGBOT_MODEL to a local snapshot from the recipe to run offline.
This exercises serving, not robot task success.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServerParams

PARAMS = OmniServerParams(
    model=os.environ.get("LINGBOT_MODEL", "lerobot/lingbot_vla_v2_robotwin"),
    port=8093,
    server_args=[
        "--deploy-config",
        "vllm_omni/deploy/lingbot_vla_v2.yaml",
        "--enforce-eager",
        "--disable-log-stats",
    ],
    env_dict={"VLLM_WORKER_MULTIPROC_METHOD": "spawn"},
)


@pytest.mark.slow
@pytest.mark.diffusion
@hardware_test(res={"cuda": "H100"})
@pytest.mark.parametrize("omni_server", [PARAMS], indirect=True)
def test_lingbot_openpi(omni_server):
    from websockets.sync.client import connect

    from vllm_omni.entrypoints.openpi.connection import _pack, _unpack

    reference = json.loads((Path(__file__).parent / "fixtures/lingbot_vla_v2_reference.json").read_text())
    cameras = reference["camera_keys"]
    y, x = np.mgrid[:256, :256]
    frames = {
        key: np.stack([(x + i * 71) % 256, (y + i * 31) % 256, ((x // 16 + y // 16) % 2) * 255], -1).astype(np.uint8)
        for i, key in enumerate(cameras)
    }
    uri = f"ws://{omni_server.host}:{omni_server.port}/v1/realtime/robot/openpi"
    with connect(uri, max_size=64 * 1024 * 1024) as ws:
        metadata = _unpack(ws.recv(timeout=60))
        assert metadata["action_dim"] == 14
        assert metadata["action_horizon"] == 50
        assert metadata["needs_session_id"] is False
        # Change camera masks in one connection, including a missing middle view.
        for keys in (cameras[:1], (cameras[0], cameras[2]), cameras):
            observation = {key: frames[key] for key in keys}
            observation.update(
                state=np.asarray(reference["state"], dtype=np.float32),
                prompt=reference["prompt"],
                seed=123,
                sampling_params={"num_inference_steps": 2},
            )
            ws.send(_pack(observation))
            actions = _unpack(ws.recv(timeout=240))
            assert isinstance(actions, np.ndarray)
            assert actions.shape == (50, 14)
            assert actions.dtype == np.float32
            assert np.isfinite(actions).all()
        observation["sampling_params"]["num_inference_steps"] = reference["num_inference_steps"]
        ws.send(_pack(observation))
        baseline = _unpack(ws.recv(timeout=240))
        np.testing.assert_allclose(baseline, reference["actions"], atol=1e-4, rtol=1e-5)
        # Interleave a different observation and a rejected request, then recover.
        other = observation | {"seed": 124, "prompt": "put the blue object in the tray"}
        ws.send(_pack(other))
        assert _unpack(ws.recv(timeout=240)).shape == (50, 14)
        ws.send(_pack(observation | {"state": np.zeros(13, dtype=np.float32)}))
        assert _unpack(ws.recv(timeout=60))["type"] == "error"
        ws.send(_pack(observation))
        np.testing.assert_allclose(_unpack(ws.recv(timeout=240)), baseline, atol=1e-4, rtol=1e-5)
        ws.send(_pack({"endpoint": "reset"}))
        assert _unpack(ws.recv(timeout=60))["status"] == "reset successful"
    with connect(uri, max_size=64 * 1024 * 1024) as ws:
        ws.recv(timeout=60)
        ws.send(_pack(observation))
        np.testing.assert_allclose(_unpack(ws.recv(timeout=240)), baseline, atol=1e-4, rtol=1e-5)
