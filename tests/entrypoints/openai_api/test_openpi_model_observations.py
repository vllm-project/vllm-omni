# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Model-shaped observations through the real codec and serving contract.

The engine returns synthetic actions; these tests do not load model weights or
measure policy quality. Observation builders are shared with the GPU E2E tests.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import FastAPI, WebSocket
from omegaconf import OmegaConf
from starlette.testclient import TestClient

from tests.helpers.client import (
    DREAMZERO_CAMERA_FILES,
    build_dreamzero_demo_observations,
    build_openpi_droid_observation,
)
from tests.helpers.runtime import pi0_make_dummy_obs
from vllm_omni.entrypoints.openpi.connection import RobotRealtimeConnection, _pack, _unpack
from vllm_omni.entrypoints.openpi.serving import ServingRealtimeRobotOpenPI
from vllm_omni.outputs import OmniRequestOutput

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]
REPO_ROOT = Path(__file__).resolve().parents[3]


def _assert_observation_equal(actual, expected):
    if isinstance(expected, np.ndarray):
        assert isinstance(actual, np.ndarray)
        assert actual.dtype == expected.dtype
        np.testing.assert_array_equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key, value in expected.items():
            _assert_observation_equal(actual[key], value)
    else:
        assert actual == expected


def _model_case(model):
    session_id = f"contract-{model}"
    if model in ("pi0", "pi05"):
        obs = pi0_make_dummy_obs(prompt="pick up the object", session_id=session_id)
        observations = [obs, obs]
        actions = np.linspace(-1, 1, 50 * 32, dtype=np.float32).reshape(50, 32)
        deploy = f"{model}.yaml"
    elif model == "gr00t":
        obs = build_openpi_droid_observation(session_id=session_id)
        observations = [obs, obs]
        actions = {
            key: np.full((1, 40, dim), 0.25, dtype=np.float32)
            for key, dim in {"eef_9d": 9, "gripper_position": 1, "joint_position": 7}.items()
        }
        deploy = "Gr00tN1d7.yaml"
    else:
        # Exercise both first-call HWC frames and subsequent four-frame history.
        frames = {
            key: np.full((24, 180, 320, 3), index + 1, dtype=np.uint8)
            for index, key in enumerate(DREAMZERO_CAMERA_FILES)
        }
        observations = build_dreamzero_demo_observations(
            frames, prompt="move the pan forward", session_id=session_id, num_chunks=2
        )
        actions = np.linspace(-1, 1, 24 * 8, dtype=np.float32).reshape(24, 8)
        deploy = "dreamzero.yaml"
    config = OmegaConf.load(REPO_ROOT / "vllm_omni" / "deploy" / deploy)
    policy_config = OmegaConf.to_container(config.stages[0].model_config.policy_server_config)
    return observations, actions, policy_config


@pytest.mark.parametrize("model", ["pi0", "pi05", "gr00t", "dreamzero"])
@pytest.mark.parametrize("invalid_horizon", [False, True], ids=["valid", "invalid-horizon"])
def test_model_observations_through_openpi(model, invalid_horizon):
    # Exercise the mandatory serving codec without patching the transport or
    # requiring optional client packages in L1 CI. Official client interoperability
    # is tested separately in test_openpi_connection.py.
    observations, expected_actions, policy_config = _model_case(model)
    horizon = 40 if model == "gr00t" else expected_actions.shape[-2]

    class SyntheticPolicyEngine:
        def __init__(self):
            self.calls = []

        def get_diffusion_od_config(self):
            return SimpleNamespace(model_config={"policy_server_config": policy_config})

        async def generate(self, **kwargs):
            self.calls.append(kwargs)
            yield OmniRequestOutput.from_diffusion(
                request_id=kwargs["request_id"],
                images=[],
                multimodal_output={
                    "actions": expected_actions,
                    "metadata": {
                        "actions": {
                            "horizon": horizon + int(invalid_horizon),
                            "valid_steps": horizon - 1,
                        }
                    },
                },
            )

    engine = SyntheticPolicyEngine()
    serving = ServingRealtimeRobotOpenPI(engine_client=engine)
    app = FastAPI()

    @app.websocket("/v1/realtime/robot/openpi")
    async def endpoint(websocket: WebSocket):
        await RobotRealtimeConnection(websocket, serving).handle_connection()

    with TestClient(app) as client, client.websocket_connect("/v1/realtime/robot/openpi") as websocket:
        assert _unpack(websocket.receive_bytes()) == policy_config
        for index, observation in enumerate(observations):
            websocket.send_bytes(_pack({**observation, "seed": 42}))
            response = _unpack(websocket.receive_bytes())
            if invalid_horizon:
                assert response == {"type": "error", "message": "Internal inference error"}
            else:
                # valid_steps is metadata, not a request to truncate the chunk.
                _assert_observation_equal(response, expected_actions)
            params = engine.calls[index]["sampling_params_list"][0]
            _assert_observation_equal(params.extra_args["robot_obs"], observation)
            assert params.seed == 42
            assert params.extra_args["session_id"] == observation["session_id"]
            assert params.extra_args["reset"] is (index == 0)
        websocket.send_bytes(_pack({"endpoint": "reset"}))
        assert _unpack(websocket.receive_bytes()) == {"status": "reset successful"}
