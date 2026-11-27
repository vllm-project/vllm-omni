# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Exercise the dedicated Motubrain socket from a standalone MessagePack client."""

import msgpack
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni.config.omni_config import VllmOmniConfig
from vllm_omni.diffusion.models.motubrain2_pipeline_config import MOTUBRAIN2_PIPELINE
from vllm_omni.entrypoints.motubrain2 import router
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]
ENDPOINT = "/v1/realtime/robot/motubrain2"


def _client_array(value):
    return {b"__ndarray__": True, b"data": value.tobytes(), b"dtype": value.dtype.str, b"shape": value.shape}


def _read_array(message):
    assert message[b"__ndarray__"] is True
    return np.frombuffer(message[b"data"], dtype=message[b"dtype"]).reshape(message[b"shape"])


def test_astribot_client_handshake_inference_and_reset():
    stage = VllmOmniConfig.from_pipeline_config(
        MOTUBRAIN2_PIPELINE, deploy_config_path=get_deploy_config_path("motubrain2-astribot.yaml")
    ).stage_configs[0]
    metadata = stage.diffusion_config.model_config["motubrain2_metadata"]
    actions = np.full((24, 25), 0.25, dtype=np.float32)
    mask = np.ones_like(actions, dtype=bool)
    mask[-1, 0] = False
    native = {
        "video": np.ones((2, 1), dtype=np.float32),
        "action": np.ones((2, 4), dtype=np.float32),
        "decoded_action": actions.astype(np.float64),
        "decoded_valid": mask,
    }
    calls = []

    class Engine:
        stage_configs = [stage]
        default_sampling_params_list = [OmniDiffusionSamplingParams(seed=17)]

        async def generate(self, *, prompt, request_id, sampling_params_list):
            (params,) = sampling_params_list
            calls.append((prompt, params))
            yield OmniRequestOutput.from_diffusion(
                request_id=request_id, images=[], multimodal_output={"actions": actions, "native_outputs": native}
            )

    app = FastAPI()
    app.state.engine_client = Engine()
    app.include_router(router)
    frame = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    obs = {
        "instruction": "Close the drawer",
        "cameras": {name: _client_array(frame) for name in metadata["camera_sources"]},
        "session_id": "rollout-A",
    }
    with TestClient(app) as client, client.websocket_connect(ENDPOINT) as socket:
        assert msgpack.unpackb(socket.receive_bytes()) == metadata
        socket.send_bytes(msgpack.packb(obs))
        reply = _read_array(msgpack.unpackb(socket.receive_bytes()))
        assert reply.dtype == np.float32
        np.testing.assert_array_equal(reply, actions)
        prompt, params = calls[-1]
        assert prompt == obs["instruction"]
        assert params.extra_args["reset"] is True
        assert params.extra_args["session_id"] == "rollout-A"
        assert params.extra_args["motubrain2_session_seed"] == 17
        for camera in metadata["camera_sources"]:
            np.testing.assert_array_equal(params.extra_args["robot_obs"]["cameras"][camera], frame)

        socket.send_bytes(
            msgpack.packb({**obs, "seed": 0, "sampling_params": {"seed": 7}, "return_native_outputs": True})
        )
        diagnostic = msgpack.unpackb(socket.receive_bytes())
        assert diagnostic.keys() == native.keys()
        for name, expected in native.items():
            actual = _read_array(diagnostic[name])
            assert actual.dtype == expected.dtype
            np.testing.assert_array_equal(actual, expected)
        assert calls[-1][1].extra_args["reset"] is False
        assert calls[-1][1].seed == 0
        assert "motubrain2_session_seed" not in calls[-1][1].extra_args

        socket.send_bytes(msgpack.packb({"endpoint": "reset"}))
        assert msgpack.unpackb(socket.receive_bytes()) == {"status": "reset successful"}
        socket.send_bytes(msgpack.packb({**obs, "sampling_params": {"seed": 0, "num_inference_steps": 2}}))
        _read_array(msgpack.unpackb(socket.receive_bytes()))
        _, params = calls[-1]
        assert params.extra_args["reset"] is True
        assert params.seed == 0 and params.num_inference_steps == 2
        assert "motubrain2_session_seed" not in params.extra_args
