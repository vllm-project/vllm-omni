# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Exercise the shared robot transport with the Motubrain Astribot deployment."""

from types import SimpleNamespace

import numpy as np
import pytest

from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni.config.omni_config import VllmOmniConfig
from vllm_omni.diffusion.models.motubrain2_pipeline_config import MOTUBRAIN2_PIPELINE
from vllm_omni.entrypoints.openpi.serving import ServingRealtimeRobotOpenPI
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.fixture
def astribot_serving():
    stage = VllmOmniConfig.from_pipeline_config(
        MOTUBRAIN2_PIPELINE,
        deploy_config_path=get_deploy_config_path("motubrain2-astribot.yaml"),
    ).stage_configs[0]
    params = OmniDiffusionSamplingParams(**stage.model_config.default_sampling_params)
    engine = SimpleNamespace(stage_configs=[stage], default_sampling_params_list=[params])
    return ServingRealtimeRobotOpenPI(engine_client=engine)


def test_astribot_observation_uses_the_native_prompt(astribot_serving):
    cameras = {
        name: np.zeros((2, 2, 3), dtype=np.uint8)
        for name in astribot_serving.policy_server_config.values["camera_sources"]
    }
    obs = {"prompt": "Close the drawer", "cameras": cameras}
    request = astribot_serving.build_request(obs, session_id="rollout", reset=True)
    assert astribot_serving.prompt_builder is None
    assert request.prompt == obs["prompt"]
    forwarded = request.sampling_params.extra_args["robot_obs"]
    assert forwarded["cameras"].keys() == {"head_camera", "left_camera", "right_camera"}
    for camera, frame in cameras.items():
        assert forwarded["cameras"][camera] is frame


def test_session_rng_preserves_omission_and_explicit_zero(astribot_serving):
    astribot_serving.engine_client.default_sampling_params_list[0].seed = 17
    omitted = astribot_serving.build_request({}, session_id="rollout", reset=False).sampling_params
    assert omitted.extra_args["openpi_session_seed"] == 17
    for obs in ({"seed": 0}, {"sampling_params": {"seed": 0}}, {"seed": 0, "sampling_params": {"seed": 7}}):
        explicit = astribot_serving.build_request(obs, session_id="rollout", reset=False).sampling_params
        assert explicit.seed == 0
        assert "openpi_session_seed" not in explicit.extra_args


def test_native_reply_is_opt_in_and_preserves_array_dtypes(astribot_serving):
    handshake = astribot_serving.policy_server_config.values
    shape = (handshake["action_horizon"], handshake["action_dim"])
    controller = np.full(shape, 0.25, dtype=np.float64)
    result = SimpleNamespace(multimodal_output={"actions": controller})
    assert astribot_serving._extract_actions(result, return_native_outputs=True).dtype == np.float32
    result.multimodal_output["native_outputs"] = {
        "decoded_action": controller,
        "decoded_valid": np.ones(shape, dtype=bool),
    }
    assert astribot_serving._extract_actions(result).dtype == np.float32
    reply = astribot_serving._extract_actions(result, return_native_outputs=True)
    assert reply["decoded_action"].dtype == np.float64
    assert reply["decoded_valid"].dtype == np.bool_
