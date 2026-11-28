# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The ABot stepwise deploy configs must keep serving on the AR-Diffusion path.

`WS /v1/realtime/video` only reaches the stepwise code when the stage selects the
AR-Diffusion engine and asks for streamed step execution, so those fields are
pinned here rather than left to drift.
"""

from __future__ import annotations

import pytest

from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni.config.pipeline_registry import OMNI_PIPELINES
from vllm_omni.config.stage_config import load_deploy_config

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

# Deploy file -> the fixed AR cache geometry (height, width) of its checkpoint.
_DEPLOYS = {
    "abot_world_stepwise.yaml": (512, 832),
    "abot_world_iris_stepwise.yaml": (704, 1280),
}


def _stage(deploy: str):
    config = load_deploy_config(get_deploy_config_path(deploy))
    assert len(config.stages) == 1, "stepwise serving is a single diffusion stage"
    return config.stages[0]


def test_pipeline_is_registered_without_a_default_deploy_config() -> None:
    pipeline = OMNI_PIPELINES["abot_world"]
    assert pipeline.model_arch == "ABotWorldCausalPipeline"
    # A default would move offline callers that pass no deploy config onto
    # the stepwise serving topology.
    assert pipeline.default_deploy_config_name is None


@pytest.mark.parametrize("deploy", _DEPLOYS)
def test_stage_selects_the_ar_diffusion_engine_and_pipeline(deploy: str) -> None:
    stage = _stage(deploy)
    assert stage.model_class_name == "ABotWorldCausalPipeline"
    assert stage.engine_extras["engine_backend"].endswith("ar_diffusion.engine.ARDiffusionEngine")


@pytest.mark.parametrize("deploy", _DEPLOYS)
def test_stage_requests_eager_streamed_step_execution_on_one_sequence(deploy: str) -> None:
    stage = _stage(deploy)
    assert stage.step_execution is True
    assert stage.engine_extras["streaming_output"] is True
    assert stage.enforce_eager is True
    assert stage.max_num_seqs == 1


@pytest.mark.parametrize(("deploy", "geometry"), _DEPLOYS.items())
def test_stage_declares_the_fixed_ar_cache_geometry(deploy: str, geometry: tuple[int, int]) -> None:
    model_config = _stage(deploy).engine_extras["model_config"]
    assert (model_config["ar_diffusion_height"], model_config["ar_diffusion_width"]) == geometry
    assert model_config["ar_diffusion_kv_config"]["warmup_cudagraph"] is False
