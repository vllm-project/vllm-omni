# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The LingBot stepwise deploy config must keep serving on the AR-Diffusion path.

`WS /v1/realtime/video` only reaches the stepwise code when the stage selects the
AR-Diffusion engine and asks for streamed step execution, so those fields are
pinned here rather than left to drift.
"""

from __future__ import annotations

import pytest

from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni.config.stage_config import load_deploy_config

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_DEPLOY = "lingbot_world_v2_stepwise.yaml"


def _stage():
    deploy = load_deploy_config(get_deploy_config_path(_DEPLOY))
    assert len(deploy.stages) == 1, "stepwise serving is a single diffusion stage"
    return deploy.stages[0]


def test_stage_selects_the_ar_diffusion_engine_and_pipeline() -> None:
    stage = _stage()
    assert stage.model_class_name == "LingBotWorldCausalDMDPipeline"
    # engine_backend/model_config reach the engine through the passthrough, the
    # same way the DreamZero AR-Diffusion deploy config does.
    assert stage.engine_extras["engine_backend"].endswith("ar_diffusion.engine.ARDiffusionEngine")


def test_stage_requests_streamed_step_execution_on_one_sequence() -> None:
    stage = _stage()
    assert stage.step_execution is True
    assert stage.engine_extras["streaming_output"] is True
    # AR-Diffusion binds one session per runner invocation; batching requests
    # would break that contract, so the runner rejects max_num_seqs > 1.
    assert stage.max_num_seqs == 1


def test_stage_declares_the_fixed_ar_cache_geometry() -> None:
    stage = _stage()
    model_config = stage.engine_extras["model_config"]
    # The KV cache geometry is fixed at load time, so a served request must ask
    # for exactly this resolution.
    assert model_config["ar_diffusion_height"] == 480
    assert model_config["ar_diffusion_width"] == 832
    assert model_config["ar_diffusion_kv_config"]["gpu_memory_fraction"] == 0.6
