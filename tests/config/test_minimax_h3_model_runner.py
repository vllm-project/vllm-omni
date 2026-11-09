# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from pathlib import Path

import pytest

from vllm_omni.config.stage_config import load_deploy_config, resolve_stage_model_runner

_DEPLOY = Path(__file__).resolve().parents[2] / "vllm_omni" / "deploy"

_CONFIGS = (
    "minimax_h3_disaggregated.yaml",
    "minimax_h3_disaggregated_decode.yaml",
    "minimax_h3_disaggregated_turbo.yaml",
    "minimax_h3_disaggregated_nixl.yaml",
    "minimax_h3_disaggregated_turbo_nixl.yaml",
)


@pytest.mark.core_model
@pytest.mark.cpu
@pytest.mark.parametrize("name", _CONFIGS)
def test_h3_encoder_stage_selects_model_runner_v2(name: str) -> None:
    """The AR encoder uses model runner v2. Diffusion and VAE stages do not."""
    deploy = load_deploy_config(_DEPLOY / name)
    assert resolve_stage_model_runner(deploy, deploy.stages[0]) == "v2"
    for stage in deploy.stages[1:]:
        assert resolve_stage_model_runner(deploy, stage) == "v1"
