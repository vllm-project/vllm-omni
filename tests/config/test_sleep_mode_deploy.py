# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Check the sleep/wake E2E topology before loading large BAGEL weights."""

import pytest

from tests.entrypoints.test_omni_sleep_mode import _sleep_deploy_config
from vllm_omni.config.omni_config import (
    VllmOmniConfig,
    VllmOmniDiffusionStageConfig,
    extract_diffusion_stage_config_kwargs,
)
from vllm_omni.config.pipeline_registry import resolve_pipeline_config
from vllm_omni.config.stage_config import load_deploy_config
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.engine.stage_init_utils import build_engine_args_dict_from_omni_stage_config

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("diffusion_only,tp_size", [(False, 1), (False, 2), (True, 2)])
def test_sleep_deploy_resolves_expected_stages(diffusion_only, tp_size):
    deploy = load_deploy_config(_sleep_deploy_config(diffusion_only=diffusion_only, tp_size=tp_size))
    pipeline = resolve_pipeline_config(deploy.pipeline or "bagel")
    config = VllmOmniConfig.from_pipeline_config(pipeline, user_deploy_config=deploy)

    assert len(config.stage_configs) == (1 if diffusion_only else 2)
    for stage in config.stage_configs:
        is_diffusion = diffusion_only or stage.stage_id == 1
        assert isinstance(stage, VllmOmniDiffusionStageConfig) == is_diffusion
        # Keep BAGEL's batch budget: a single multimodal item needs 8625 tokens.
        assert stage.scheduler_config.max_num_batched_tokens == 32768
        assert stage.model_config.enable_sleep_mode is True
        assert stage.parallel_config.tensor_parallel_size == tp_size
        assert stage.runtime_config.devices == ",".join(str(stage.stage_id * tp_size + rank) for rank in range(tp_size))
        if not is_diffusion:
            assert stage.cache_config.gpu_memory_utilization == 0.8
        if is_diffusion:
            engine_args = build_engine_args_dict_from_omni_stage_config(stage, "test-bagel")
            terminal = OmniDiffusionConfig.from_kwargs(
                **extract_diffusion_stage_config_kwargs(
                    engine_args, stage_id=stage.stage_id, include_engine_adapter_metadata=True
                )
            )
            assert terminal.enable_sleep_mode is True
            assert terminal.parallel_config.tensor_parallel_size == tp_size

    if not diffusion_only:
        assert config.stage_by_id(1).stage_pipeline_config.input_sources == (0,)
        assert deploy.stages[1].input_connectors["from_stage_0"] == "shared_memory_connector"
