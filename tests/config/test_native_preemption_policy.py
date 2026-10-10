# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
from pathlib import Path

import pytest

from vllm_omni.config.omni_config import VllmOmniConfig
from vllm_omni.config.pipeline_registry import OMNI_PIPELINES
from vllm_omni.config.stage_config import load_deploy_config
from vllm_omni.engine.stage_init_utils import build_engine_args_dict_from_omni_stage_config

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_real_lychee_pipeline_yaml_resolves_preemption_capabilities_with_structured_owner():
    pipeline = OMNI_PIPELINES["step_audio_2_full_duplex"]
    path = Path(__file__).parents[2] / "vllm_omni" / "deploy" / "lychee_fd_single_gpu.yaml"
    deploy = load_deploy_config(path)
    assert deploy.stages[0].supports_native_preemption is False
    assert deploy.stages[0].supports_running_prefix_cache_reset is False
    config = VllmOmniConfig.from_pipeline_config(pipeline, cli_overrides={"model": "/models/Lychee-FD"})
    stage = config.stage_by_id(0)
    assert stage.model_config.supports_native_preemption is False
    assert stage.model_config.supports_running_prefix_cache_reset is False
    assert stage.cache_config.enable_prefix_caching is False
    engine = build_engine_args_dict_from_omni_stage_config(stage, model="/models/Lychee-FD")
    assert engine["supports_native_preemption"] is False
    assert engine["supports_running_prefix_cache_reset"] is False
    assert engine["enable_prefix_caching"] is False
