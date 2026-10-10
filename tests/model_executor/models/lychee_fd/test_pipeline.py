# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from pathlib import Path

import pytest

from vllm_omni.config.pipeline_registry import OMNI_PIPELINES
from vllm_omni.config.stage_config import PipelineConfig, StageExecutionType, load_deploy_config

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_lychee_pipeline_declares_unified_duplex_plugin() -> None:
    pipeline = OMNI_PIPELINES["step_audio_2_full_duplex"]

    assert isinstance(pipeline, PipelineConfig)
    assert pipeline.duplex_plugin == ("vllm_omni.model_executor.models.lychee_fd.duplex.plugin.LycheeDuplexPlugin")
    assert pipeline.default_deploy_config_name == "lychee_fd_single_gpu.yaml"
    assert len(pipeline.stages) == 2
    stage = pipeline.stages[0]
    assert stage.execution_type == StageExecutionType.LLM_AR
    assert stage.model_stage == "lychee_fd"
    assert stage.final_output is True
    assert stage.final_output_type == "text"
    assert stage.supports_native_preemption is False
    assert stage.supports_running_prefix_cache_reset is False
    synthesis = pipeline.stages[1]
    assert synthesis.execution_type is StageExecutionType.LLM_GENERATION
    assert synthesis.input_sources == (stage.stage_id,)
    assert synthesis.model_arch == "LycheeToken2WavForConditionalGeneration"
    assert synthesis.final_output is True
    assert synthesis.final_output_type == synthesis.engine_output_type == "audio"
    assert synthesis.sync_process_input_func == (
        "vllm_omni.model_executor.stage_input_processors.lychee_fd.lychee2token2wav"
    )


def test_lychee_default_deploy_is_single_gpu_duplex_mrv2() -> None:
    deploy_path = Path(__file__).resolve().parents[4] / "vllm_omni" / "deploy" / "lychee_fd_single_gpu.yaml"
    deploy = load_deploy_config(deploy_path)

    assert deploy.pipeline == "step_audio_2_full_duplex"
    assert deploy.session_mode == "duplex"
    assert deploy.model_runner == "v2"
    assert deploy.async_chunk is False
    assert deploy.duplex_session.max_sessions == 1
    assert len(deploy.stages) == 2
    assert deploy.stages[0].devices == "0"
    assert deploy.stages[0].tensor_parallel_size == 1
    assert deploy.stages[0].max_num_seqs == 1
    assert deploy.stages[0].supports_native_preemption is False
    assert deploy.stages[0].supports_running_prefix_cache_reset is False
    assert deploy.stages[0].engine_extras["enable_prefix_caching"] is False

    assert deploy.stages[1].devices == deploy.stages[0].devices
    assert deploy.stages[1].tensor_parallel_size == 1
    assert deploy.stages[1].max_num_seqs == 1
    assert deploy.stages[1].enforce_eager is True
    assert deploy.stages[0].gpu_memory_utilization + deploy.stages[1].gpu_memory_utilization < 1.0
