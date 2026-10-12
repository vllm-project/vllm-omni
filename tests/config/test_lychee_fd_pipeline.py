# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from pathlib import Path

import pytest

from vllm_omni.config.pipeline_registry import resolve_pipeline_config
from vllm_omni.config.stage_config import StageExecutionType, load_deploy_config

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_lychee_checkpoint_model_type_resolves_native_ar_pipeline() -> None:
    pipeline = resolve_pipeline_config("step_audio_2_full_duplex")
    assert pipeline is not None
    assert pipeline.model_arch == "LycheeFullDuplexForConditionalGeneration"
    assert len(pipeline.stages) == 2
    stage = pipeline.stages[0]
    assert stage.execution_type is StageExecutionType.LLM_AR
    assert stage.model_stage == "lychee_fd"
    assert stage.engine_output_type == "latent"


def test_lychee_correctness_profile_is_single_gpu_mrv2() -> None:
    root = Path(__file__).parents[2]
    deploy = load_deploy_config(root / "vllm_omni" / "deploy" / "lychee_fd_single_gpu.yaml")
    assert deploy.pipeline == "step_audio_2_full_duplex"
    assert deploy.model_runner == "v2"
    assert len(deploy.stages) == 2
    stage = deploy.stages[0]
    assert stage.devices == "0"
    assert stage.tensor_parallel_size == 1
    assert stage.max_num_seqs == 1
    assert stage.enforce_eager is True


def test_lychee_native_synthesis_stage_uses_same_gpu_b1():
    pipeline = resolve_pipeline_config("step_audio_2_full_duplex")
    stage = pipeline.stages[1]
    assert stage.execution_type is StageExecutionType.LLM_GENERATION
    assert stage.input_sources == (0,)
    assert stage.model_arch == "LycheeToken2WavForConditionalGeneration"
    assert stage.final_output_type == "audio"
    deploy = load_deploy_config(Path(__file__).parents[2] / "vllm_omni/deploy/lychee_fd_single_gpu.yaml")
    assert deploy.stages[1].devices == deploy.stages[0].devices == "0"
    assert deploy.stages[1].max_num_seqs == 1
    assert deploy.stages[1].enforce_eager
