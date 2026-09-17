# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest

import vllm_omni.config.pipeline_registry  # noqa: F401  (populate registry)
from vllm_omni.config.pipeline_registry import OMNI_PIPELINES as _PIPELINE_REGISTRY
from vllm_omni.config.stage_config import (
    DeployConfig,
    StageExecutionType,
)
from vllm_omni.model_executor.stage_input_processors.forced_aligner import (
    POOLING_OUTPUT_DECODER_PATH,
    TIMESTAMPS_MODALITY,
)
from vllm_omni.utils.forced_aligner import _DEFAULT_CONFIG_PATH, inject_forced_aligner_stage

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_inject_appends_pooling_aligner_stage():
    pipeline = _PIPELINE_REGISTRY["qwen3_tts"]
    deploy = DeployConfig()
    n0 = len(pipeline.stages)

    ext_pipeline, ext_deploy = inject_forced_aligner_stage(
        pipeline, deploy, {"forced_aligner": "/models/Qwen3-ForcedAligner-0.6B"}
    )

    # A new pooling stage is appended; the original pipeline is unchanged.
    assert len(pipeline.stages) == n0
    assert len(ext_pipeline.stages) == n0 + 1
    # The original deploy config is not mutated either (double-injection safe).
    assert len(deploy.stages) == 0
    assert len(ext_deploy.stages) == 1
    aligner = ext_pipeline.stages[-1]
    # Pooling stage reuses LLM_AR (+ runner=pooling), no dedicated LLM_POOLING.
    assert aligner.execution_type == StageExecutionType.LLM_AR
    assert aligner.final_output is True
    assert aligner.final_output_type == TIMESTAMPS_MODALITY
    assert aligner.input_sources == (n0 - 1,)
    assert aligner.requires_multimodal_data is True
    assert aligner.owns_tokenizer is True

    # The deploy stage carries the separate aligner checkpoint + decoder hook.
    ds = ext_deploy.stages[-1]
    assert ds.engine_extras["model"] == "/models/Qwen3-ForcedAligner-0.6B"
    assert ds.engine_extras["runner"] == "pooling"
    assert ds.engine_extras["pooling_output_decoder"] == POOLING_OUTPUT_DECODER_PATH


def test_inject_from_config_only():
    # --forced-aligner-config alone (model in the YAML) must also inject.
    pipeline = _PIPELINE_REGISTRY["qwen3_tts"]
    deploy = DeployConfig()
    n0 = len(pipeline.stages)

    ext_pipeline, ext_deploy = inject_forced_aligner_stage(
        pipeline, deploy, {"forced_aligner": None, "forced_aligner_config": str(_DEFAULT_CONFIG_PATH)}
    )

    assert len(ext_pipeline.stages) == n0 + 1
    assert ext_pipeline.stages[-1].final_output_type == TIMESTAMPS_MODALITY
    assert ext_deploy.stages[-1].engine_extras["model"] == "Qwen/Qwen3-ForcedAligner-0.6B"


def test_inject_noop_without_forced_aligner():
    pipeline = _PIPELINE_REGISTRY["qwen3_tts"]
    deploy = DeployConfig()

    ext_pipeline, ext_deploy = inject_forced_aligner_stage(pipeline, deploy, {})

    assert ext_pipeline is pipeline
    assert len(ext_deploy.stages) == 0


@pytest.mark.parametrize("async_chunk", [True, False])
@pytest.mark.parametrize("cli_async_chunk", [None, True, False])
def test_aligner_uses_completed_audio_without_disabling_upstream_chunks(async_chunk, cli_async_chunk):
    from vllm_omni.config.config_factory import StageConfigFactory

    pipeline = _PIPELINE_REGISTRY["qwen3_tts"]
    deploy = DeployConfig(async_chunk=async_chunk)
    resolution = StageConfigFactory._resolve_legacy_from_registry(
        pipeline,
        {"forced_aligner": "/models/Qwen3-ForcedAligner-0.6B", "async_chunk": cli_async_chunk},
        user_deploy_config=deploy,
    )
    stages = [stage.to_omegaconf() for stage in resolution.stage_configs]
    expected_async = async_chunk if cli_async_chunk is None else cli_async_chunk
    assert len(stages) == 3
    assert [stage.engine_args.async_chunk for stage in stages] == [expected_async, expected_async, False]
    assert stages[2].custom_process_input_func.endswith("code2wav2aligner")
    assert stages[2].engine_args.runner == "pooling"
    assert len(pipeline.stages) == 2
    assert deploy.stages == []
