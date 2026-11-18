# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from dataclasses import replace

import pytest
from vllm.pooling_params import PoolingParams

import vllm_omni.config.pipeline_registry  # noqa: F401  (populate registry)
from vllm_omni.config.omni_config import VllmOmniConfig
from vllm_omni.config.pipeline_registry import OMNI_PIPELINES as _PIPELINE_REGISTRY
from vllm_omni.config.stage_config import (
    DeployConfig,
    StageExecutionType,
    StagePipelineConfig,
)
from vllm_omni.engine.stage_init_utils import (
    build_engine_args_dict_from_omni_stage_config,
    extract_stage_metadata_from_omni_stage_config,
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


def test_injected_aligner_survives_typed_config_and_runtime_projections():
    pipeline, deploy = inject_forced_aligner_stage(
        _PIPELINE_REGISTRY["qwen3_tts"],
        DeployConfig(),
        {"forced_aligner": "/models/Qwen3-ForcedAligner-0.6B"},
    )
    config = VllmOmniConfig.from_pipeline_config(
        pipeline,
        user_deploy_config=deploy,
        cli_overrides={"model": "/models/Qwen3-TTS"},
    )
    stage = config.stage_configs[-1]

    assert stage.pooling_config.runner == "pooling"
    assert stage.pooling_config.pooling_output_decoder == POOLING_OUTPUT_DECODER_PATH
    assert isinstance(stage.pooling_config.default_pooling_params, PoolingParams)
    assert stage.pooling_config.default_pooling_params.task == deploy.stages[-1].default_pooling_params["task"]

    metadata = extract_stage_metadata_from_omni_stage_config(stage)
    assert isinstance(metadata.default_sampling_params, PoolingParams)
    assert metadata.default_sampling_params.task == deploy.stages[-1].default_pooling_params["task"]

    engine_args = build_engine_args_dict_from_omni_stage_config(
        stage,
        stage.model_config.model or "/models/Qwen3-ForcedAligner-0.6B",
    )
    assert engine_args["runner"] == "pooling"
    assert engine_args["pooling_output_decoder"] == POOLING_OUTPUT_DECODER_PATH


def test_qwen3_asr_registers_transcript_pipeline_and_injects_asr_aligner():
    pipeline = _PIPELINE_REGISTRY["qwen3_asr"]
    assert len(pipeline.stages) == 1
    assert pipeline.default_deploy_config_name == "qwen3_asr.yaml"
    assert pipeline.stages[0].model_arch == "Qwen3ASRForConditionalGeneration"
    assert pipeline.stages[0].final_output_type == "text"

    extended, deploy = inject_forced_aligner_stage(
        pipeline, DeployConfig(), {"forced_aligner": "/models/Qwen3-ForcedAligner-0.6B"}
    )
    assert len(extended.stages) == 2
    assert extended.stages[0].final_output is True
    assert extended.stages[-1].custom_process_input_func.endswith("asr2aligner")
    assert extended.stages[-1].input_sources == (0,)
    assert extended.stages[-1].final_output_type == TIMESTAMPS_MODALITY
    assert deploy.stages[-1].default_pooling_params == {"task": "token_classify"}
    assert deploy.stages[-1].async_scheduling is False
    assert deploy.stages[-1].async_chunk is False


def test_qwen3_asr_aligner_config_preserves_aligner_model_and_device(tmp_path):
    from vllm_omni.config.config_factory import StageConfigFactory

    config_path = tmp_path / "aligner.yaml"
    config_path.write_text("forced_aligner:\n  model: /models/Qwen3-ForcedAligner-0.6B\n")
    resolution = StageConfigFactory._resolve_legacy_from_registry(
        _PIPELINE_REGISTRY["qwen3_asr"],
        {
            "model": "/models/Qwen3-ASR-1.7B",
            "forced_aligner_config": str(config_path),
            "forced_aligner_device": "1",
        },
        user_deploy_config=DeployConfig(async_chunk=False),
    )
    stages = [stage.to_omegaconf() for stage in resolution.stage_configs]
    assert len(stages) == 2
    assert stages[0].engine_args.model_arch == "Qwen3ASRForConditionalGeneration"
    assert stages[1].engine_args.model == "/models/Qwen3-ForcedAligner-0.6B"
    assert stages[1].engine_args.runner == "pooling"
    assert stages[1].runtime.devices == "1"
    assert stages[1].custom_process_input_func.endswith("asr2aligner")


@pytest.mark.parametrize("model_type", ["whisper", "another_asr"])
def test_asr_aligner_injection_uses_topology_instead_of_a_model_family_name(model_type):
    pipeline = replace(_PIPELINE_REGISTRY["qwen3_asr"], model_type=model_type)
    extended, _ = inject_forced_aligner_stage(pipeline, DeployConfig(), {"forced_aligner": "aligner"})
    assert extended.stages[-1].custom_process_input_func.endswith("asr2aligner")


def test_tts_aligner_injection_uses_the_audio_terminal_instead_of_a_model_family_name():
    pipeline = replace(_PIPELINE_REGISTRY["qwen3_tts"], model_type="another_tts")
    extended, _ = inject_forced_aligner_stage(pipeline, DeployConfig(), {"forced_aligner": "aligner"})
    assert extended.stages[-1].custom_process_input_func.endswith("code2wav2aligner")


@pytest.mark.parametrize("topology", ["image", "nonterminal_audio"])
def test_forced_aligner_rejects_unsupported_topology(topology):
    pipeline = _PIPELINE_REGISTRY["qwen3_asr"]
    entry = pipeline.stages[0]
    stages: tuple[StagePipelineConfig, ...]
    if topology == "image":
        stages = (replace(entry, final_output_type="image"),)
    else:
        stages = (entry, replace(entry, stage_id=1, input_sources=(0,), final_output=False, final_output_type="audio"))
    with pytest.raises(ValueError, match="Forced alignment requires"):
        inject_forced_aligner_stage(replace(pipeline, stages=stages), DeployConfig(), {"forced_aligner": "aligner"})


@pytest.mark.parametrize("prefix_stages", [1, 3])
def test_asr_aligner_can_follow_a_multi_stage_pipeline(prefix_stages):
    pipeline = _PIPELINE_REGISTRY["qwen3_asr"]
    source = pipeline.stages[0]
    prefix = tuple(
        replace(
            source,
            stage_id=i,
            input_sources=() if i == 0 else (i - 1,),
            final_output=False,
        )
        for i in range(prefix_stages)
    )
    pipeline = replace(
        pipeline, stages=(*prefix, replace(source, stage_id=prefix_stages, input_sources=(prefix_stages - 1,)))
    )
    extended, deploy = inject_forced_aligner_stage(pipeline, DeployConfig(), {"forced_aligner": "aligner"})

    assert extended.stages[:-1] == pipeline.stages
    assert extended.stages[-1].input_sources == (prefix_stages,)
    assert extended.stages[-1].custom_process_input_func.endswith("asr2aligner")
    assert deploy.stages[-1].stage_id == prefix_stages + 1
