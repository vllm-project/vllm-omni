# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from pathlib import Path

import pytest
from vllm.model_executor.models.interfaces import supports_multimodal

from vllm_omni.config.pipeline_registry import OMNI_PIPELINES, resolve_pipeline_config
from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    load_deploy_config,
    merge_pipeline_deploy,
)
from vllm_omni.model_executor.models.llama_omni2.llama_omni2 import (
    Omni2Speech2SQwen2ForCausalLM,
)
from vllm_omni.model_executor.models.registry import _OMNI_MODELS

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_llama_omni2_pipeline_has_native_three_stage_topology():
    pipeline = resolve_pipeline_config("omni2_speech2s_qwen2")

    assert isinstance(pipeline, PipelineConfig)
    assert pipeline.model_arch == "Omni2Speech2SQwen2ForCausalLM"
    assert pipeline.default_deploy_config_name == "llama_omni2.yaml"
    assert [stage.execution_type for stage in pipeline.stages] == [
        StageExecutionType.LLM_AR,
        StageExecutionType.LLM_AR,
        StageExecutionType.LLM_GENERATION,
    ]

    thinker, talker, code2wav = pipeline.stages
    assert thinker.model_stage == "thinker"
    assert thinker.owns_tokenizer is True
    assert thinker.requires_multimodal_data is True
    assert thinker.final_output is True
    assert thinker.final_output_type == "text"
    assert thinker.hf_config_name == "thinker_config"
    assert thinker.engine_output_type == "latent"
    assert thinker.custom_process_next_stage_input_func.endswith("thinker2talker_full_payload")
    assert thinker.async_chunk_process_next_stage_input_func.endswith("thinker2talker_async_chunk")

    assert talker.model_stage == "talker"
    assert talker.input_sources == (0,)
    assert talker.hf_config_name == "talker_config"
    assert talker.engine_output_type == "latent"
    assert talker.sync_process_input_func.endswith("thinker2talker_token_only")
    assert talker.custom_process_next_stage_input_func.endswith("talker2code2wav_full_payload")
    assert talker.async_chunk_process_next_stage_input_func.endswith("talker2code2wav_async_chunk")
    assert talker.sampling_constraints["detokenize"] is False
    assert talker.sampling_constraints["stop_token_ids"] == [151643]

    assert code2wav.model_stage == "code2wav"
    assert code2wav.input_sources == (1,)
    assert code2wav.final_output is True
    assert code2wav.final_output_type == "audio"
    assert code2wav.engine_output_type == "audio"
    assert code2wav.hf_config_name is None
    assert code2wav.model_arch == "LlamaOmni2Code2Wav"


def test_llama_omni2_pipeline_and_architectures_are_registered():
    assert "omni2_speech2s_qwen2" in OMNI_PIPELINES
    assert _OMNI_MODELS["Omni2Speech2SQwen2ForCausalLM"] == (
        "llama_omni2",
        "llama_omni2",
        "Omni2Speech2SQwen2ForCausalLM",
    )
    assert _OMNI_MODELS["LlamaOmni2ThinkerForConditionalGeneration"] == (
        "llama_omni2",
        "llama_omni2_thinker",
        "LlamaOmni2ThinkerForConditionalGeneration",
    )
    assert _OMNI_MODELS["LlamaOmni2TalkerForConditionalGeneration"] == (
        "llama_omni2",
        "llama_omni2_talker",
        "LlamaOmni2TalkerForConditionalGeneration",
    )
    assert _OMNI_MODELS["LlamaOmni2Code2Wav"] == (
        "llama_omni2",
        "llama_omni2_code2wav",
        "LlamaOmni2Code2Wav",
    )


def test_llama_omni2_outer_architecture_registers_multimodal_processor():
    assert supports_multimodal(Omni2Speech2SQwen2ForCausalLM)
    assert hasattr(Omni2Speech2SQwen2ForCausalLM, "_processor_factory")


def test_llama_omni2_default_deploy_uses_independent_decoder_checkpoint():
    deploy_path = Path(__file__).parents[4] / "vllm_omni" / "deploy" / "llama_omni2.yaml"

    deploy = load_deploy_config(deploy_path)
    pipeline = resolve_pipeline_config("omni2_speech2s_qwen2")
    assert isinstance(pipeline, PipelineConfig)
    stages = merge_pipeline_deploy(
        pipeline,
        deploy,
        {"model": "ICTNLP/LLaMA-Omni2-0.5B"},
    )
    for stage in stages:
        stage.runtime_overrides = {"model": "ICTNLP/LLaMA-Omni2-0.5B"}

    assert deploy.async_chunk is True
    assert [stage.devices for stage in deploy.stages] == ["0", "1", "1"]
    assert deploy.stages[0].model is None
    assert deploy.stages[1].model is None
    assert deploy.stages[2].model == "ICTNLP/cosy2_decoder"
    assert stages[0].yaml_engine_args["async_scheduling"] is True
    assert stages[1].yaml_engine_args["async_scheduling"] is False
    assert stages[1].yaml_extras["default_sampling_params"]["stop_token_ids"] == [151643]
    assert stages[2].yaml_engine_args["async_scheduling"] is False
    assert stages[2].yaml_engine_args["dtype"] == "float32"
    assert stages[2].yaml_engine_args["enforce_eager"] is True
    assert stages[2].yaml_engine_args["enable_chunked_prefill"] is False
    assert stages[2].yaml_engine_args["max_num_seqs"] >= 8
    assert stages[0].to_omegaconf().engine_args.model == "ICTNLP/LLaMA-Omni2-0.5B"
    assert stages[1].to_omegaconf().engine_args.model == "ICTNLP/LLaMA-Omni2-0.5B"
    assert stages[2].to_omegaconf().engine_args.model == "ICTNLP/cosy2_decoder"
    assert stages[0].yaml_engine_args["custom_process_next_stage_input_func"].endswith("thinker2talker_async_chunk")
    assert stages[1].custom_process_input_func is None
    assert stages[1].yaml_engine_args["custom_process_next_stage_input_func"].endswith("talker2code2wav_async_chunk")
