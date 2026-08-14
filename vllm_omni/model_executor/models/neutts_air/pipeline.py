# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""NeuTTS-Air pipeline: Qwen2 speech-token generation to NeuCodec audio."""

from typing import Any

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

_PROC = "vllm_omni.model_executor.stage_input_processors.neutts_air"
_SPEECH_GENERATION_END_TOKEN_ID = 151670

# NeuTTS-Air publishes a generic qwen2/Qwen2ForCausalLM config without a
# model-specific marker. Use its full architecture fingerprint so unrelated
# Qwen2 checkpoints are not routed into this TTS pipeline.
_CONFIG_FINGERPRINT: dict[str, Any] = {
    "model_type": "qwen2",
    "vocab_size": 217652,
    "hidden_size": 896,
    "num_hidden_layers": 24,
    "num_attention_heads": 14,
    "num_key_value_heads": 2,
}


def is_neutts_air_config(hf_config: Any) -> bool:
    """Return whether an HF config matches the NeuTTS-Air backbone."""
    return all(getattr(hf_config, field, None) == expected for field, expected in _CONFIG_FINGERPRINT.items())


NEUTTS_AIR_PIPELINE = PipelineConfig(
    model_type="neutts_air",
    default_deploy_config_name="neutts_air.yaml",
    model_arch="NeuTTSAirForCausalLM",
    hf_architectures=("Qwen2ForCausalLM",),
    hf_config_predicate=is_neutts_air_config,
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="backbone",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            owns_tokenizer=True,
            engine_output_type="latent",
            async_chunk_process_next_stage_input_func=(f"{_PROC}.llm2neucodec_async_chunk"),
            sampling_constraints={
                "detokenize": False,
                "stop_token_ids": [_SPEECH_GENERATION_END_TOKEN_ID],
            },
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="neucodec",
            execution_type=StageExecutionType.LLM_GENERATION,
            input_sources=(0,),
            final_output=True,
            final_output_type="audio",
            engine_output_type="audio",
            model_arch="NeuTTSAirCode2Wav",
            sync_process_input_func=f"{_PROC}.llm2neucodec_sync",
            sampling_constraints={"detokenize": False},
        ),
    ),
)
