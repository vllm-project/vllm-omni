# SPDX-License-Identifier: Apache-2.0
"""Moshi TTS pipeline: DSM Talker (text → RVQ codec) → Mimi Decoder (codec → audio)."""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

_PROC = "vllm_omni.model_executor.stage_input_processors.moshi_tts"

MOSHI_TTS_PIPELINE = PipelineConfig(
    model_type="moshi",
    model_arch="MoshiTTSTalkerForConditionalGeneration",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="moshi_tts_talker",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            owns_tokenizer=True,
            engine_output_type="latent",
            async_chunk_process_next_stage_input_func=f"{_PROC}.moshi_tts_to_mimi_async_chunk",
            sync_process_input_func=f"{_PROC}.moshi_tts_to_mimi",
            sampling_constraints={
                "detokenize": False,
                "stop_token_ids": [1],
            },
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="mimi_decoder",
            model_arch="MoshiMimiDecoder",
            execution_type=StageExecutionType.LLM_GENERATION,
            input_sources=(0,),
            final_output=True,
            final_output_type="audio",
            engine_output_type="audio",
            sampling_constraints={"detokenize": True},
        ),
    ),
)
