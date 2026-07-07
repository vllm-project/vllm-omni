# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Miso TTS pipeline: talker (``generate_frame`` / MisoLabs Model) → Mimi decode.

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

_PROC = "vllm_omni.model_executor.stage_input_processors.miso_tts"

MISO_TTS_PIPELINE = PipelineConfig(
    model_type="miso_tts",
    model_arch="MisoTTSTalkerForConditionalGeneration",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="miso_tts",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            owns_tokenizer=False,
            engine_output_type="latent",
            async_chunk_process_next_stage_input_func=f"{_PROC}.talker2mimi_async_chunk",
            custom_process_next_stage_input_func=f"{_PROC}.talker2mimi_full_payload",
            sampling_constraints={
                "detokenize": False,
                "stop_token_ids": [2],
            },
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="mimi",
            execution_type=StageExecutionType.LLM_GENERATION,
            input_sources=(0,),
            final_output=True,
            final_output_type="audio",
            engine_output_type="audio",
            model_arch="MisoTTSMimiDecoder",
            # Stage-1 needs at least one placeholder token per scheduling step to
            # run forward() and consume async connector payloads.
            custom_process_input_func=f"{_PROC}.talker2mimi_token_only",
            sync_process_input_func=f"{_PROC}.talker2mimi_token_only",
            sampling_constraints={"detokenize": True},
        ),
    ),
)

MISO_TTS_SINGLE_STAGE_PIPELINE = PipelineConfig(
    model_type="miso_tts",
    model_arch="MisoTTSSingleStageForVLLM",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="miso_tts",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            owns_tokenizer=False,
            final_output=True,
            final_output_type="audio",
            engine_output_type="audio",
            sampling_constraints={
                "detokenize": False,
            },
        ),
    ),
)
