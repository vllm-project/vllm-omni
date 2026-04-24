# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3-TTS pipeline: Talker (text → RVQ codec) → Code2Wav (codec → audio).

Chunked vs end-to-end mode is dispatched from ``deploy.async_chunk``.
"""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

_PROC = "vllm_omni.model_executor.stage_input_processors.qwen3_tts"

QWEN3_TTS_PIPELINE = PipelineConfig(
    model_type="qwen3_tts",
    # Pipeline-level default; the code2wav stage overrides per-stage below.
    model_arch="Qwen3TTSTalkerForConditionalGeneration",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="qwen3_tts",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            owns_tokenizer=True,
            engine_output_type="latent",
            async_chunk_process_next_stage_input_func=(f"{_PROC}.talker2code2wav_async_chunk"),
            sampling_constraints={
                "detokenize": False,
                "stop_token_ids": [2150],
            },
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="code2wav",
            execution_type=StageExecutionType.LLM_GENERATION,
            input_sources=(0,),
            final_output=True,
            final_output_type="audio",
            engine_output_type="audio",
            model_arch="Qwen3TTSCode2Wav",
            sync_process_input_func=f"{_PROC}.talker2code2wav",
            sampling_constraints={"detokenize": True},
            extras={"tts_args": {"max_instructions_length": 500}},
        ),
    ),
)
