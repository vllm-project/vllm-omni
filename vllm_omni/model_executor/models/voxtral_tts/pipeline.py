# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Voxtral TTS pipeline topology (frozen).

Stage 0: audio_generation  — text → acoustic latents (LLM_AR, tokenizer owner).
Stage 1: audio_tokenizer   — acoustic latents → waveform (LLM_GENERATION).
"""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

_PROC = "vllm_omni.model_executor.stage_input_processors.voxtral_tts"

VOXTRAL_TTS_PIPELINE = PipelineConfig(
    model_type="voxtral_tts",
    model_arch="VoxtralTTSForConditionalGeneration",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="audio_generation",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            final_output=False,
            final_output_type="text",
            owns_tokenizer=True,
            engine_output_type="latent",
            async_chunk_process_next_stage_input_func=(f"{_PROC}.generator2tokenizer_async_chunk"),
            sampling_constraints={"detokenize": True},
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="audio_tokenizer",
            execution_type=StageExecutionType.LLM_GENERATION,
            input_sources=(0,),
            final_output=True,
            final_output_type="audio",
            engine_output_type="audio",
            sampling_constraints={"detokenize": True},
            extras={"tts_args": {"max_instructions_length": 500}},
        ),
    ),
)
