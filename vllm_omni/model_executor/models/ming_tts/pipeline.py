# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ming TTS pipeline: Stage-0 LLM+flow -> Stage-1 audio VAE."""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

_PROC = "vllm_omni.model_executor.stage_input_processors.ming_tts"

MING_TTS_PIPELINE = PipelineConfig(
    model_type="ming_tts",
    model_arch="MingTTSForConditionalGeneration",
    hf_architectures=("MingTTSForConditionalGeneration",),
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="llm",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            owns_tokenizer=True,
            hf_config_name="llm_config",
            engine_output_type="latent",
            async_chunk_process_next_stage_input_func=(f"{_PROC}.llm2audio_vae_async_chunk"),
            sampling_constraints={
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": -1,
                "max_tokens": 512,
                "detokenize": True,
            },
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="audio_vae",
            execution_type=StageExecutionType.LLM_GENERATION,
            input_sources=(0,),
            final_output=True,
            final_output_type="audio",
            hf_config_name="llm_config",
            engine_output_type="audio",
            sync_process_input_func=f"{_PROC}.llm2audio_vae",
            sampling_constraints={
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": -1,
                "max_tokens": 1,
                "detokenize": False,
            },
        ),
    ),
)
