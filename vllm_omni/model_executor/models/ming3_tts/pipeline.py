# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

_PROCESSOR = "vllm_omni.model_executor.stage_input_processors.ming3_tts"

MING3_TTS_PIPELINE = PipelineConfig(
    model_type="ming3_tts",
    default_deploy_config_name="ming3_tts.yaml",
    model_arch="Ming3TTSForConditionalGeneration",
    hf_architectures=("Ming3TTSForConditionalGeneration", "LingV3TTSForConditionalGeneration"),
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="llm",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            owns_tokenizer=True,
            hf_config_name="llm_config",
            engine_output_type="latent",
            async_chunk_process_next_stage_input_func=(f"{_PROCESSOR}.llm2audio_vae_async_chunk"),
            sampling_constraints={
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": -1,
                "max_tokens": 801,
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
            sync_process_input_func=f"{_PROCESSOR}.llm2audio_vae",
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

__all__ = ["MING3_TTS_PIPELINE"]
