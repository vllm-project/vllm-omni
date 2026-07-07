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
    hf_architectures=("MingTTSForConditionalGeneration", "BailingMMNativeForConditionalGeneration"),
    # The dense (0.5B) and MoE (16.8B) share architectures=["BailingMMNativeForConditionalGeneration"]
    # Here we disambiguate by the model_type the upstream HF config reports
    hf_config_predicate=lambda c: str(getattr(c, "model_type", "")) == "dense",
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


# MoE variant (inclusionAI/Ming-omni-tts-16.8B-A3B)
# Same two-stage topology and model class as the above,
# but with a distinct model_type reported by the upstream HF config.
# Keep this for auto-detection of deploy config yaml.
MING_TTS_MOE_PIPELINE = PipelineConfig(
    model_type="ming_tts_moe",
    model_arch="MingTTSForConditionalGeneration",
    hf_architectures=("MingTTSForConditionalGeneration", "BailingMMNativeForConditionalGeneration"),
    # Disambiguate from the dense-0.5B ming_tts pipeline
    hf_config_predicate=lambda c: str(getattr(c, "model_type", "")) == "bailingmm",
    stages=MING_TTS_PIPELINE.stages,
)
