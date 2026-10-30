# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

# Stage 0 decodes its latents with the AudioVAE decoder
# and returns PCM as the final output.
MING3_TTS_PIPELINE = PipelineConfig(
    model_type="ming3_tts",
    default_deploy_config_name="ming3_tts.yaml",
    single_stage_async_chunk=True,
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
            final_output=True,
            final_output_type="audio",
            # The API output processor tags ``model_outputs`` by this modality.
            engine_output_type="audio",
            sampling_constraints={
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": -1,
                "max_tokens": 801,
                "detokenize": True,
            },
        ),
    ),
)

__all__ = ["MING3_TTS_PIPELINE"]
