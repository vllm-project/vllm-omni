# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""GLM-TTS pipeline: Stage 0 (AR) → Stage 1 (DiT)."""

from dataclasses import replace

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)
from vllm_omni.serving_capabilities import (
    AUDIO_RESPONSE,
    TEXT_TO_SPEECH,
    declare_capabilities,
)

_PROC = "vllm_omni.model_executor.stage_input_processors.glm_tts"

GLM_TTS_PIPELINE = PipelineConfig(
    serving_capabilities=declare_capabilities(
        TEXT_TO_SPEECH,
        replace(AUDIO_RESPONSE, support="unknown", reason_code="chat_prompt_contract_undeclared"),
        unknown_reason="chat_prompt_contract_undeclared",
        sources=("vllm_omni/model_executor/models/glm_tts/pipeline.py", "docs/serving/speech_api.md"),
    ),
    model_type="glm_tts",
    default_deploy_config_name="glm_tts.yaml",
    model_arch="GLMTTSForConditionalGeneration",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="glm_tts",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            owns_tokenizer=True,
            engine_output_type="latent",
            async_chunk_process_next_stage_input_func=(f"{_PROC}.ar_to_dit_async_chunk"),
            sampling_constraints={
                # GLM-TTS uses 👂 (token string "👂", ID 59253) as
                # end-of-audio marker.  The ID is resolved dynamically by
                # GLMTTSForConditionalGeneration.__init__ from the tokenizer and
                # validated at runtime against this hardcoded value.  If the
                # upstream checkpoint changes the mapping, the model will log a
                # warning so the constant here can be updated.
                "stop_token_ids": [59253],
            },
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="glm_tts_dit",
            execution_type=StageExecutionType.LLM_GENERATION,
            input_sources=(0,),
            final_output=True,
            final_output_type="audio",
            engine_output_type="latent",
            sync_process_input_func=f"{_PROC}.ar_to_dit",
        ),
    ),
)
