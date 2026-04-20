# SPDX-License-Identifier: Apache-2.0
"""Fun-Audio-Chat-8B pipeline: Stage 0 (LM_AR: text + CRQ tokens) → Stage 1 (token2wav → WAV)."""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
    register_pipeline,
)

_PROC = "vllm_omni.model_executor.stage_input_processors.funaudiochat"

FUNAUDIOCHAT_PIPELINE = PipelineConfig(
    model_type="funaudiochat",
    model_arch="FunAudioChatForConditionalGeneration",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="funaudiochat",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            owns_tokenizer=True,
            engine_output_type="latent",
            sampling_constraints={
                "detokenize": False,
            },
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="funaudiochat_token2wav",
            execution_type=StageExecutionType.LLM_GENERATION,
            input_sources=(0,),
            final_output=True,
            final_output_type="audio",
            engine_output_type="audio",
            model_arch="FunAudioChatToken2Wav",
            sync_process_input_func=f"{_PROC}.talker2code2wav",
            sampling_constraints={"detokenize": True},
        ),
    ),
)

register_pipeline(FUNAUDIOCHAT_PIPELINE)
