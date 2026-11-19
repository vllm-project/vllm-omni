# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Step-Audio2 pipeline topologies (frozen)."""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)
from vllm_omni.serving_capabilities import (
    AUDIO_RESPONSE,
    TEXT_RESPONSE,
    TEXT_TO_SPEECH,
    declare_capabilities,
)

_PROC = "vllm_omni.model_executor.stage_input_processors.step_audio2"
_HF_ARCHITECTURES = (
    "StepAudio2ForCausalLM",
    "StepAudio2ForConditionalGeneration",
)


STEP_AUDIO2_PIPELINE = PipelineConfig(
    serving_capabilities=declare_capabilities(
        TEXT_RESPONSE,
        AUDIO_RESPONSE,
        TEXT_TO_SPEECH,
        sources=("vllm_omni/model_executor/models/step_audio2/pipeline.py",),
    ),
    model_type="step_audio_2",
    model_arch="StepAudio2ForConditionalGeneration",
    hf_architectures=_HF_ARCHITECTURES,
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="step_audio2_thinker",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            final_output=True,
            final_output_type="text",
            owns_tokenizer=True,
            requires_multimodal_data=True,
            engine_output_type="latent",
            model_arch="StepAudio2ThinkerForConditionalGeneration",
            async_chunk_process_next_stage_input_func=f"{_PROC}.thinker2token2wav_async_chunk",
            sampling_constraints={"detokenize": True},
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="token2wav",
            execution_type=StageExecutionType.LLM_GENERATION,
            input_sources=(0,),
            final_output=True,
            final_output_type="audio",
            engine_output_type="audio",
            model_arch="StepAudio2Token2WavModel",
            sync_process_input_func=f"{_PROC}.thinker2token2wav",
            sampling_constraints={"detokenize": False},
        ),
    ),
)


STEP_AUDIO2_ASR_PIPELINE = PipelineConfig(
    serving_capabilities=declare_capabilities(
        TEXT_RESPONSE,
        sources=("vllm_omni/model_executor/models/step_audio2/pipeline.py",),
    ),
    model_type="step_audio_2_asr",
    model_arch="StepAudio2ForConditionalGeneration",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="step_audio2_thinker",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            final_output=True,
            final_output_type="text",
            owns_tokenizer=True,
            requires_multimodal_data=True,
            engine_output_type="text",
            model_arch="StepAudio2ThinkerForConditionalGeneration",
            sampling_constraints={"detokenize": True},
        ),
    ),
)
