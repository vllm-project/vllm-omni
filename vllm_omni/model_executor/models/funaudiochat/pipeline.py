# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""FunAudioChat speech-to-speech pipeline topology."""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

_PROC = "vllm_omni.model_executor.stage_input_processors.funaudiochat"

FUN_AUDIO_CHAT_PIPELINE = PipelineConfig(
    model_type="funaudiochat",
    default_deploy_config_name="funaudiochat.yaml",
    model_arch="FunAudioChatForConditionalGeneration",
    hf_architectures=("FunAudioChatForConditionalGeneration",),
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="s2s",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            owns_tokenizer=True,
            requires_multimodal_data=True,
            engine_output_type="latent",
            model_arch="FunAudioChatForConditionalGeneration",
            async_chunk_process_next_stage_input_func=f"{_PROC}.funaudiochat2code2wav_async_chunk",
            sampling_constraints={"detokenize": True},
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="cosyvoice3_code2wav",
            execution_type=StageExecutionType.LLM_GENERATION,
            input_sources=(0,),
            final_output=True,
            final_output_type="audio",
            engine_output_type="audio",
            model_arch="FunAudioChatCosyVoice3Code2Wav",
            sync_process_input_func=f"{_PROC}.funaudiochat2code2wav",
            sampling_constraints={"detokenize": False},
        ),
    ),
)

__all__ = ["FUN_AUDIO_CHAT_PIPELINE"]
