# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AURA Omni pipeline topology.

Semantic modules:
  1. Qwen3-ASR: microphone audio -> transcript text
  2. AURA/Qwen3-VL: transcript + video -> text or <|silent|>
  3. Qwen3-TTS: response text -> audio

The Qwen3-TTS module is represented as two native engine stages, Talker and
Code2Wav, to reuse the existing vLLM-Omni implementation without wrapping or
duplicating codec streaming behavior.
"""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

_AURA_PROC = "vllm_omni.model_executor.stage_input_processors.aura_omni"
_QWEN3_TTS_PROC = "vllm_omni.model_executor.stage_input_processors.qwen3_tts"


AURA_OMNI_PIPELINE = PipelineConfig(
    model_type="aura_omni",
    model_arch="Qwen3ASRForConditionalGeneration",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="asr",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            owns_tokenizer=True,
            requires_multimodal_data=True,
            engine_output_type="text",
            model_arch="Qwen3ASRForConditionalGeneration",
            sampling_constraints={"detokenize": True},
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="aura",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(0,),
            final_output=True,
            final_output_type="text",
            owns_tokenizer=True,
            requires_multimodal_data=True,
            engine_output_type="text",
            model_arch="AuraQwen3VLForConditionalGeneration",
            custom_process_input_func=f"{_AURA_PROC}.asr2aura",
            sampling_constraints={"detokenize": True},
        ),
        StagePipelineConfig(
            stage_id=2,
            model_stage="qwen3_tts",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(1,),
            owns_tokenizer=True,
            engine_output_type="latent",
            model_arch="Qwen3TTSTalkerForConditionalGeneration",
            custom_process_input_func=f"{_AURA_PROC}.aura2tts",
            custom_process_next_stage_input_func=f"{_QWEN3_TTS_PROC}.talker2code2wav_full_payload",
            async_chunk_process_next_stage_input_func=f"{_QWEN3_TTS_PROC}.talker2code2wav_async_chunk",
            sampling_constraints={
                "detokenize": False,
                "stop_token_ids": [2150],
            },
        ),
        StagePipelineConfig(
            stage_id=3,
            model_stage="code2wav",
            execution_type=StageExecutionType.LLM_GENERATION,
            input_sources=(2,),
            final_output=True,
            final_output_type="audio",
            engine_output_type="audio",
            model_arch="Qwen3TTSCode2Wav",
            sync_process_input_func=f"{_QWEN3_TTS_PROC}.talker2code2wav_token_only",
            sampling_constraints={"detokenize": True},
            extras={"tts_args": {"max_instructions_length": 500}},
        ),
    ),
)
