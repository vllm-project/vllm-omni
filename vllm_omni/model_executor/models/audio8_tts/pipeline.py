# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Audio8 TTS Preview pipeline topology.

Stage 0: ``audio8_tts_slow_ar``       -- text -> semantic tokens + codec codes.
Stage 1: ``audio8_tts_codec_decoder`` -- codec codes -> 44.1 kHz waveform.
"""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

_PROC = "vllm_omni.model_executor.stage_input_processors.audio8_tts"

#: ``<|im_end|>``: the Slow AR ends the utterance with an end-of-turn token.
AUDIO8_TTS_EOS_TOKEN_ID = 151645

AUDIO8_TTS_PIPELINE = PipelineConfig(
    model_type="arktts",
    default_deploy_config_name="audio8_tts.yaml",
    model_arch="Audio8TTSSlowARForConditionalGeneration",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="audio8_tts_slow_ar",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            owns_tokenizer=True,
            engine_output_type="latent",
            async_chunk_process_next_stage_input_func=(f"{_PROC}.slow_ar_to_codec_decoder_async_chunk"),
            sampling_constraints={
                "detokenize": False,
                "stop_token_ids": [AUDIO8_TTS_EOS_TOKEN_ID],
            },
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="audio8_tts_codec_decoder",
            model_arch="Audio8TTSCodecDecoder",
            execution_type=StageExecutionType.LLM_GENERATION,
            input_sources=(0,),
            final_output=True,
            final_output_type="audio",
            engine_output_type="audio",
            sampling_constraints={"detokenize": True},
        ),
    ),
)
