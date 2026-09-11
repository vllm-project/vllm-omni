# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Audio8 TTS Preview pipeline topology.

Stage 0: ``audio8_tts_slow_ar``       -- text -> semantic tokens + codec codes.
Stage 1: ``audio8_tts_codec_decoder`` -- codec codes -> 44.1 kHz waveform.

Two checkpoints share ``model_type = "arktts"``: the 0.6b (Qwen2 Slow AR) and
the 0.1b (Falcon-H1 Slow AR). ``resolve_arktts_pipeline`` disambiguates them by
the config's ``slow_backbone`` field, since the codec (Stage 1) is identical.
"""

from transformers import PretrainedConfig

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

from .configuration_audio8_tts import ARKTTS_SLOW_BACKBONE_FALCON_H1

_PROC = "vllm_omni.model_executor.stage_input_processors.audio8_tts"

#: ``<|im_end|>``: the Slow AR ends the utterance with an end-of-turn token.
AUDIO8_TTS_EOS_TOKEN_ID = 151645
#: 0.1b tokenizer maps ``<|im_end|>`` to a different id.
AUDIO8_TTS_01B_EOS_TOKEN_ID = 228

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


#: Audio8 TTS Preview 0.1b: Falcon-H1 hybrid Slow AR. Same topology as 0.6b and
#: the identical codec decoder in Stage 1; only Stage 0's model class and the
#: end-of-turn token differ.
AUDIO8_TTS_01B_PIPELINE = PipelineConfig(
    model_type="arktts",
    default_deploy_config_name="audio8_tts_01b.yaml",
    model_arch="Audio8TTS01BSlowARForConditionalGeneration",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="audio8_tts_01b_slow_ar",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            owns_tokenizer=True,
            engine_output_type="latent",
            async_chunk_process_next_stage_input_func=(f"{_PROC}.slow_ar_to_codec_decoder_async_chunk"),
            sampling_constraints={
                "detokenize": False,
                "stop_token_ids": [AUDIO8_TTS_01B_EOS_TOKEN_ID],
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


def resolve_arktts_pipeline(hf_config: PretrainedConfig | None = None) -> PipelineConfig:
    """Pick the Slow AR pipeline by ``slow_backbone`` (0.1b Falcon-H1 vs 0.6b)."""
    backbone = getattr(hf_config, "slow_backbone", None) if hf_config is not None else None
    if backbone == ARKTTS_SLOW_BACKBONE_FALCON_H1:
        return AUDIO8_TTS_01B_PIPELINE
    return AUDIO8_TTS_PIPELINE
