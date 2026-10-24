# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Audio8 TTS Preview pipeline topologies.

Dense variant (0.6b):
Stage 0: ``audio8_tts_slow_ar``       -- text -> semantic tokens + codec codes.
Stage 1: ``audio8_tts_codec_decoder`` -- codec codes -> 44.1 kHz waveform.

Hybrid variant (0.1b, ``slow_backbone = "falcon_h1"``) keeps the same two-stage
topology and only swaps the stage-0 architecture; ``resolve_audio8_pipeline``
selects between them from the checkpoint config, since both checkpoints share
``model_type = "arktts"``.
"""

from typing import TYPE_CHECKING

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

if TYPE_CHECKING:
    from transformers import PretrainedConfig

_PROC = "vllm_omni.model_executor.stage_input_processors.audio8_tts"

#: ``<|im_end|>``: the 0.6b Slow AR ends the utterance with an end-of-turn token.
AUDIO8_TTS_EOS_TOKEN_ID = 151645
#: The 0.1b checkpoint uses its own tokenizer; ``eos_token_id`` is 228 there.
AUDIO8_TTS_HYBRID_EOS_TOKEN_ID = 228


def _stages(*, slow_ar_stage: str, slow_ar_arch: str, eos_token_id: int) -> tuple[StagePipelineConfig, ...]:
    return (
        StagePipelineConfig(
            stage_id=0,
            model_stage=slow_ar_stage,
            model_arch=slow_ar_arch,
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            owns_tokenizer=True,
            engine_output_type="latent",
            async_chunk_process_next_stage_input_func=(f"{_PROC}.slow_ar_to_codec_decoder_async_chunk"),
            sampling_constraints={
                "detokenize": False,
                "stop_token_ids": [eos_token_id],
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
    )


AUDIO8_TTS_PIPELINE = PipelineConfig(
    model_type="arktts",
    default_deploy_config_name="audio8_tts.yaml",
    model_arch="Audio8TTSSlowARForConditionalGeneration",
    stages=_stages(
        slow_ar_stage="audio8_tts_slow_ar",
        slow_ar_arch="Audio8TTSSlowARForConditionalGeneration",
        eos_token_id=AUDIO8_TTS_EOS_TOKEN_ID,
    ),
)

#: Hybrid-Mamba2 variant (0.1b). Same topology, FalconH1 slow AR.
AUDIO8_TTS_HYBRID_PIPELINE = PipelineConfig(
    model_type="arktts",
    default_deploy_config_name="audio8_tts_hybrid.yaml",
    model_arch="Audio8TTSHybridSlowARForConditionalGeneration",
    stages=_stages(
        slow_ar_stage="audio8_tts_hybrid_slow_ar",
        slow_ar_arch="Audio8TTSHybridSlowARForConditionalGeneration",
        eos_token_id=AUDIO8_TTS_HYBRID_EOS_TOKEN_ID,
    ),
)


def resolve_audio8_pipeline(config: "PretrainedConfig | None") -> PipelineConfig:
    """Pick the Audio8 topology from the checkpoint config.

    Both checkpoints declare ``model_type = "arktts"``; the distinguishing field
    is ``slow_backbone`` (absent on the 0.6b, ``"falcon_h1"`` on the 0.1b).
    """
    slow_backbone = getattr(config, "slow_backbone", None) if config is not None else None
    if slow_backbone == "falcon_h1":
        return AUDIO8_TTS_HYBRID_PIPELINE
    return AUDIO8_TTS_PIPELINE
