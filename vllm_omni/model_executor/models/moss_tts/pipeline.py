# Copyright 2026 OpenMOSS and the vLLM-Omni team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
"""Pipeline topology for all MOSS-TTS variants (2-stage: talker → codec)."""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

_PROC = "vllm_omni.model_executor.stage_input_processors.moss_tts"

# ---------------------------------------------------------------------------
# Shared 2-stage pipeline (used by all 5 MOSS-TTS variants)
#
#   Stage 0  (LLM_AR)         — Qwen3 backbone + (n_vq+1) parallel heads
#                                emits interleaved text + audio VQ codes
#   Stage 1  (LLM_GENERATION) — MOSS Audio Tokenizer decode
#                                emits 24 kHz mono waveform chunks
# ---------------------------------------------------------------------------

MOSS_TTS_PIPELINE = PipelineConfig(
    model_type="moss_tts",
    model_arch="MossTTSDelayModel",  # HF architectures string
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="moss_tts",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            owns_tokenizer=True,
            engine_output_type="latent",
            async_chunk_process_next_stage_input_func=(f"{_PROC}.talker2codec_delay_async_chunk"),
            sampling_constraints={
                "detokenize": False,
            },
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="moss_tts_codec",
            execution_type=StageExecutionType.LLM_GENERATION,
            input_sources=(0,),
            final_output=True,
            final_output_type="audio",
            engine_output_type="audio",
            model_arch="MossTTSCodecDecoder",
            sync_process_input_func=f"{_PROC}.talker2codec",
            sampling_constraints={"detokenize": True},
        ),
    ),
)

MOSS_TTS_REALTIME_PIPELINE = PipelineConfig(
    model_type="moss_tts_realtime",
    model_arch="MossTTSRealtime",  # different talker class from the delay variant
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="moss_tts",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            owns_tokenizer=True,
            engine_output_type="latent",
            async_chunk_process_next_stage_input_func=(f"{_PROC}.talker2codec_raw_async_chunk"),
            sampling_constraints={"detokenize": False},
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="moss_tts_codec",
            execution_type=StageExecutionType.LLM_GENERATION,
            input_sources=(0,),
            final_output=True,
            final_output_type="audio",
            engine_output_type="audio",
            model_arch="MossTTSCodecDecoder",
            sync_process_input_func=f"{_PROC}.talker2codec",
            sampling_constraints={"detokenize": True},
        ),
    ),
)

MOSS_TTS_LOCAL_PIPELINE = PipelineConfig(
    model_type="moss_tts_local",
    model_arch="MossTTSLocalModel",  # different talker class: GPT2-style local depth transformer
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="moss_tts_local",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            owns_tokenizer=True,
            engine_output_type="latent",
            async_chunk_process_next_stage_input_func=(f"{_PROC}.talker2codec_raw_async_chunk"),
            sampling_constraints={
                "detokenize": False,
                "stop_token_ids": [151645],
            },
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="moss_tts_local_codec",
            execution_type=StageExecutionType.LLM_GENERATION,
            input_sources=(0,),
            final_output=True,
            final_output_type="audio",
            engine_output_type="audio",
            model_arch="MossTTSCodecDecoder",
            sync_process_input_func=f"{_PROC}.talker2codec",
            sampling_constraints={"detokenize": True},
        ),
    ),
)

# The pipeline config is otherwise the same for all variants; the per-variant
# differences (n_vq, backbone size, generation strategy) are encoded in the
# HF config.json and the deploy YAML. Realtime and Local are split out because
# they have different talker architectures from the delay variant
# (MossTTSRealtime / MossTTSLocalModel vs MossTTSDelayModel).

__all__ = ["MOSS_TTS_PIPELINE", "MOSS_TTS_REALTIME_PIPELINE", "MOSS_TTS_LOCAL_PIPELINE"]
