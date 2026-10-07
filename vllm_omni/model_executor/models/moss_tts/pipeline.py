# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# Copyright 2026 OpenMOSS and the vLLM-Omni team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
"""Pipeline topology for all MOSS-TTS variants (2-stage: talker → codec)."""

import shutil
from dataclasses import replace

from transformers import PretrainedConfig
from vllm.logger import init_logger
from vllm.sampling_params import RequestOutputKind

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)
from vllm_omni.model_executor.models.moss_tts.configuration_moss_tts import MossTTSLocalConfig

_PROC = "vllm_omni.model_executor.stage_input_processors.moss_tts"
logger = init_logger(__name__)

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
    default_deploy_config_name="moss_tts.yaml",
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
            retains_state_across_chunks=True,
            sync_process_input_func=f"{_PROC}.talker2codec",
            sampling_constraints={"detokenize": True},
        ),
    ),
)

MOSS_TTS_REALTIME_PIPELINE = PipelineConfig(
    model_type="moss_tts_realtime",
    default_deploy_config_name="moss_tts_realtime.yaml",
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
            retains_state_across_chunks=True,
            sync_process_input_func=f"{_PROC}.talker2codec",
            sampling_constraints={"detokenize": True},
        ),
    ),
)

MOSS_TTS_LOCAL_PIPELINE = PipelineConfig(
    model_type="moss_tts_local",
    default_deploy_config_name="moss_tts_local.yaml",
    model_arch="MossTTSLocalModel",  # different talker class: GPT2-style local depth transformer
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="moss_tts_local",
            supports_native_mrv2_data_plane=True,
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            owns_tokenizer=True,
            engine_output_type="latent",
            async_chunk_process_next_stage_input_func=(f"{_PROC}.talker2codec_raw_async_chunk"),
            sampling_constraints={
                "detokenize": False,
                "stop_token_ids": [151645],
                # The worker connector streams codes directly to the codec.
                # This internal stage only needs to publish its terminal
                # result; codec audio output remains incremental.
                "output_kind": RequestOutputKind.FINAL_ONLY,
            },
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="moss_tts_local_codec",
            supports_native_mrv2_data_plane=True,
            execution_type=StageExecutionType.LLM_GENERATION,
            input_sources=(0,),
            final_output=True,
            final_output_type="audio",
            engine_output_type="audio",
            model_arch="MossTTSCodecDecoder",
            retains_state_across_chunks=True,
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


def resolve_moss_tts_local_pipeline(hf_config: PretrainedConfig | None = None) -> PipelineConfig | None:
    """Select CUDA MRV2, with the C128 system profile when prerequisites are met.

    The platform memory query uses NVML, avoiding CUDA initialization before
    workers spawn. Explicit deploy configs override this pipeline default.
    """
    from vllm.platforms import current_platform

    from vllm_omni.platforms import current_omni_platform

    if hf_config is not None and not isinstance(hf_config, MossTTSLocalConfig):
        return None
    if current_omni_platform.device_name != "cuda":
        return replace(MOSS_TTS_LOCAL_PIPELINE, default_deploy_config_name="moss_tts_local_v1.yaml")
    if shutil.which("nvidia-cuda-mps-control") is None:
        logger.info("MOSS Local defaults to CUDA MRV2/C64 without MPS: nvidia-cuda-mps-control is unavailable")
        return replace(MOSS_TTS_LOCAL_PIPELINE, default_deploy_config_name="moss_tts_local_mrv2.yaml")
    try:
        memory_bytes = current_platform.get_device_total_memory()
    except Exception as exc:
        logger.warning("Cannot query CUDA memory for MOSS Local; using MRV2/C64 with MPS: %s", exc)
        return MOSS_TTS_LOCAL_PIPELINE
    if memory_bytes >= 140 * 1024**3:
        return replace(
            MOSS_TTS_LOCAL_PIPELINE,
            default_deploy_config_name="moss_tts_local_mrv2_optimized.yaml",
        )
    return MOSS_TTS_LOCAL_PIPELINE


__all__ = [
    "MOSS_TTS_PIPELINE",
    "MOSS_TTS_REALTIME_PIPELINE",
    "MOSS_TTS_LOCAL_PIPELINE",
    "resolve_moss_tts_local_pipeline",
]
