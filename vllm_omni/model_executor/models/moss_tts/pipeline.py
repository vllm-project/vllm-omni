# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# Copyright 2026 OpenMOSS and the vLLM-Omni team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
"""Pipeline topology for all MOSS-TTS variants (2-stage: talker → codec)."""

from collections.abc import Mapping, Sequence
from dataclasses import replace

from vllm.sampling_params import RequestOutputKind

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)
from vllm_omni.serving_capabilities import (
    AUDIO_RESPONSE,
    CAPTION_TO_AUDIO,
    TEXT_TO_SPEECH,
    ServingCapabilities,
    declare_capabilities,
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
    serving_capabilities_resolver="vllm_omni.model_executor.models.moss_tts.pipeline.resolve_moss_capabilities",
    serving_capabilities=declare_capabilities(
        TEXT_TO_SPEECH,
        replace(AUDIO_RESPONSE, support="unknown", reason_code="chat_prompt_contract_undeclared"),
        unknown_reason="chat_prompt_contract_undeclared",
        sources=("vllm_omni/model_executor/models/moss_tts/pipeline.py", "docs/serving/speech_api.md"),
    ),
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
            # The local depth transformer folds each sampled code frame into this
            # step's inputs_embeds inside talker_mtp (vllm-project/vllm-omni#6179).
            recompute_preemption="fail",
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
    serving_capabilities_resolver="vllm_omni.model_executor.models.moss_tts.pipeline.resolve_moss_capabilities",
    serving_capabilities=declare_capabilities(
        TEXT_TO_SPEECH,
        replace(AUDIO_RESPONSE, support="unknown", reason_code="chat_prompt_contract_undeclared"),
        unknown_reason="chat_prompt_contract_undeclared",
        sources=("vllm_omni/model_executor/models/moss_tts/pipeline.py", "docs/serving/speech_api.md"),
    ),
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
            # The local depth transformer folds each sampled code frame into this
            # step's inputs_embeds inside talker_mtp (vllm-project/vllm-omni#6179).
            recompute_preemption="fail",
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
    serving_capabilities_resolver="vllm_omni.model_executor.models.moss_tts.pipeline.resolve_moss_capabilities",
    serving_capabilities=declare_capabilities(
        TEXT_TO_SPEECH,
        replace(AUDIO_RESPONSE, support="unknown", reason_code="chat_prompt_contract_undeclared"),
        unknown_reason="chat_prompt_contract_undeclared",
        sources=("vllm_omni/model_executor/models/moss_tts/pipeline.py", "docs/serving/speech_api.md"),
    ),
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
            # The local depth transformer folds each sampled code frame into this
            # step's inputs_embeds inside talker_mtp (vllm-project/vllm-omni#6179).
            recompute_preemption="fail",
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


__all__ = [
    "MOSS_TTS_PIPELINE",
    "MOSS_TTS_REALTIME_PIPELINE",
    "MOSS_TTS_LOCAL_PIPELINE",
]


def resolve_moss_capabilities(declaration: ServingCapabilities, stages: Sequence[object]) -> ServingCapabilities:
    """Use the exact variant dispatch shared by the initialized speech adapter."""
    from vllm_omni.model_executor.models.moss_tts.model_variant import detect_moss_variant

    first = stages[0]
    args = first.get("engine_args") if isinstance(first, Mapping) else getattr(first, "engine_args", None)
    model = args.get("model") if isinstance(args, Mapping) else getattr(args, "model", None)
    if not isinstance(model, str):
        return declare_capabilities(
            replace(TEXT_TO_SPEECH, support="unknown", reason_code="serving_variant_unavailable"),
            replace(CAPTION_TO_AUDIO, support="unknown", reason_code="serving_variant_unavailable"),
            replace(AUDIO_RESPONSE, support="unknown", reason_code="chat_prompt_contract_undeclared"),
            sources=declaration.sources,
            unknown_reason="serving_variant_unavailable",
        )
    variant = detect_moss_variant(model)
    if variant == "sound_effect":
        profile = replace(CAPTION_TO_AUDIO, required_inputs=("ambient_sound",))
    elif variant == "voice_generator":
        profile = replace(TEXT_TO_SPEECH, required_inputs=("text", "instructions"))
    else:
        profile = replace(TEXT_TO_SPEECH, constraints=(("reference_audio_or_registered_voice", True),))
    return declare_capabilities(
        profile,
        replace(AUDIO_RESPONSE, support="unknown", reason_code="chat_prompt_contract_undeclared"),
        unknown_reason="chat_prompt_contract_undeclared",
        sources=("vllm_omni/entrypoints/openai/tts_adapters/moss_tts.py", "recipes/OpenMOSS/MOSS-TTS.md"),
    )
