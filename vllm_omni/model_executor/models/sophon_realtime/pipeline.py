# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Sophon-Realtime pipelines (three stages on one GPU by default).

Stage 0: thinker  — DuplexJev: speech (and an optional image) -> reply text
                    with sound tags. Its text output is the transcript.
Stage 1: talker   — streams the reply text in while it is being written:
                    Tag: CosyVoice3 LM -> speech tokens;
                    Flash: Qwen3-TTS talker -> RVQ codes.
Stage 2: vocoder  — Tag: CosyVoice3 flow + HiFT; Flash: Qwen3-TTS code2wav.

All checkpoints share ``model_type: sophon_realtime``. The root config's
``talker_kind`` selects the pipeline (``tag_tts`` or ``qwen3_tts``; checkpoints
without it are told apart by ``talker/config.json``), and the presence of a
vision tower in ``thinker/config.json`` selects the thinker class.
"""

from __future__ import annotations

import json
import os

from transformers import PretrainedConfig

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)
from vllm_omni.outputs.output_modality import (
    TensorAccumulationStrategy,
    register_key_accumulation_strategy,
)

_PROC = "vllm_omni.model_executor.stage_input_processors.sophon_realtime"
_COSYVOICE3 = "vllm_omni.model_executor.stage_input_processors.cosyvoice3"
_QWEN3_TTS = "vllm_omni.model_executor.stage_input_processors.qwen3_tts"

THINKER_ARCH = "DuplexJevForConditionalGeneration"
THINKER_VISION_ARCH = "DuplexJevVisionForConditionalGeneration"
TAG_TALKER_ARCH = "SophonTagTalker"
FLASH_TALKER_ARCH = "SophonFlashTalker"
HF_ARCHITECTURES = ("SophonRealtimeForConditionalGeneration",)
DUPLEX_PLUGIN = "vllm_omni.model_executor.models.sophon_realtime.duplex.plugin.SophonRealtimeDuplexPlugin"

register_key_accumulation_strategy("codes.audio", TensorAccumulationStrategy.CONCAT_DIM0)


def _thinker_stage(arch: str, next_stage_func: str) -> StagePipelineConfig:
    return StagePipelineConfig(
        stage_id=0,
        model_stage="thinker",
        execution_type=StageExecutionType.LLM_AR,
        input_sources=(),
        final_output=True,
        final_output_type="text",
        owns_tokenizer=True,
        requires_multimodal_data=True,
        engine_output_type="latent",
        model_subdir="thinker",
        tokenizer_subdir="thinker",
        model_arch=arch,
        async_chunk_process_next_stage_input_func=next_stage_func,
        sampling_constraints={"detokenize": True},
    )


def _tag_pipeline(thinker_arch: str) -> PipelineConfig:
    return PipelineConfig(
        model_type="sophon_realtime",
        default_deploy_config_name="sophon_realtime_tag.yaml",
        model_arch=thinker_arch,
        hf_architectures=HF_ARCHITECTURES,
        duplex_plugin=DUPLEX_PLUGIN,
        default_session_mode="turn",
        stages=(
            _thinker_stage(thinker_arch, f"{_PROC}.thinker2talker_tag_async_chunk"),
            StagePipelineConfig(
                stage_id=1,
                model_stage="cosyvoice3_talker",
                execution_type=StageExecutionType.LLM_AR,
                input_sources=(0,),
                engine_output_type="latent",
                model_subdir="talker",
                tokenizer_subdir="talker/CosyVoice-BlankEN",
                model_arch=TAG_TALKER_ARCH,
                async_chunk_process_next_stage_input_func=f"{_COSYVOICE3}.talker2code2wav_async_chunk",
                custom_process_next_stage_input_func=f"{_COSYVOICE3}.text2flow_full_payload",
                sampling_constraints={"detokenize": False, "stop_token_ids": [6562]},
                requires_full_payload_input=True,
            ),
            StagePipelineConfig(
                stage_id=2,
                model_stage="cosyvoice3_code2wav",
                execution_type=StageExecutionType.LLM_GENERATION,
                input_sources=(1,),
                final_output=True,
                final_output_type="audio",
                engine_output_type="latent",
                model_subdir="talker",
                tokenizer_subdir="talker/CosyVoice-BlankEN",
                model_arch=TAG_TALKER_ARCH,
                sync_process_input_func=f"{_COSYVOICE3}.text2flow_token_only",
                requires_full_payload_input=True,
            ),
        ),
    )


def _flash_pipeline(thinker_arch: str) -> PipelineConfig:
    return PipelineConfig(
        model_type="sophon_realtime",
        default_deploy_config_name="sophon_realtime_flash.yaml",
        model_arch=thinker_arch,
        hf_architectures=HF_ARCHITECTURES,
        duplex_plugin=DUPLEX_PLUGIN,
        default_session_mode="turn",
        stages=(
            _thinker_stage(thinker_arch, f"{_PROC}.thinker2talker_flash_async_chunk"),
            StagePipelineConfig(
                stage_id=1,
                model_stage="talker",
                execution_type=StageExecutionType.LLM_AR,
                input_sources=(0,),
                engine_output_type="latent",
                model_subdir="talker",
                tokenizer_subdir="talker",
                model_arch=FLASH_TALKER_ARCH,
                async_chunk_process_next_stage_input_func=f"{_QWEN3_TTS}.talker2code2wav_async_chunk",
                custom_process_next_stage_input_func=f"{_QWEN3_TTS}.talker2code2wav_full_payload",
                sampling_constraints={"detokenize": False, "stop_token_ids": [2150]},
                requires_full_payload_input=True,
            ),
            StagePipelineConfig(
                stage_id=2,
                model_stage="code2wav",
                execution_type=StageExecutionType.LLM_GENERATION,
                input_sources=(1,),
                final_output=True,
                final_output_type="audio",
                engine_output_type="audio",
                model_subdir="talker",
                tokenizer_subdir="talker",
                model_arch="Qwen3TTSCode2Wav",
                sampling_constraints={"detokenize": True},
                requires_full_payload_input=True,
            ),
        ),
    )


SOPHON_REALTIME_TAG_PIPELINE = _tag_pipeline(THINKER_VISION_ARCH)
SOPHON_REALTIME_FLASH_PIPELINE = _flash_pipeline(THINKER_VISION_ARCH)


def _read_json(path: str) -> dict | None:
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def _thinker_has_vision(hf_config: PretrainedConfig) -> bool:
    root = getattr(hf_config, "_name_or_path", "") or ""
    cfg = _read_json(os.path.join(root, getattr(hf_config, "thinker_subdir", "thinker"), "config.json"))
    return True if cfg is None else "vision_config" in cfg  # the released checkpoints have a vision tower


def _talker_kind(hf_config: PretrainedConfig) -> str:
    kind = getattr(hf_config, "talker_kind", None)
    if kind:
        return kind
    root = getattr(hf_config, "_name_or_path", "") or ""
    talker = _read_json(os.path.join(root, getattr(hf_config, "talker_subdir", "talker"), "config.json")) or {}
    return "qwen3_tts" if talker.get("model_type") == "qwen3_tts" else "tag_tts"


def resolve_sophon_realtime_pipeline(hf_config: PretrainedConfig | None = None) -> PipelineConfig | None:
    """Pick the Tag or Flash pipeline and the thinker class for a checkpoint."""
    if hf_config is None:
        return SOPHON_REALTIME_TAG_PIPELINE
    if getattr(hf_config, "model_type", None) != "sophon_realtime":
        return None
    thinker_arch = THINKER_VISION_ARCH if _thinker_has_vision(hf_config) else THINKER_ARCH
    kind = _talker_kind(hf_config)
    if kind == "qwen3_tts":
        return _flash_pipeline(thinker_arch)
    if kind == "tag_tts":
        return _tag_pipeline(thinker_arch)
    raise ValueError(f"Sophon-Realtime: talker_kind {kind!r} is not supported yet (supported: 'tag_tts', 'qwen3_tts')")
