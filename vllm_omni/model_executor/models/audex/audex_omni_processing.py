# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Adapted from nvidia/Nemotron-Labs-Audex-2B (Apache-2.0):
#   inference_scripts_vllm/audioqa_scripts/audex_2b_vllm/audio_features.py
#   inference_scripts_vllm/audioqa_scripts/audex_2b_vllm/processing_audex_vllm.py
"""Multimodal processing for Audex audio understanding (S2T / audio QA).

No unified HF ``ProcessorMixin`` exists for Audex (only a
``WhisperFeatureExtractor`` under ``audio_preprocessor/``), so
``_call_hf_processor`` is fully overridden: it tokenizes the prompt and
builds NV-Whisper features with 30 s non-overlapping windows (last clip
zero-padded), exactly mirroring the release HF path.

Placeholder expansion uses vLLM's ``PromptReplacement``: a single
``<so_embedding>`` per audio expands to ``<so_start>`` + N*``<so_embedding>``
+ ``<so_end>`` where N = num_clips * 750.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import BatchFeature
from vllm.config.multimodal import BaseDummyOptions
from vllm.inputs import MultiModalDataDict
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.parse import MultiModalDataItems, MultiModalDataParser
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)

SOUND_TOKEN = "<so_embedding>"
SOUND_START_TOKEN = "<so_start>"
SOUND_END_TOKEN = "<so_end>"

SOUND_EMBEDDING_SIZE = 750
SOUND_CLIP_DURATION = 30.0
SOUND_TARGET_RATE = 16000

MAX_AUDIO_SECONDS = 900
MAX_AUDIO_CLIPS = 30
MAX_AUDIO_TOKENS = MAX_AUDIO_CLIPS * SOUND_EMBEDDING_SIZE

AUDIO_ENCODER_MICRO_BATCH = 8


# ---------------------------------------------------------------- features


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """Return mono float32 audio in [-1, 1], matching the Megatron/HF path."""
    audio = np.asarray(audio)
    if audio.ndim == 2:
        if audio.shape[1] <= 2:
            audio = audio.mean(axis=1)
        elif audio.shape[0] <= 2:
            audio = audio.mean(axis=0)
        else:
            raise ValueError(f"Unsupported audio shape: {audio.shape}")

    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype != np.float32:
        audio = audio.astype(np.float32)

    max_abs = float(np.abs(audio).max()) if audio.size else 0.0
    if max_abs > 1.0:
        audio = audio / max_abs
    return audio.astype(np.float32, copy=False)


def split_audio_into_clips(
    audio: np.ndarray,
    sample_rate: int = SOUND_TARGET_RATE,
    clip_duration: float = SOUND_CLIP_DURATION,
) -> list[np.ndarray]:
    """Split audio into fixed clips; pad the final clip to full length."""
    audio = normalize_audio(audio)
    clip_samples = int(round(sample_rate * clip_duration))
    if clip_samples <= 0:
        raise ValueError(f"Invalid clip_samples: {clip_samples}")
    if audio.size == 0:
        audio = np.zeros(1, dtype=np.float32)

    num_clips = max(1, math.ceil(audio.shape[0] / clip_samples))
    clips: list[np.ndarray] = []
    for idx in range(num_clips):
        start = idx * clip_samples
        clip = audio[start : start + clip_samples]
        if clip.shape[0] < clip_samples:
            clip = np.pad(clip, (0, clip_samples - clip.shape[0]))
        clips.append(clip.astype(np.float32, copy=False))
    return clips


def extract_whisper_features(
    feature_extractor,
    audio: np.ndarray,
    sample_rate: int = SOUND_TARGET_RATE,
    clip_duration: float = SOUND_CLIP_DURATION,
) -> torch.Tensor:
    """Return NV-Whisper input features shaped (num_clips, 128, 3000)."""
    clips = split_audio_into_clips(audio, sample_rate=sample_rate, clip_duration=clip_duration)
    features = feature_extractor(
        clips,
        sampling_rate=sample_rate,
        return_tensors="pt",
        padding="max_length",
        return_attention_mask=False,
    )
    input_features = features.input_features
    if input_features.ndim != 3:
        raise ValueError(f"Expected 3D Whisper features, got {tuple(input_features.shape)}")
    return input_features


# ---------------------------------------------------------------- processor


class AudexProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self):
        return self.ctx.model_config.hf_config

    def get_feature_extractor(self, **kwargs: object):
        from transformers import AutoFeatureExtractor

        config = self.get_hf_config()
        path = getattr(config, "audio_preprocessor_path", None) or "audio_preprocessor"
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = Path(self.ctx.model_config.model) / candidate
        return AutoFeatureExtractor.from_pretrained(str(candidate))

    def get_data_parser(self):
        return MultiModalDataParser(target_sr=SOUND_TARGET_RATE, target_channels=1)

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"audio": 1}

    def get_mm_max_tokens_per_item(self, seq_len: int, mm_counts: Mapping[str, int] | None = None) -> Mapping[str, int]:
        return {"audio": MAX_AUDIO_TOKENS}

    def sound_token_ids(self) -> tuple[int, int, int]:
        cfg = self.get_hf_config()
        return (
            int(cfg.sound_start_token_id),
            int(cfg.sound_token_id),
            int(cfg.sound_end_token_id),
        )


class AudexDummyInputsBuilder(BaseDummyInputsBuilder[AudexProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return SOUND_TOKEN * mm_counts.get("audio", 0)

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        num_audios = mm_counts.get("audio", 0)
        audio_len = int(SOUND_CLIP_DURATION * SOUND_TARGET_RATE)
        overrides = mm_options.get("audio") if mm_options else None
        return {"audio": self._get_dummy_audios(length=audio_len, num_audios=num_audios, overrides=overrides)}


class AudexMultiModalProcessor(BaseMultiModalProcessor[AudexProcessingInfo]):
    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, Any],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        tokenizer = self.info.get_tokenizer()
        prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids

        audios = mm_data.get("audios", []) or mm_data.get("audio", [])
        if not audios:
            return BatchFeature(dict(input_ids=[prompt_ids]), tensor_type="pt")

        feature_extractor = self.info.get_feature_extractor()
        all_features: list[torch.Tensor] = []
        audio_num_clips: list[int] = []
        for item in audios:
            audio = item[0] if isinstance(item, tuple) else item
            audio = normalize_audio(np.asarray(audio))
            duration = len(audio) / SOUND_TARGET_RATE
            if duration > MAX_AUDIO_SECONDS:
                raise ValueError(f"Audio duration {duration:.1f}s exceeds MAX_AUDIO_SECONDS={MAX_AUDIO_SECONDS}s.")
            clips = split_audio_into_clips(audio, sample_rate=SOUND_TARGET_RATE, clip_duration=SOUND_CLIP_DURATION)
            if len(clips) > MAX_AUDIO_CLIPS:
                raise ValueError(f"Audio needs {len(clips)} clips > MAX_AUDIO_CLIPS={MAX_AUDIO_CLIPS}.")
            feats = extract_whisper_features(
                feature_extractor,
                audio,
                sample_rate=SOUND_TARGET_RATE,
                clip_duration=SOUND_CLIP_DURATION,
            )
            all_features.append(feats)
            audio_num_clips.append(feats.shape[0])

        input_features = torch.cat(all_features, dim=0)
        return BatchFeature(
            dict(
                input_ids=[prompt_ids],
                input_features=input_features,
                audio_num_clips=torch.tensor(audio_num_clips, dtype=torch.long),
            ),
            tensor_type="pt",
        )

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        audio_num_clips = hf_inputs.get("audio_num_clips", torch.zeros(0, dtype=torch.long))
        return dict(
            input_features=MultiModalFieldConfig.flat_from_sizes("audio", audio_num_clips),
            audio_num_clips=MultiModalFieldConfig.batched("audio"),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        so_start, so_embed, so_end = self.info.sound_token_ids()
        out_mm_data = out_mm_kwargs.get_data()
        num_clips = out_mm_data.get("audio_num_clips", torch.zeros(0, dtype=torch.long))
        num_clips_list = num_clips.flatten().tolist() if hasattr(num_clips, "flatten") else list(num_clips)

        def get_replacement_audex(item_idx: int):
            n = int(num_clips_list[item_idx]) * SOUND_EMBEDDING_SIZE
            if n < 1:
                raise ValueError(f"Audio item {item_idx} produced 0 embeddings.")
            full = [so_start] + [so_embed] * n + [so_end]
            return PromptUpdateDetails.select_token_id(full, so_embed)

        return [
            PromptReplacement(
                modality="audio",
                target=[so_embed],
                replacement=get_replacement_audex,
            )
        ]
