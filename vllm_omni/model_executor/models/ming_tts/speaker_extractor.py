# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adopted from https://github.com/inclusionAI/Ming-omni-tts/blob/main/spkemb_extractor.py
from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

import torch
from vllm.logger import init_logger

from vllm_omni.model_executor.models.common.ming.speaker_extractor import SpeakerEmbeddingExtractor

from .audio_prep import coerce_prompt_waveform, coerce_speaker_embeddings
from .constants import (
    KEY_SPEAKER_EMBEDDING,
    KEY_SPEAKER_SAMPLE_RATES,
    KEY_SPEAKER_WAVEFORM,
    KEY_SPEAKER_WAVEFORM_LENGTHS,
)

if TYPE_CHECKING:
    from .ming_tts import MingTTSForConditionalGeneration

logger = init_logger(__name__)
_EXTRACTOR_LOAD_LOCK = threading.Lock()


# Ming's bundle ships the CAM++ artifact as `campplus.onnx` in the model
# directory, which is the shared extractor's default.
def _load_speaker_extractor(wrapper: MingTTSForConditionalGeneration) -> SpeakerEmbeddingExtractor:
    extractor = getattr(wrapper, "_speaker_extractor", None)
    if extractor is not None:
        return extractor

    with _EXTRACTOR_LOAD_LOCK:
        extractor = getattr(wrapper, "_speaker_extractor", None)
        if extractor is None:
            model_path = wrapper.vllm_config.model_config.model
            extractor = SpeakerEmbeddingExtractor(model_path, target_sr=16000)
            wrapper._speaker_extractor = extractor
    return extractor


def _speaker_waveforms_from_info(
    info_dict: dict[str, Any],
    default_sample_rate: int,
) -> list[tuple[torch.Tensor, int]]:
    raw_waveform = info_dict.get(KEY_SPEAKER_WAVEFORM)
    if raw_waveform is None:
        return []

    waveform = coerce_prompt_waveform(raw_waveform)
    raw_lengths = info_dict.get(KEY_SPEAKER_WAVEFORM_LENGTHS)
    if raw_lengths is None:
        parts = [waveform]
    else:
        lengths = torch.as_tensor(raw_lengths).detach().reshape(-1).to(torch.int64).cpu().tolist()
        if not lengths or any(int(length) <= 0 for length in lengths):
            raise ValueError(f"Invalid Ming speaker waveform lengths: {lengths}")
        if sum(int(length) for length in lengths) > int(waveform.shape[-1]):
            raise ValueError(
                "Ming speaker waveform lengths exceed the waveform size: "
                f"lengths={lengths}, waveform_samples={int(waveform.shape[-1])}"
            )

        parts = []
        offset = 0
        for length in lengths:
            end = offset + int(length)
            parts.append(waveform[:, offset:end])
            offset = end

    raw_sample_rates = info_dict.get(KEY_SPEAKER_SAMPLE_RATES)
    if raw_sample_rates is None:
        sample_rates = [int(default_sample_rate)] * len(parts)
    else:
        sample_rates = torch.as_tensor(raw_sample_rates).detach().reshape(-1).to(torch.int64).cpu().tolist()
        if len(sample_rates) != len(parts) or any(int(sample_rate) <= 0 for sample_rate in sample_rates):
            raise ValueError(
                f"Invalid Ming speaker sample rates: sample_rates={sample_rates}, waveform_count={len(parts)}"
            )
    return [(waveform_part, int(sample_rate)) for waveform_part, sample_rate in zip(parts, sample_rates, strict=True)]


def _resolve_speaker_embeddings(
    wrapper: MingTTSForConditionalGeneration,
    info_dict: dict[str, Any],
) -> list[torch.Tensor] | None:
    direct_embedding = info_dict.get(KEY_SPEAKER_EMBEDDING, info_dict.get("speaker_embedding"))
    use_zero_spk_emb = bool(info_dict.get("use_zero_spk_emb", False))
    if direct_embedding is not None or use_zero_spk_emb:
        return coerce_speaker_embeddings(direct_embedding, use_zero_spk_emb=use_zero_spk_emb)

    voice_name = info_dict.get("voice_name")
    if isinstance(voice_name, (list, tuple)):
        voice_name = voice_name[0] if voice_name else None
    created_at = info_dict.get("voice_created_at", 0)
    if isinstance(created_at, (list, tuple)):
        created_at = created_at[0] if created_at else 0
    if isinstance(created_at, torch.Tensor):
        created_at = created_at.detach().reshape(-1)[0].item() if created_at.numel() else 0

    cache = wrapper._speaker_cache
    cache_key = None
    if isinstance(voice_name, str) and voice_name.strip():
        cache_key = cache.make_cache_key(voice_name, model_type="ming_tts", created_at=int(created_at or 0))
        cached = cache.get(cache_key)
        if cached is not None:
            cached_embeddings = coerce_speaker_embeddings(cached.get("speaker_embedding"))
            if cached_embeddings is not None:
                logger.debug("Speaker cache HIT for Ming-TTS speaker '%s'", voice_name)
                return cached_embeddings

    waveform_inputs = _speaker_waveforms_from_info(info_dict, int(wrapper.ming_config.sample_rate))
    if not waveform_inputs:
        if cache_key is not None:
            raise ValueError(f"Ming-TTS speaker '{voice_name}' was requested without reference audio on cache miss")
        return None

    extractor = _load_speaker_extractor(wrapper)
    embeddings = [
        extractor.extract_from_waveform(waveform, sample_rate).detach().reshape(-1).to(torch.float32).cpu()
        for waveform, sample_rate in waveform_inputs
    ]
    resolved = coerce_speaker_embeddings(embeddings)
    if cache_key is not None and resolved is not None:
        cached_embedding = resolved[0] if len(resolved) == 1 else torch.stack(resolved, dim=0)
        cache.put(cache_key, {"speaker_embedding": cached_embedding})
        logger.debug("Speaker cache STORE for Ming-TTS speaker '%s'", voice_name)
    return resolved


__all__ = [
    "_load_speaker_extractor",
    "_resolve_speaker_embeddings",
]
