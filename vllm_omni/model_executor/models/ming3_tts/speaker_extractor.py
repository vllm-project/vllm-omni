# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import threading
from typing import Any

import torch

from vllm_omni.model_executor.models.common.ming.speaker_extractor import (
    SpeakerEmbeddingExtractor,
)
from vllm_omni.model_executor.models.ming_tts.audio_prep import (
    coerce_prompt_waveform,
    coerce_speaker_embeddings,
)

from .constants import (
    KEY_SPEAKER_EMBEDDING,
    KEY_SPEAKER_SAMPLE_RATES,
    KEY_SPEAKER_WAVEFORM,
    KEY_SPEAKER_WAVEFORM_LENGTHS,
)

_LOAD_LOCK = threading.Lock()
# TODO: Move shared lazy CAM++ loading and locking into common.ming.speaker_extractor.


def resolve_speaker_embeddings(wrapper: Any, info: dict[str, Any]) -> list[torch.Tensor]:
    direct = info.get(KEY_SPEAKER_EMBEDDING)
    if direct is not None:
        embeddings = coerce_speaker_embeddings(direct)
        if embeddings:
            return embeddings

    voice_name = info.get("voice_name")
    if isinstance(voice_name, (list, tuple)):
        voice_name = voice_name[0] if voice_name else None
    created_at = info.get("voice_created_at", 0)
    if isinstance(created_at, (list, tuple)):
        created_at = created_at[0] if created_at else 0
    if isinstance(created_at, torch.Tensor):
        created_at = created_at.detach().reshape(-1)[0].item() if created_at.numel() else 0

    cache_key = None
    if isinstance(voice_name, str) and voice_name.strip():
        cache = wrapper._speaker_cache
        cache_key = cache.make_cache_key(voice_name, model_type="ming3_tts", created_at=int(created_at or 0))
        cached = cache.get(cache_key)
        if cached is not None:
            embeddings = coerce_speaker_embeddings(cached.get("speaker_embedding"))
            if embeddings:
                return embeddings

    waveform = info.get(KEY_SPEAKER_WAVEFORM)
    if waveform is None:
        raise ValueError("Ming-3.0-TTS requires reference audio for speaker conditioning")
    waveform = coerce_prompt_waveform(waveform)
    lengths = torch.as_tensor(info.get(KEY_SPEAKER_WAVEFORM_LENGTHS, [waveform.shape[-1]])).reshape(-1)
    sample_rates = torch.as_tensor(info.get(KEY_SPEAKER_SAMPLE_RATES, [wrapper.ming_config.sample_rate])).reshape(-1)
    if lengths.numel() == 0 or lengths.numel() != sample_rates.numel():
        raise ValueError("Each Ming3 reference audio clip requires a length and sample rate")
    if (lengths <= 0).any() or (sample_rates <= 0).any() or int(lengths.sum()) != waveform.shape[-1]:
        raise ValueError("Invalid Ming3 reference audio lengths or sample rates")
    if cache_key is not None and lengths.numel() != 1:
        raise ValueError("A named Ming3 voice must contain exactly one reference audio clip")

    audio_keys = info.get("speaker_audio_keys")
    if audio_keys is not None and len(audio_keys) != lengths.numel():
        raise ValueError("Ming3 speaker cache keys must match the reference audio clips")
    embeddings = []
    offset = 0
    for index, (length, sample_rate) in enumerate(zip(lengths.tolist(), sample_rates.tolist(), strict=True)):
        reference_key = None
        if audio_keys is not None and audio_keys[index]:
            cache = wrapper._speaker_cache
            reference_key = cache.make_cache_key(
                audio_keys[index], model_type=f"ming3_tts.reference:{wrapper.vllm_config.model_config.model}"
            )
            cached = cache.get(reference_key)
            if cached is not None:
                extracted = coerce_speaker_embeddings(cached.get("speaker_embedding"))
                if len(extracted) != 1:
                    raise RuntimeError("Invalid cached Ming3 reference speaker embedding")
                embeddings.extend(extracted)
                offset += length
                continue
        extractor = getattr(wrapper, "_speaker_extractor", None)
        if extractor is None:
            with _LOAD_LOCK:
                extractor = getattr(wrapper, "_speaker_extractor", None)
                if extractor is None:
                    extractor = SpeakerEmbeddingExtractor(wrapper.vllm_config.model_config.model, target_sr=16000)
                    wrapper._speaker_extractor = extractor
        embedding = extractor.extract_from_waveform(waveform[:, offset : offset + length], sample_rate)
        extracted = coerce_speaker_embeddings(embedding)
        if len(extracted) != 1:
            raise RuntimeError("CAM++ must return one Ming3 speaker embedding per reference clip")
        if reference_key is not None:
            cache.put(reference_key, {"speaker_embedding": extracted[0]})
        embeddings.extend(extracted)
        offset += length
    if cache_key is not None:
        cache.put(cache_key, {"speaker_embedding": embeddings[0]})
    return embeddings


__all__ = ["resolve_speaker_embeddings"]
