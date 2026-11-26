# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import threading
from dataclasses import dataclass, field
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
    SPEAKER_EMBEDDING_DIM,
)

_LOAD_LOCK = threading.Lock()
# TODO: Move shared lazy CAM++ loading and locking into common.ming.speaker_extractor.


def _cached_embeddings(value: Any) -> list[torch.Tensor] | None:
    """Embeddings from the speaker cache, left on their device.

    The cache holds the GPU encoder's embeddings; ``coerce_speaker_embeddings``
    moves rows to the host, which would wait for all queued GPU work.
    """
    if isinstance(value, torch.Tensor) and value.ndim in (1, 2) and value.shape[-1] == SPEAKER_EMBEDDING_DIM:
        return list(value.detach().float().reshape(-1, SPEAKER_EMBEDDING_DIM))
    return coerce_speaker_embeddings(value)


@dataclass
class SpeakerPlan:
    """A request's speaker embeddings: those known now, and the reference clips still to embed."""

    embeddings: list[torch.Tensor | None]
    # (embedding index, waveform [1, samples], sample rate, reference cache key)
    clips: list[tuple[int, torch.Tensor, int, Any]] = field(default_factory=list)
    voice_key: Any = None


def plan_speaker_embeddings(wrapper: Any, info: dict[str, Any]) -> SpeakerPlan:
    direct = info.get(KEY_SPEAKER_EMBEDDING)
    if direct is not None:
        embeddings = coerce_speaker_embeddings(direct)
        if embeddings:
            return SpeakerPlan(list(embeddings))

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
            embeddings = _cached_embeddings(cached.get("speaker_embedding"))
            if embeddings:
                return SpeakerPlan(list(embeddings))

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
    plan = SpeakerPlan([None] * lengths.numel(), voice_key=cache_key)
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
                extracted = _cached_embeddings(cached.get("speaker_embedding"))
                if len(extracted) != 1:
                    raise RuntimeError("Invalid cached Ming3 reference speaker embedding")
                plan.embeddings[index] = extracted[0]
                offset += length
                continue
        plan.clips.append((index, waveform[:, offset : offset + length], int(sample_rate), reference_key))
        offset += length
    return plan


def extract_speaker_embeddings(wrapper: Any, clips: list[tuple[torch.Tensor, int]]) -> list[torch.Tensor]:
    """CAM++ embeddings of ``(waveform, sample rate)`` clips: batched on the GPU when the model has the encoder."""
    encoder = getattr(wrapper, "_speaker_encoder", None)
    if encoder is not None:
        return encoder.embed(clips)
    extractor = getattr(wrapper, "_speaker_extractor", None)
    if extractor is None:
        with _LOAD_LOCK:
            extractor = getattr(wrapper, "_speaker_extractor", None)
            if extractor is None:
                extractor = SpeakerEmbeddingExtractor(wrapper.vllm_config.model_config.model, target_sr=16000)
                wrapper._speaker_extractor = extractor
    embeddings = []
    for waveform, sample_rate in clips:
        extracted = coerce_speaker_embeddings(extractor.extract_from_waveform(waveform, sample_rate))
        if len(extracted) != 1:
            raise RuntimeError("CAM++ must return one Ming3 speaker embedding per reference clip")
        embeddings.extend(extracted)
    return embeddings


def finish_speaker_embeddings(wrapper: Any, plan: SpeakerPlan, extracted: list[torch.Tensor]) -> list[torch.Tensor]:
    """Fill ``plan`` with the embeddings of its clips and cache them."""
    for (index, _, _, reference_key), embedding in zip(plan.clips, extracted, strict=True):
        if reference_key is not None:
            wrapper._speaker_cache.put(reference_key, {"speaker_embedding": embedding})
        plan.embeddings[index] = embedding
    if plan.voice_key is not None:
        wrapper._speaker_cache.put(plan.voice_key, {"speaker_embedding": plan.embeddings[0]})
    return plan.embeddings


def resolve_speaker_embeddings(wrapper: Any, info: dict[str, Any]) -> list[torch.Tensor]:
    plan = plan_speaker_embeddings(wrapper, info)
    if not plan.clips:
        return plan.embeddings
    extracted = extract_speaker_embeddings(wrapper, [(waveform, rate) for _, waveform, rate, _ in plan.clips])
    return finish_speaker_embeddings(wrapper, plan, extracted)


__all__ = [
    "SpeakerPlan",
    "extract_speaker_embeddings",
    "finish_speaker_embeddings",
    "plan_speaker_embeddings",
    "resolve_speaker_embeddings",
]
