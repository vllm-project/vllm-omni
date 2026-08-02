# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Reference-audio encoding + speaker cache for the MOSS-TTS-family talker.

This lives in the model package (not the shared serving layer) so all
MOSS-specific reference handling stays with the model — mirroring how Fish
Speech (``dac_encoder.encode_reference_audio_codes``), CosyVoice3, and
Qwen3-TTS keep their reference/speaker extraction next to the model rather than
in ``serving_speech.py``. The serving layer just calls
:func:`encode_reference_codes` with its generic helpers (the audio resolver and
the process-wide speaker cache).

Kept import-light (only ``asyncio`` / ``torch`` plus the logger)
so importing it from the API-server process does not pull the talker/codec.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)

# In-flight encodes, keyed by the same key the speaker cache uses so named
# voices and anonymous content-hashed references collapse identically. The
# cache is only written *after* an encode finishes, so without this table a
# burst of requests sharing one cold reference (the common voice-clone shape:
# one speaker, many utterances) each runs its own multi-second CPU codec pass.
# Single event loop per API-server process, so plain dict access between two
# awaits is atomic and needs no lock.
_INFLIGHT: dict[tuple[str, str, int], asyncio.Task[torch.Tensor]] = {}


def _encode_wav_sync(processor: Any, wav_list: list, sr: int, sr_target: int, n_vq: int) -> torch.Tensor:
    """Blocking resample + CPU codec encode (the expensive bit)."""
    wav = torch.tensor(wav_list, dtype=torch.float32)
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    if sr != sr_target:
        import torchaudio

        wav = torchaudio.functional.resample(wav, sr, sr_target)
    with torch.no_grad():
        codes_list = processor.encode_audios_from_wav([wav], sampling_rate=sr_target, n_vq=n_vq)
    return codes_list[0]


async def _resolve_and_encode(
    ref_str: str,
    *,
    processor: Any,
    resolve_ref_audio: Callable[[str], Awaitable[tuple[list, int, str]]],
    speaker_cache: Any,
    cache_key: tuple[str, str, int],
    speaker_name: str,
    n_vq: int,
    sr_target: int,
) -> torch.Tensor:
    """Resolve + encode one reference and publish it to the speaker cache.

    Runs as the single-flight leader task. Returns the same CPU tensor that was
    stored, so callers must clone before handing it downstream.
    """
    wav_list, sr, _ = await resolve_ref_audio(ref_str)
    return await _encode_and_store(
        processor=processor,
        wav_list=wav_list,
        sr=sr,
        speaker_cache=speaker_cache,
        cache_key=cache_key,
        speaker_name=speaker_name,
        n_vq=n_vq,
        sr_target=sr_target,
    )


async def _encode_and_store(
    *,
    processor: Any,
    wav_list: list,
    sr: int,
    speaker_cache: Any,
    cache_key: tuple[str, str, int],
    speaker_name: str,
    n_vq: int,
    sr_target: int,
) -> torch.Tensor:
    """Encode already-resolved audio and publish it to the speaker cache."""
    codes = await asyncio.to_thread(_encode_wav_sync, processor, wav_list, sr, sr_target, n_vq)
    codes = codes.detach().cpu()
    speaker_cache.put(cache_key, {"codes": codes})
    logger.debug("Speaker cache STORE for MOSS-TTS reference '%s'", speaker_name)
    return codes


def _retire_inflight(task: asyncio.Task[torch.Tensor], key: tuple[str, str, int]) -> None:
    """Drop the finished task and retrieve any exception.

    Removing the entry on failure means the next request retries rather than
    inheriting a poisoned slot — failures are never cached. Reading the
    exception here keeps asyncio quiet when every waiter was cancelled before
    the leader failed.
    """
    if _INFLIGHT.get(key) is task:
        _INFLIGHT.pop(key)
    if not task.cancelled():
        task.exception()


def _join_or_start(
    cache_key: tuple[str, str, int],
    speaker_name: str,
    encode_factory: Callable[[], Awaitable[torch.Tensor]],
) -> asyncio.Task[torch.Tensor]:
    """Return the current encode or create and register one without yielding."""
    task = _INFLIGHT.get(cache_key)
    if task is None:
        task = asyncio.ensure_future(encode_factory())
        _INFLIGHT[cache_key] = task
        task.add_done_callback(lambda t, k=cache_key: _retire_inflight(t, k))
    else:
        logger.debug("Speaker cache JOIN in-flight encode for MOSS-TTS reference '%s'", speaker_name)
    return task


async def encode_reference_codes(
    ref_str: str,
    *,
    processor: Any,
    resolve_ref_audio: Callable[[str], Awaitable[tuple[list, int, str]]],
    speaker_cache: Any,
    variant: str,
    n_vq: int,
    sr_target: int,
    voice_name: str | None = None,
    voice_created_at: int = 0,
) -> torch.Tensor:
    """Encode one reference clip into MOSS RVQ codes, reusing the speaker cache.

    The MOSS audio tokenizer sits on CPU (to spare ~6.7 GiB next to the 8B
    talker), so re-encoding the same reference is a fixed per-request cost that
    otherwise dominates the 8B voice-clone variants and serializes under
    concurrency. Mirror CosyVoice3 / Qwen3-TTS: cache by named voice when one is
    supplied (``voice_created_at`` invalidates on re-upload), else by a content
    hash of the reference. The blocking encode runs in a worker thread via
    ``asyncio.to_thread`` so cold/anonymous encodes from concurrent requests
    overlap instead of serializing on the event loop.

    For named voices the speaker cache is checked *before* resolving the audio,
    avoiding the decode cost when the cache is warm.  For anonymous references
    the audio is resolved first so the cache key incorporates mtime/size from
    a single stat (no TOCTOU window).

    Concurrent requests for the same *uncached* reference are single-flighted:
    the first starts the encode, the rest join its task. The cache alone cannot
    do this — it is written only after an encode completes, so a burst arriving
    on a cold reference would otherwise run N duplicate codec passes.

    Args:
        ref_str: The raw reference audio (URL / path / data URL) as received.
        processor: Upstream MOSS processor exposing ``encode_audios_from_wav``.
        resolve_ref_audio: Async callable mapping ``ref_str`` to ``(wav_list, sr, cache_key)``.
        speaker_cache: Process-wide ``SpeakerEmbeddingCache``.
        variant: MOSS sub-variant (``tts`` / ``ttsd`` / ...), namespaces the cache key.
        n_vq: Number of RVQ codebooks (also namespaces the cache key).
        sr_target: Target sample rate for the codec.
        voice_name: Named/uploaded voice, if any (enables stable cross-request caching).
        voice_created_at: Upload timestamp; bumps the cache slot on re-upload.

    Returns:
        The reference RVQ codes tensor (CPU), ready to pass to the processor.
    """
    model_type = f"moss_tts_{variant}_nq{n_vq}"

    # ---- Named-voice branch: check cache *before* resolving ----
    # When voice_name is set, the cache key is (voice_name, created_at) which
    # does not depend on the resolved audio content.  The serving layer only
    # passes voice_name for uploaded speakers without an inline ref_audio;
    # re-upload bumps created_at and clears the speaker-cache slot.
    if voice_name:
        speaker_name = voice_name
        created_at = int(voice_created_at)
        cache_key = speaker_cache.make_cache_key(
            speaker_name,
            model_type=model_type,
            created_at=created_at,
        )
        cached = speaker_cache.get(cache_key)
        if cached is not None:
            logger.debug("Speaker cache HIT for MOSS-TTS reference '%s'", speaker_name)
            return cached["codes"].clone()
        # The final key is already known, so named requests single-flight the
        # resolve and encode together.
        task = _join_or_start(
            cache_key,
            speaker_name,
            lambda: _resolve_and_encode(
                ref_str,
                processor=processor,
                resolve_ref_audio=resolve_ref_audio,
                speaker_cache=speaker_cache,
                cache_key=cache_key,
                speaker_name=speaker_name,
                n_vq=n_vq,
                sr_target=sr_target,
            ),
        )
    else:
        # ---- Anonymous branch: resolve first for content-addressed key ----
        # The resolve incorporates mtime/size from a single stat so the key
        # and the waveform always come from the same filesystem snapshot.
        wav_list, sr, resolve_cache_key = await resolve_ref_audio(ref_str)
        speaker_name = "ref:" + resolve_cache_key
        created_at = 0
        cache_key = speaker_cache.make_cache_key(
            speaker_name,
            model_type=model_type,
            created_at=created_at,
        )
        cached = speaker_cache.get(cache_key)
        if cached is not None:
            logger.debug("Speaker cache HIT for MOSS-TTS reference '%s'", speaker_name)
            return cached["codes"].clone()
        # Anonymous requests must resolve before joining so the in-flight
        # identity is the resolver-derived content key, not the raw locator.
        task = _join_or_start(
            cache_key,
            speaker_name,
            lambda: _encode_and_store(
                processor=processor,
                wav_list=wav_list,
                sr=sr,
                speaker_cache=speaker_cache,
                cache_key=cache_key,
                speaker_name=speaker_name,
                n_vq=n_vq,
                sr_target=sr_target,
            ),
        )

    # ``shield`` so a caller that gives up (client disconnect) neither cancels
    # the shared encode nor drags the other waiters down with it; the encode
    # still finishes and warms the cache. Every caller gets its own clone for
    # the same reason the cache-hit path clones: downstream may mutate.
    codes = await asyncio.shield(task)
    return codes.clone()
