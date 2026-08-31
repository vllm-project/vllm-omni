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
        # Cache miss — resolve the audio for encoding.
        wav_list, sr, _ = await resolve_ref_audio(ref_str)
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

    # ---- Cold miss: encode in a worker thread and store ----
    codes = await asyncio.to_thread(_encode_wav_sync, processor, wav_list, sr, sr_target, n_vq)
    speaker_cache.put(cache_key, {"codes": codes.detach().cpu()})
    logger.debug("Speaker cache STORE for MOSS-TTS reference '%s'", speaker_name)
    return codes
