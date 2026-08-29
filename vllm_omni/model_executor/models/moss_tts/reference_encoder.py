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


# ---------------------------------------------------------------------------
# MOSS-TTS-Realtime prompt building (serving path)
#
# The realtime talker's ``preprocess`` reads a pre-built ``(L, 1+channels)``
# grid: column 0 is the text/special token stream, columns ``1:`` are the
# reference audio-code grid (``codes.ref``); text past the prefill window streams
# one token/step via ``additional_information["ids"]["all"]``. The online serving
# path previously forwarded the raw ``text``/``prompt_audio_array``, which the
# talker ignores — it then prefilled a bare ``[1]`` placeholder with no text and
# produced unconditioned (flaky, mostly-garbled) audio. The builders below
# produce the same grid the offline path (``test_moss_tts._build_realtime_request``)
# and upstream ``end2end.py``/``streaming_mossttsrealtime.py`` produce.
# ---------------------------------------------------------------------------

# Prefill this many text tokens into the grid; the rest stream one/step during
# decode (mirrors the processor's ``delay_tokens_len`` and upstream
# ``MossTTSRealtimeInference._build_prefill_batch``).
_REALTIME_PREFILL_MAX_TEXT = 12

_RT_PROC_CACHE: dict[str, Any] = {}
_RT_CODEC_CACHE: dict[str, Any] = {}


def _load_realtime_processor(model_id_or_path: str) -> Any:
    """Load (and cache) the snapshot's ``MossTTSRealtimeProcessor``.

    It is not auto-discovered by ``AutoProcessor`` (no ``processor_config.json``
    in the snapshot), so load it from ``processing_mossttsrealtime.py`` directly.
    """
    cached = _RT_PROC_CACHE.get(model_id_or_path)
    if cached is not None:
        return cached

    import importlib.util
    import os
    import sys

    if os.path.isdir(model_id_or_path):
        snap_dir = model_id_or_path
    else:
        from huggingface_hub import snapshot_download

        snap_dir = snapshot_download(repo_id=model_id_or_path)

    proc_path = os.path.join(snap_dir, "processing_mossttsrealtime.py")
    spec = importlib.util.spec_from_file_location("_moss_rt_proc", proc_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id_or_path, trust_remote_code=True)
    proc = mod.MossTTSRealtimeProcessor(tokenizer=tokenizer)
    _RT_PROC_CACHE[model_id_or_path] = proc
    return proc


def _load_realtime_codec(codec_path: str) -> Any:
    """Lazily load (and cache) the MOSS audio tokenizer on CPU for reference
    encoding. The repo carries an ``auto_map`` so it loads via ``AutoModel``.
    """
    cached = _RT_CODEC_CACHE.get(codec_path)
    if cached is not None:
        return cached
    from transformers import AutoModel

    codec = AutoModel.from_pretrained(codec_path, trust_remote_code=True).to("cpu").eval()
    _RT_CODEC_CACHE[codec_path] = codec
    return codec


def encode_realtime_reference(
    wav_list: list,
    sr: int,
    *,
    codec_path: str,
    sr_target: int = 24000,
    n_vq: int = 16,
) -> torch.Tensor:
    """Blocking: encode a reference waveform into ``(T, n_vq)`` RVQ codes.

    Mirrors upstream ``StreamingTextToAudio.set_voice_prompt`` (resample →
    ``codec.batch_encode``). Returns ``(T, n_vq)`` int tensor on CPU.
    """
    wav = torch.tensor(wav_list, dtype=torch.float32)
    if wav.dim() > 1:
        wav = wav.reshape(-1)
    if sr != sr_target:
        import torchaudio

        wav = torchaudio.functional.resample(wav, sr, sr_target)
    codec = _load_realtime_codec(codec_path)
    with torch.no_grad():
        out = codec.batch_encode([wav], num_quantizers=n_vq)
    codes = out.audio_codes  # (n_vq, B=1, T)
    if out.audio_codes_lengths is not None:
        t = int(out.audio_codes_lengths[0].item())
        codes = codes[:, :, :t]
    codes = codes[:, 0, :].to(torch.int64).cpu()  # (n_vq, T)
    return codes.transpose(0, 1).contiguous()  # (T, n_vq)


def build_realtime_prompt(
    model_id_or_path: str,
    text: str,
    *,
    reference_audio_tokens: Any | None = None,
    seed: int | None = None,
    max_new_frames: int | None = None,
    prefill_max_text: int = _REALTIME_PREFILL_MAX_TEXT,
) -> tuple[list[int], dict[str, Any]]:
    """Return ``(prompt_token_ids, additional_information)`` for one realtime
    request, building the talker's ``(L, 1+channels)`` grid.

    ``reference_audio_tokens`` is the encoded reference clip ``(T, channels)``
    for voice cloning; ``None`` falls back to the default-voice system prompt
    (text content is still synthesised correctly — only timbre differs).
    """
    import numpy as np

    proc = _load_realtime_processor(model_id_or_path)
    tokenizer = proc.tokenizer
    n_cols = proc.channels + 1  # text/special column + per-codebook columns
    pad = int(proc.audio_channel_pad)
    bos = int(proc.audio_bos_token)

    ref_tokens = None
    if reference_audio_tokens is not None:
        if isinstance(reference_audio_tokens, torch.Tensor):
            ref_tokens = reference_audio_tokens.detach().cpu().numpy()
        else:
            ref_tokens = np.asarray(reference_audio_tokens)

    system_grid = proc.make_ensemble(prompt_audio_tokens=ref_tokens)  # (L_sys, n_cols)

    header_ids = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
    header_grid = np.full((len(header_ids), n_cols), pad, dtype=np.int64)
    header_grid[:, 0] = header_ids

    text_ids = tokenizer.encode(text, add_special_tokens=False)
    if not text_ids:
        raise ValueError("MOSS-TTS-Realtime: empty text after tokenisation")
    cur = min(len(text_ids), max(1, int(prefill_max_text)))
    text_grid = np.full((cur, n_cols), pad, dtype=np.int64)
    text_grid[:, 0] = text_ids[:cur]
    text_grid[-1, 1] = bos  # audio_bos on the last prefilled text token

    grid = np.concatenate([system_grid, header_grid, text_grid], axis=0)

    prompt_token_ids: list[int] = grid[:, 0].astype(np.int64).tolist()
    info: dict[str, Any] = {
        "codes": {"ref": torch.from_numpy(grid[:, 1:].astype(np.int64).copy())},  # (L, channels)
    }
    if seed is not None:
        info["seed"] = [int(seed)]
    if max_new_frames is not None:
        info["max_new_frames"] = [int(max_new_frames)]
    remaining = list(text_ids[cur:])
    if remaining:
        info["ids"] = {"all": remaining}

    logger.debug(
        "MOSS-TTS-Realtime prompt: %d grid rows (%d sys + %d header + %d text), %d streamed, ref=%s",
        grid.shape[0],
        system_grid.shape[0],
        header_grid.shape[0],
        text_grid.shape[0],
        len(remaining),
        ref_tokens is not None,
    )
    return prompt_token_ids, info
