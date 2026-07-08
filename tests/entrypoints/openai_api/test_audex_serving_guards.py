# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Serving-layer regressions for the Audex zero-codec-token contract.

A zero-codec-token request reaches the serving layer as a stream whose only
audio payloads are zero-size tensors. The streaming generator must not turn
that into a header-only/empty successful stream: the raw-bytes generator
raises (before any WAV header), and the SSE wrapper surfaces
``speech.audio.error`` instead of ``speech.audio.done``.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.entrypoints.openai.serving_speech import OmniOpenAIServingSpeech


def _serving(model_type: str = "audex") -> OmniOpenAIServingSpeech:
    serving = OmniOpenAIServingSpeech.__new__(OmniOpenAIServingSpeech)
    serving._tts_model_type = model_type
    serving._mark_ref_audio_artifact_ready_for_request = lambda request_id: None
    serving._discard_ref_audio_artifact_warmup = lambda request_id: None
    return serving


def _result(audio: torch.Tensor) -> SimpleNamespace:
    return SimpleNamespace(multimodal_output={"model_outputs": audio, "sr": torch.tensor(16000, dtype=torch.int32)})


async def _gen(*results):
    for res in results:
        yield res


async def _collect(chunks):
    out = []
    async for chunk in chunks:
        out.append(chunk)
    return out


@pytest.mark.asyncio
async def test_raw_stream_raises_on_empty_audio_before_any_header():
    serving = _serving()
    chunks = serving._generate_audio_chunks(
        _gen(_result(torch.empty(0)), _result(torch.empty(0))),
        request_id="req-empty",
        response_format="wav",
    )
    with pytest.raises(ValueError, match="no audio"):
        await _collect(chunks)


@pytest.mark.asyncio
async def test_raw_stream_yields_audio_normally_when_non_empty():
    serving = _serving()
    chunks = serving._generate_audio_chunks(
        _gen(_result(torch.zeros(320)), _result(torch.empty(0))),
        request_id="req-ok",
        response_format="wav",
    )
    collected = await _collect(chunks)
    # WAV header + one PCM chunk; the trailing empty chunk is skipped silently.
    assert len(collected) == 2
    assert collected[0][:4] == b"RIFF"
    assert len(collected[1]) == 320 * 2  # int16 PCM


@pytest.mark.asyncio
async def test_sse_stream_emits_error_not_done_on_empty_audio():
    serving = _serving()
    events = serving._generate_audio_sse_events(
        _gen(_result(torch.empty(0))),
        request_id="req-empty",
        response_format="pcm",
    )
    collected = await _collect(events)
    assert any("speech.audio.error" in event for event in collected)
    assert not any("speech.audio.done" in event for event in collected)


@pytest.mark.asyncio
async def test_non_audex_empty_chunks_keep_legacy_behavior():
    """Other models keep their existing semantics (no new rejection)."""
    serving = _serving(model_type="qwen3_tts")
    chunks = serving._generate_audio_chunks(
        _gen(_result(torch.empty(0))),
        request_id="req-other",
        response_format="pcm",
    )
    await _collect(chunks)  # must not raise
