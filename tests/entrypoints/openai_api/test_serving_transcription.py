# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for word-level alignment on ``/v1/audio/transcriptions``.

Covers the three things this subclass adds on top of upstream serving: the
decoded-waveform handoff from the ASR pass to the aligner, synthesising a
verbose envelope for models that cannot emit segment timestamps, and degrading
to a plain transcript when alignment is unavailable or fails.

Upstream's ``create_transcription`` is stubbed; these run on CPU with no
weights and no engine.
"""

import asyncio
import io
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
from vllm.entrypoints.speech_to_text.transcription.protocol import (
    TranscriptionResponse,
    TranscriptionResponseVerbose,
)

from vllm_omni.entrypoints.openai import serving_transcription as st
from vllm_omni.entrypoints.openai.serving_transcription import OmniServingTranscription
from vllm_omni.utils.forced_aligner import ForcedAlignerConfig, ForcedAlignerLoadError, WordTimestamp

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

SAMPLE_RATE = 16000


def _wav_bytes(seconds: float = 1.0, sr: int = SAMPLE_RATE) -> bytes:
    """A real WAV container, so the decode helpers exercise real soundfile."""
    t = np.linspace(0, seconds, int(seconds * sr), endpoint=False, dtype=np.float32)
    buf = io.BytesIO()
    sf.write(buf, 0.25 * np.sin(2 * np.pi * 440 * t), sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _resp(text: str) -> TranscriptionResponse:
    """A valid upstream json response; `usage` is required by the model."""
    return TranscriptionResponse(text=text, usage={"type": "duration", "seconds": 1})


def _serving(*, aligner: bool = True, segment_timestamps: bool = False) -> OmniServingTranscription:
    """An instance with only the state these paths touch.

    Built via __new__ because the real __init__ needs an engine client, a model
    config and a loaded model; none of that participates in the logic here.
    """
    obj = OmniServingTranscription.__new__(OmniServingTranscription)
    obj.forced_aligner_config = ForcedAlignerConfig(model="stub") if aligner else None
    obj._decoded = __import__("collections").OrderedDict()
    obj._pending = 0
    obj._decoded_lock = __import__("threading").Lock()
    obj._reuse_hits = 0
    obj._reuse_misses = 0
    obj._decode_pool = None
    # The reuse path takes the model's sample rate from here.
    obj.asr_config = SimpleNamespace(sample_rate=SAMPLE_RATE)
    # `model_cls` is a cached_property upstream; set the attribute directly.
    obj.__dict__["model_cls"] = SimpleNamespace(supports_segment_timestamp=segment_timestamps)
    return obj


def _request(*, granularities=("word",), response_format="verbose_json", language=None):
    return SimpleNamespace(
        timestamp_granularities=list(granularities),
        response_format=response_format,
        language=language,
        model_copy=lambda update: _request(
            granularities=granularities,
            response_format=update.get("response_format", response_format),
            language=language,
        ),
    )


@pytest.fixture
def stub_upstream(monkeypatch):
    """Stub the upstream create_transcription this subclass delegates to."""

    def _apply(result):
        async def fake(self, *, audio_data, request, raw_request=None):
            return result

        monkeypatch.setattr(
            "vllm.entrypoints.speech_to_text.transcription.serving.OpenAIServingTranscription.create_transcription",
            fake,
        )

    return _apply


@pytest.fixture
def stub_align(monkeypatch):
    """Stub the aligner, returning what it was called with for assertions."""

    def _apply(result=None, *, raises=None):
        calls: list[dict] = []

        async def fake(*, audio, text, sample_rate, config, language=None):
            calls.append({"audio": audio, "text": text, "sample_rate": sample_rate, "language": language})
            if raises is not None:
                raise raises
            return result

        monkeypatch.setattr(st, "forced_align", fake)
        return calls

    return _apply


# --- request gating -------------------------------------------------------


@pytest.mark.parametrize(
    ("granularities", "response_format", "expected"),
    [
        (("word",), "verbose_json", True),
        (("word", "segment"), "verbose_json", True),
        (("segment",), "verbose_json", False),
        ((), "verbose_json", False),
        (("word",), "json", False),  # words are only expressible in verbose_json
    ],
)
def test_wants_word_timestamps(granularities, response_format, expected):
    request = _request(granularities=granularities, response_format=response_format)
    assert OmniServingTranscription._wants_word_timestamps(request) is expected


async def test_plain_request_is_passed_through_untouched(stub_upstream, stub_align):
    """The default path must not pay for alignment it did not ask for."""
    stub_upstream(_resp("hello world"))
    calls = stub_align([WordTimestamp("hello", 0, 100)])

    got = await _serving().create_transcription(
        audio_data=_wav_bytes(), request=_request(granularities=(), response_format="json")
    )

    assert isinstance(got, TranscriptionResponse)
    assert got.text == "hello world"
    assert calls == [], "aligner ran for a request that did not ask for timestamps"


# --- verbose envelope synthesis -------------------------------------------


async def test_verbose_envelope_is_synthesised_when_model_lacks_segment_timestamps(stub_upstream, stub_align):
    """Upstream 400s on verbose_json without segment support, so we downgrade the
    inner call to json and build the envelope ourselves. Qwen3-ASR sets
    supports_segment_timestamp=False, so this is the normal path, not an edge."""
    stub_upstream(_resp("hello world"))
    stub_align([WordTimestamp("hello", 0, 500), WordTimestamp("world", 500, 1000)])

    got = await _serving(segment_timestamps=False).create_transcription(
        audio_data=_wav_bytes(seconds=1.0), request=_request()
    )

    assert isinstance(got, TranscriptionResponseVerbose)
    assert got.text == "hello world"
    assert got.segments is None
    assert [w.word for w in got.words] == ["hello", "world"]
    # ms on the aligner's side of the boundary, seconds on the OpenAI side.
    assert (got.words[0].start, got.words[0].end) == (0.0, 0.5)
    assert float(got.duration) == pytest.approx(1.0, abs=0.05)


async def test_language_hint_is_echoed_and_defaults_to_auto(stub_upstream, stub_align):
    """Qwen3-ASR auto-detects and does not report the language back."""
    stub_upstream(_resp("bonjour"))
    stub_align([WordTimestamp("bonjour", 0, 100)])
    serving = _serving()

    got = await serving.create_transcription(audio_data=_wav_bytes(), request=_request(language="fr"))
    assert got.language == "fr"

    got = await serving.create_transcription(audio_data=_wav_bytes(), request=_request(language=None))
    assert got.language == "auto"


async def test_error_from_the_inner_call_passes_straight_through(stub_upstream, stub_align):
    error = SimpleNamespace(error="boom")
    stub_upstream(error)
    calls = stub_align([WordTimestamp("x", 0, 1)])

    got = await _serving().create_transcription(audio_data=_wav_bytes(), request=_request())

    assert got is error
    assert calls == [], "aligned an error response"


# --- degradation ----------------------------------------------------------


async def test_words_requested_without_an_aligner_returns_the_transcript(stub_upstream, caplog):
    """Serving without --forced-aligner should not fail the request."""
    stub_upstream(_resp("hello"))

    got = await _serving(aligner=False).create_transcription(audio_data=_wav_bytes(), request=_request())

    assert got.text == "hello"
    assert getattr(got, "words", None) is None
    assert "no forced aligner configured" in caplog.text


async def test_alignment_failure_still_returns_the_transcript(stub_upstream, stub_align):
    """A per-request aligner failure is None; the text is already correct."""
    stub_upstream(_resp("hello"))
    stub_align(None)

    got = await _serving().create_transcription(audio_data=_wav_bytes(), request=_request())

    assert got.text == "hello"
    assert got.words is None


async def test_empty_transcript_yields_no_words_without_calling_the_aligner(stub_upstream, stub_align):
    stub_upstream(_resp("   "))
    calls = stub_align([WordTimestamp("x", 0, 1)])

    got = await _serving().create_transcription(audio_data=_wav_bytes(), request=_request())

    assert got.words == []
    assert calls == [], "aligned an empty transcript"


async def test_load_failure_propagates(stub_upstream, stub_align):
    """A missing or unallocatable aligner is permanent, so it must not be
    silently degraded to a transcript without words on every request."""
    stub_upstream(_resp("hello"))
    stub_align(raises=ForcedAlignerLoadError("cannot allocate"))

    with pytest.raises(ForcedAlignerLoadError):
        await _serving().create_transcription(audio_data=_wav_bytes(), request=_request())


# --- decoded-waveform handoff ---------------------------------------------


def test_decode_handoff_round_trips():
    serving = _serving()
    audio, wave = _wav_bytes(), np.ones(8, dtype=np.float32)

    serving._stash_decoded(audio, [wave], 1.5)
    got = serving._take_decoded(audio)

    assert got is not None and got[1] == 1.5
    assert np.array_equal(got[0], wave)
    assert serving._pending == 0, "handoff not reclaimed"


def test_identical_audio_from_concurrent_requests_does_not_collide():
    """Regression test for a 65% miss rate under load.

    The key is a content hash, so retries, replayed fixtures and the same file
    submitted twice land on one key. With a single slot per key each decode
    clobbered the last and only one consumer could pop it, so the other
    requests silently decoded the upload a second time.
    """
    serving = _serving()
    audio = _wav_bytes()
    waves = [np.full(4, i, dtype=np.float32) for i in range(3)]

    for w in waves:
        serving._stash_decoded(audio, [w], 1.0)

    got = [serving._take_decoded(audio) for _ in range(3)]

    assert all(g is not None for g in got), "concurrent identical audio lost a handoff"
    assert serving._reuse_misses == 0
    assert serving._pending == 0


def test_chunked_audio_is_not_reused():
    """Chunks overlap by design, so rejoining them would not reconstruct the
    original waveform. Better to decode again than to align against a splice."""
    serving = _serving()
    audio = _wav_bytes()

    serving._stash_decoded(audio, [np.ones(4), np.ones(4)], 2.0)

    assert serving._take_decoded(audio) is None


def test_handoff_is_bounded_against_leaks():
    """A request dying between decode and pickup would otherwise pin its
    waveform forever."""
    serving = _serving()
    serving._MAX_PENDING_DECODES = 4
    for i in range(10):
        serving._stash_decoded(_wav_bytes(seconds=0.1 + i * 0.01), [np.ones(2, dtype=np.float32)], 1.0)

    assert serving._pending <= 4


async def test_alignment_reuses_the_asr_decode_rather_than_decoding_again(stub_upstream, stub_align, monkeypatch):
    """Decoding the upload twice roughly halves the ceiling of the aligned
    path, since decode is GIL-bound."""
    stub_upstream(_resp("hello"))
    calls = stub_align([WordTimestamp("hello", 0, 100)])

    def fail(*a, **k):
        raise AssertionError("fell back to decoding the upload a second time")

    monkeypatch.setattr(st, "_to_int16_pcm", fail)

    serving = _serving()
    audio = _wav_bytes()
    wave = np.zeros(SAMPLE_RATE, dtype=np.float32)
    serving._stash_decoded(audio, [wave], 1.0)

    got = await serving.create_transcription(audio_data=audio, request=_request())

    assert [w.word for w in got.words] == ["hello"]
    assert calls[0]["sample_rate"] == SAMPLE_RATE


async def test_falls_back_to_decoding_when_no_handoff_is_available(stub_upstream, stub_align):
    """Chunked audio and cache misses must still align, just more expensively."""
    stub_upstream(_resp("hello"))
    calls = stub_align([WordTimestamp("hello", 0, 100)])

    got = await _serving().create_transcription(audio_data=_wav_bytes(), request=_request())

    assert [w.word for w in got.words] == ["hello"]
    assert calls[0]["audio"], "aligner received no audio on the fallback path"


# --- audio conversion -----------------------------------------------------


def test_to_int16_pcm_downmixes_and_resamples():
    sr = 48000
    t = np.linspace(0, 0.5, int(0.5 * sr), endpoint=False, dtype=np.float32)
    stereo = np.stack([0.5 * np.sin(2 * np.pi * 440 * t), 0.5 * np.sin(2 * np.pi * 440 * t)], axis=1)
    buf = io.BytesIO()
    sf.write(buf, stereo, sr, format="WAV", subtype="PCM_16")

    pcm = st._to_int16_pcm(buf.getvalue())

    samples = np.frombuffer(pcm, dtype="<i2")
    assert len(samples) == pytest.approx(0.5 * SAMPLE_RATE, rel=0.01), "not resampled to 16 kHz mono"


def test_float32_to_int16_pcm_clips_out_of_range_input():
    """Values outside [-1, 1] would wrap rather than saturate without the clip."""
    pcm = st._float32_to_int16_pcm(np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32))

    samples = np.frombuffer(pcm, dtype="<i2")
    assert samples[0] == samples[1] == -32767
    assert samples[3] == samples[4] == 32767


def test_audio_key_is_content_addressed():
    audio = _wav_bytes()
    assert st._audio_key(audio) == st._audio_key(bytes(audio))
    assert st._audio_key(audio) != st._audio_key(_wav_bytes(seconds=2.0))


def test_duration_seconds_reads_the_container():
    assert st._duration_seconds(_wav_bytes(seconds=1.5)) == pytest.approx(1.5, abs=0.01)


async def test_concurrent_requests_each_get_their_own_words(stub_upstream, stub_align):
    """End-to-end guard against handoffs or results crossing between requests."""
    stub_upstream(_resp("hello"))
    stub_align([WordTimestamp("hello", 0, 100)])
    serving = _serving()

    results = await asyncio.gather(
        *(serving.create_transcription(audio_data=_wav_bytes(), request=_request()) for _ in range(8))
    )

    assert all([w.word for w in r.words] == ["hello"] for r in results)
    assert serving._pending == 0
