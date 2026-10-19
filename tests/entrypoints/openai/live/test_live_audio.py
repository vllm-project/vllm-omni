# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for Live session audio format negotiation and transcoding."""

from __future__ import annotations

import base64

import numpy as np
import pytest

from vllm_omni.entrypoints.openai.live import audio
from vllm_omni.entrypoints.openai.live.audio import (
    InputAudioDecoder,
    LiveAudioFormat,
    OutputAudioEncoder,
    ResampleStream,
    parse_audio_format,
    resample,
)
from vllm_omni.entrypoints.openai.live.protocol import LiveProtocolError

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

PCM16_STEP = 1.0 / 32767.0


def _tone(rate: int, seconds: float = 0.1, amplitude: float = 0.5) -> np.ndarray:
    t = np.arange(int(rate * seconds), dtype=np.float32) / rate
    return (amplitude * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)


def _pcm16_b64(values: list[int]) -> str:
    return base64.b64encode(np.asarray(values, dtype="<i2").tobytes()).decode()


class TestParseAudioFormat:
    def test_none_selects_24k_pcm(self) -> None:
        assert parse_audio_format(None) == LiveAudioFormat("audio/pcm", 24_000)

    @pytest.mark.parametrize("rate", [16_000, 24_000])
    def test_pcm_rates(self, rate: int) -> None:
        fmt = parse_audio_format({"type": "audio/pcm", "rate": rate})
        assert fmt == LiveAudioFormat("audio/pcm", rate)
        assert fmt.to_dict() == {"type": "audio/pcm", "rate": rate}

    @pytest.mark.parametrize("fmt_type", ["audio/pcmu", "audio/pcma"])
    @pytest.mark.parametrize("rate", [None, 8_000])
    def test_g711_is_8k(self, fmt_type: str, rate: int | None) -> None:
        raw: dict[str, object] = {"type": fmt_type} if rate is None else {"type": fmt_type, "rate": rate}
        assert parse_audio_format(raw) == LiveAudioFormat(fmt_type, 8_000)

    @pytest.mark.parametrize(
        "raw",
        [
            "audio/pcm",
            ["audio/pcm"],
            {"type": "audio/pcm"},
            {"type": "audio/pcm", "rate": 8_000},
            {"type": "audio/pcm", "rate": 48_000},
            {"type": "audio/pcm", "rate": "16000"},
            {"type": "audio/pcmu", "rate": 16_000},
            {"type": "audio/opus", "rate": 24_000},
            {},
        ],
    )
    def test_rejects_unsupported(self, raw: object) -> None:
        with pytest.raises(LiveProtocolError) as exc_info:
            parse_audio_format(raw)
        assert exc_info.value.code == "unsupported_session_config"
        assert exc_info.value.param == "audio.format"


class TestInputAudioDecoder:
    def test_pcm16_samples_and_duration(self) -> None:
        decoder = InputAudioDecoder(LiveAudioFormat("audio/pcm", 16_000))
        samples, ms = decoder.decode(_pcm16_b64([0, 16384, -32768, 32767]))
        assert samples.dtype == np.float32
        np.testing.assert_allclose(samples, [0.0, 0.5, -1.0, 32767 / 32768], atol=1e-6)
        assert ms == pytest.approx(4 * 1000 / 16_000)

    @pytest.mark.parametrize("rate", [16_000, 24_000])
    def test_pcm_duration(self, rate: int) -> None:
        decoder = InputAudioDecoder(LiveAudioFormat("audio/pcm", rate))
        samples, ms = decoder.decode(_pcm16_b64([0] * (rate // 10)))
        assert samples.size == rate // 10
        assert ms == pytest.approx(100.0)

    @pytest.mark.parametrize("fmt_type", ["audio/pcmu", "audio/pcma"])
    def test_g711_one_byte_per_sample(self, fmt_type: str) -> None:
        decoder = InputAudioDecoder(LiveAudioFormat(fmt_type, 8_000))
        samples, ms = decoder.decode(base64.b64encode(bytes(800)).decode())
        assert samples.size == 800
        assert ms == pytest.approx(100.0)

    @pytest.mark.parametrize("value", [None, 123, b"AAAA", ["AAAA"]])
    def test_rejects_non_string(self, value: object) -> None:
        decoder = InputAudioDecoder(LiveAudioFormat("audio/pcm", 16_000))
        with pytest.raises(LiveProtocolError, match="base64 string") as exc_info:
            decoder.decode(value)
        assert (exc_info.value.code, exc_info.value.param) == ("invalid_audio", "audio")

    @pytest.mark.parametrize("value", ["not base64!", "AAA", "AA=A"])
    def test_rejects_invalid_base64(self, value: str) -> None:
        decoder = InputAudioDecoder(LiveAudioFormat("audio/pcm", 16_000))
        with pytest.raises(LiveProtocolError, match="not valid base64") as exc_info:
            decoder.decode(value)
        assert exc_info.value.code == "invalid_audio"

    def test_rejects_partial_pcm16_sample(self) -> None:
        decoder = InputAudioDecoder(LiveAudioFormat("audio/pcm", 16_000))
        with pytest.raises(LiveProtocolError, match="whole number of samples") as exc_info:
            decoder.decode(base64.b64encode(b"\x00\x00\x00").decode())
        assert exc_info.value.code == "invalid_audio"

    def test_rejects_oversized_append(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(audio, "MAX_APPEND_BYTES", 30)
        decoder = InputAudioDecoder(LiveAudioFormat("audio/pcm", 16_000))
        decoder.decode(_pcm16_b64([0] * 15))
        with pytest.raises(LiveProtocolError, match="too large") as exc_info:
            decoder.decode(_pcm16_b64([0] * 18))
        assert exc_info.value.code == "invalid_audio"

    def test_empty_append(self) -> None:
        samples, ms = InputAudioDecoder(LiveAudioFormat("audio/pcm", 24_000)).decode("")
        assert samples.size == 0
        assert ms == 0.0


class TestOutputAudioEncoder:
    def test_pcm_round_trip_at_model_rate(self) -> None:
        fmt = LiveAudioFormat("audio/pcm", 24_000)
        tone = _tone(24_000)
        decoded, ms = InputAudioDecoder(fmt).decode(OutputAudioEncoder(fmt, 24_000).encode(tone))
        np.testing.assert_allclose(decoded, tone, atol=PCM16_STEP)
        assert ms == pytest.approx(100.0)

    def test_clips_out_of_range(self) -> None:
        fmt = LiveAudioFormat("audio/pcm", 16_000)
        raw = base64.b64decode(OutputAudioEncoder(fmt, 16_000).encode(np.array([2.0, -2.0, 0.0], dtype=np.float32)))
        assert np.frombuffer(raw, dtype="<i2").tolist() == [32767, -32767, 0]

    def test_accepts_2d_input(self) -> None:
        fmt = LiveAudioFormat("audio/pcm", 16_000)
        flat = OutputAudioEncoder(fmt, 16_000).encode(np.zeros(8, dtype=np.float32))
        assert OutputAudioEncoder(fmt, 16_000).encode(np.zeros((1, 8), dtype=np.float32)) == flat

    def test_resamples_model_rate_to_session_rate(self) -> None:
        fmt = LiveAudioFormat("audio/pcm", 16_000)
        encoder = OutputAudioEncoder(fmt, 24_000)
        tone = _tone(24_000, seconds=0.5)
        chunks = [InputAudioDecoder(fmt).decode(encoder.encode(part))[0] for part in np.array_split(tone, 5)]
        out = np.concatenate(chunks)
        expected = resample(tone, 24_000, 16_000)
        # The streaming resampler holds back its filter tail until more input arrives.
        assert 0 < expected.size - out.size < 100
        np.testing.assert_allclose(out, expected[: out.size], atol=2 * PCM16_STEP)

    @pytest.mark.parametrize(("fmt_type", "atol"), [("audio/pcmu", 0.02), ("audio/pcma", 0.02)])
    def test_g711_round_trip(self, fmt_type: str, atol: float) -> None:
        fmt = parse_audio_format({"type": fmt_type})
        tone = _tone(8_000)
        payload = OutputAudioEncoder(fmt, 8_000).encode(tone)
        assert len(base64.b64decode(payload)) == tone.size
        decoded, ms = InputAudioDecoder(fmt).decode(payload)
        np.testing.assert_allclose(decoded, tone, atol=atol)
        assert ms == pytest.approx(100.0)


class TestResample:
    def test_stream_identity_when_rates_match(self) -> None:
        tone = _tone(16_000)
        assert ResampleStream(16_000, 16_000).process(tone) is tone

    @pytest.mark.parametrize(("source", "target"), [(24_000, 16_000), (16_000, 24_000), (24_000, 8_000)])
    def test_chunked_stream_matches_one_shot_stream(self, source: int, target: int) -> None:
        tone = _tone(source, seconds=0.3)
        chunked = ResampleStream(source, target)
        out = np.concatenate([chunked.process(part) for part in np.array_split(tone, 7)])
        np.testing.assert_array_equal(out, ResampleStream(source, target).process(tone))

    @pytest.mark.parametrize(("source", "target"), [(24_000, 16_000), (16_000, 24_000)])
    def test_one_shot_length(self, source: int, target: int) -> None:
        tone = _tone(source, seconds=0.25)
        assert resample(tone, source, target).size == tone.size * target // source

    def test_one_shot_passthrough(self) -> None:
        tone = _tone(16_000)
        np.testing.assert_array_equal(resample(tone, 16_000, 16_000), tone)
        assert resample(np.zeros(0, dtype=np.float32), 24_000, 16_000).size == 0
