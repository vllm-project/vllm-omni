import base64
import wave
from urllib.parse import parse_qs, urlsplit

import pytest

from vllm_omni.experimental.fullduplex.client import (
    RealtimeDuplexClient,
    RealtimeEventCollector,
    build_realtime_url,
    read_pcm16_wav,
    write_pcm16_wav,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_realtime_client_builds_explicit_native_duplex_url():
    url = build_realtime_url(
        "ws://localhost:8099/v1/realtime?custom=1",
        "openbmb/MiniCPM-o-4_5",
        session_id="session-a",
    )

    query = parse_qs(urlsplit(url).query)
    assert query == {
        "custom": ["1"],
        "duplex": ["1"],
        "model": ["openbmb/MiniCPM-o-4_5"],
        "minicpmo45_native_duplex": ["1"],
        "session_id": ["session-a"],
    }


def test_realtime_client_builds_resume_only_url_when_autostart_disabled():
    url = build_realtime_url(
        "ws://localhost:8099/v1/realtime?duplex=1",
        "openbmb/MiniCPM-o-4_5",
        autostart=False,
    )

    query = parse_qs(urlsplit(url).query)
    assert query["autostart"] == ["0"]
    assert query["minicpmo45_native_duplex"] == ["1"]


@pytest.mark.asyncio
async def test_realtime_client_configure_omits_ref_audio_by_default():
    class Client(RealtimeDuplexClient):
        def __init__(self):
            super().__init__("ws://unused")
            self.sent = []

        async def send(self, event):
            self.sent.append(event)
            self.events.add({"type": "session.created"})

    client = Client()

    await client.configure("openbmb/MiniCPM-o-4_5", timeout_s=1)

    session = client.sent[0]["session"]
    assert "ref_audio" not in session


@pytest.mark.asyncio
async def test_realtime_client_configure_sends_explicit_ref_audio():
    class Client(RealtimeDuplexClient):
        def __init__(self):
            super().__init__("ws://unused")
            self.sent = []

        async def send(self, event):
            self.sent.append(event)
            self.events.add({"type": "session.created"})

    client = Client()

    await client.configure(
        "openbmb/MiniCPM-o-4_5",
        ref_audio="data:audio/wav;base64,AAAA",
        timeout_s=1,
    )

    session = client.sent[0]["session"]
    assert session["ref_audio"] == "data:audio/wav;base64,AAAA"


def test_realtime_event_collector_partitions_audio_by_response():
    collector = RealtimeEventCollector()
    collector.add({"type": "response.created", "response": {"id": "resp-a"}})
    collector.add(
        {
            "type": "response.audio.delta",
            "response_id": "resp-a",
            "delta": base64.b64encode(b"audio-a").decode("ascii"),
            "sample_rate_hz": 16_000,
        }
    )

    assert collector.response_ids == ["resp-a"]
    assert collector.audio_bytes("resp-a") == b"audio-a"
    assert collector.output_sample_rate_hz == 16_000
    assert collector.first_received_at("response.created") is not None
    assert collector.last_received_at("response.audio.delta") is not None


def test_realtime_event_collector_reports_engine_token_and_audio_intervals():
    collector = RealtimeEventCollector()
    collector.add(
        {"type": "response.created", "response": {"id": "resp-a"}},
        received_at_s=10.0,
    )
    stage_metrics = {
        "0": {
            "num_tokens_out": 4,
            "vllm_ttft_ms": 120.0,
            "vllm_tpot_ms": 15.0,
            "vllm_itl_ms": 14.0,
            "vllm_itls_ms": [10.0, 14.0, 18.0],
        }
    }
    for received_at_s, cumulative_audio_ms in ((10.2, 80), (10.25, 160), (10.36, 240)):
        collector.add(
            {
                "type": "response.audio.delta",
                "response_id": "resp-a",
                "delta": base64.b64encode(b"audio").decode("ascii"),
                "sample_rate_hz": 16_000,
                "metadata": {
                    "audio_duration_ms": cumulative_audio_ms,
                    "vllm_omni": {"stage_metrics": stage_metrics},
                },
            },
            received_at_s=received_at_s,
        )

    timing = collector.timing_summary(
        after_s=10.0,
        input_committed_at_s=9.9,
        response_id="resp-a",
    )

    assert timing["stage0_tokens"] == {
        "source": "engine_stage_metrics",
        "output_token_count": 4,
        "ttft_ms": 120.0,
        "tpot_ms": 15.0,
        "inter_token_interval_ms": {
            "count": 3,
            "mean": 14.0,
            "p50": 14.0,
            "p95": 18.0,
            "max": 18.0,
        },
    }
    assert timing["audio_output"] == {
        "source": "client_monotonic_receive",
        "chunk_count": 3,
        "response_created_to_first_audio_ms": 200.0,
        "commit_to_first_audio_ms": 300.0,
        "inter_chunk_interval_ms": {
            "count": 2,
            "mean": 80.0,
            "p50": 50.0,
            "p95": 110.0,
            "max": 110.0,
        },
        "chunk_duration_ms": {
            "count": 3,
            "mean": 80.0,
            "p50": 80.0,
            "p95": 80.0,
            "max": 80.0,
        },
        "max_chunk_gap_ms": 110.0,
    }


def test_response_timing_ignores_unowned_session_level_metrics():
    collector = RealtimeEventCollector()
    collector.add(
        {"type": "response.created", "response": {"id": "resp-a"}},
        received_at_s=10.0,
    )
    collector.add(
        {
            "type": "response.audio.delta",
            "response_id": "resp-a",
            "delta": base64.b64encode(b"audio").decode("ascii"),
            "metadata": {
                "vllm_omni": {
                    "stage_metrics": {
                        "0": {
                            "num_tokens_out": 20,
                            "vllm_ttft_ms": 157.0,
                            "vllm_tpot_ms": 16.0,
                            "vllm_itls_ms": [15.0, 17.0],
                        }
                    }
                }
            },
        },
        received_at_s=10.2,
    )
    collector.add(
        {
            "type": "response.listen",
            "metadata": {
                "vllm_omni": {
                    "stage_metrics": {
                        "0": {
                            "num_tokens_out": 2,
                            "vllm_ttft_ms": 106.0,
                            "vllm_tpot_ms": 0.0,
                            "vllm_itls_ms": [],
                        }
                    }
                }
            },
        },
        received_at_s=10.3,
    )

    timing = collector.timing_summary(after_s=10.0, response_id="resp-a")

    assert timing["stage0_tokens"]["output_token_count"] == 20
    assert timing["stage0_tokens"]["ttft_ms"] == 157.0


def test_realtime_client_pcm16_wav_round_trip(tmp_path):
    path = tmp_path / "audio.wav"
    pcm16 = b"\x01\x00\x02\x00"

    write_pcm16_wav(path, pcm16, sample_rate_hz=16_000)

    with wave.open(str(path), "rb") as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getframerate() == 16_000
    assert read_pcm16_wav(path) == pcm16
