# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CPU contract tests for the SGLang native duplex replay adapter."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping

import pybase64 as base64
import pytest

from vllm_omni.benchmarks.duplex import omni_duplex_eval_runner as runner
from vllm_omni.benchmarks.duplex.sglang_replay_session import (
    SGLangReplaySession,
    build_sglang_realtime_url,
)
from vllm_omni.benchmarks.duplex_session_inputs import FRAME_TRANSPORT_SGLANG_IMAGE
from vllm_omni.benchmarks.duplex_session_metrics import collect_duplex_session_metrics
from vllm_omni.experimental.fullduplex.client import RealtimeEventCollector

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.benchmark]


class _FakeRealtimeClient:
    def __init__(
        self,
        *,
        finish_mode: str = "normal",
        supported_sampling: list[str] | None = None,
        uncorrelated_update_error: bool = False,
    ) -> None:
        self.events = RealtimeEventCollector()
        self.sent: list[dict[str, object]] = []
        self.media_clock_ms: list[int] = []
        self._received_at_s = 0.0
        self.finish_mode = finish_mode
        self.supported_sampling = supported_sampling or ["temperature", "force_listen_count"]
        self.uncorrelated_update_error = uncorrelated_update_error
        self.events.add(
            {"type": "session.created", "session": {"id": "session-1"}},
            received_at_s=self._next_time(),
        )

    def _next_time(self, increment: float = 0.01) -> float:
        self._received_at_s += increment
        return self._received_at_s

    def _add(self, event: dict[str, object], *, increment: float = 0.01) -> None:
        self.events.add(event, received_at_s=self._next_time(increment))

    async def send(self, event: dict[str, object]) -> None:
        self.sent.append(event)
        event_type = event.get("type")
        event_id = event.get("event_id")
        if event_type == "input_audio_buffer.append":
            self.events.note_client_append({"sent_at_s": self._received_at_s, "event_id": event_id})
        elif event_type == "session.update":
            requested_session = event["session"]
            assert isinstance(requested_session, Mapping)
            requested_extension = requested_session.get("sglang", {})
            assert isinstance(requested_extension, Mapping)
            if self.uncorrelated_update_error:
                self._add(
                    {
                        "type": "error",
                        "event_id": "unrelated-event",
                        "error": {"message": "unrelated asynchronous failure"},
                    }
                )
            self._add(
                {
                    "type": "session.updated",
                    "client_event_id": event_id,
                    "session": {
                        "audio": {"output": {"format": {"type": "audio/pcm", "rate": 16000}}},
                        "sglang": {
                            **requested_extension,
                            "granted": {
                                "native_full_duplex": True,
                                "client_commit": False,
                                "tail_policy": "pad",
                                "input_modalities": ["audio", "image"],
                                "output_modalities": ["audio"],
                                "native_unit_ms": 1000,
                                "input_audio_format": {"type": "audio/pcm", "rate": 16000},
                                "output_audio_format": {"type": "audio/pcm", "rate": 16000},
                                "sampling_parameters": self.supported_sampling,
                            },
                        },
                    },
                }
            )
        elif event_type == "sglang.input_audio.end":
            if self.finish_mode == "no_response":
                self._add({"type": "sglang.input_audio.drained", "client_event_id": event_id})
                return
            response_id = "response-1"
            audio = base64.b64encode(bytes(3200)).decode("ascii")
            origin = {
                "ttft": "model-turn request start to first non-empty text output",
                "ttfp": "model-turn request start to first non-empty audio output",
            }
            self._add({"type": "sglang.input_audio.ended", "client_event_id": event_id})
            self._add({"type": "response.created", "response": {"id": response_id}})
            self._add(
                {
                    "type": "response.output_audio_transcript.delta",
                    "response_id": response_id,
                    "delta": "hello",
                    "response_request_metrics": {
                        "source": "server_monotonic_request_start",
                        "measurement_origin": origin,
                        "ttft_ms": 12.5,
                    },
                }
            )
            self._add(
                {
                    "type": "response.output_audio.delta",
                    "response_id": response_id,
                    "delta": audio,
                    "response_request_metrics": {
                        "source": "server_monotonic_request_start",
                        "measurement_origin": origin,
                        "ttft_ms": 12.5,
                        "ttfp_ms": 34.5,
                    },
                }
            )
            self._add(
                {
                    "type": "response.output_audio.delta",
                    "response_id": response_id,
                    "delta": audio,
                },
                increment=0.25,
            )
            done: dict[str, object] = {
                "type": "response.done",
                "response": {"id": response_id, "status": "completed"},
                "response_request_metrics": {
                    "source": "server_monotonic_request_start",
                    "measurement_origin": origin,
                    "ttft_ms": 12.5,
                    "ttfp_ms": 34.5,
                },
            }
            if self.finish_mode == "normal":
                self._add(done)
                self._add({"type": "sglang.input_audio.drained", "client_event_id": event_id})
            else:
                self._add({"type": "sglang.input_audio.drained", "client_event_id": event_id})
                if self.finish_mode == "drained_before_done":
                    asyncio.get_running_loop().call_later(0.01, lambda: self._add(done))
        elif event_type == "session.close":
            self._add({"type": "session.closed", "client_event_id": event_id})

    def advance_media_clock(self, media_clock_ms: int) -> None:
        self.media_clock_ms.append(media_clock_ms)

    def raise_if_reader_stopped(self) -> None:
        return None


def _session_payload(client: _FakeRealtimeClient) -> Mapping[str, object]:
    event = next(event for event in client.sent if event["type"] == "session.update")
    session = event["session"]
    assert isinstance(session, Mapping)
    return session


def test_build_sglang_realtime_url_preserves_endpoint_and_query():
    assert (
        build_sglang_realtime_url("https://example.test/v1/realtime?deployment=one")
        == "wss://example.test/v1/realtime?deployment=one"
    )


@pytest.mark.asyncio
async def test_sglang_session_replays_media_and_uses_shared_server_metrics():
    client = _FakeRealtimeClient()
    session = SGLangReplaySession(client)
    reference = "data:audio/wav;base64," + base64.b64encode(b"wav").decode("ascii")
    await session.configure(
        "served-model",
        ref_audio=reference,
        instructions="answer briefly",
        extra_body={"temperature": 0.2},
        turn_detection="none",
    )

    payload = _session_payload(client)
    extension = payload["sglang"]
    assert isinstance(extension, Mapping)
    assert extension["sampling"] == {"temperature": 0.2, "force_listen_count": 0}
    assert extension["reference_audio"] == {"media_type": "audio/wav", "data": "d2F2"}
    assert payload["instructions"] == "answer briefly"
    assert session.frame_transport == FRAME_TRANSPORT_SGLANG_IMAGE
    assert client.events.output_sample_rate_hz == 16000
    assert session.session_configuration == {
        "backend": "sglang",
        "requested_sampling": {"temperature": 0.2, "force_listen_count": 0},
        "effective_sampling": {"temperature": 0.2, "force_listen_count": 0},
        "sampling_source": "server_echo",
        "supported_sampling_parameters": ["temperature", "force_listen_count"],
    }

    pcm = bytes(2 * 32000)
    await session.stream_input(
        pcm,
        ((0.25, b"image-0"), (1.25, b"image-1")),
        sample_id="sample-1",
        unit_ms=1000,
        realtime=False,
    )
    media_events = [
        event for event in client.sent if event["type"] in {"sglang.input_image.append", "input_audio_buffer.append"}
    ]
    assert [event["type"] for event in media_events] == [
        "sglang.input_image.append",
        "input_audio_buffer.append",
        "sglang.input_image.append",
        "input_audio_buffer.append",
    ]
    appends = [event for event in media_events if event["type"] == "input_audio_buffer.append"]
    assert [event["sglang"] for event in appends] == [
        {"seq": 0, "t_start_ms": 0},
        {"seq": 1, "t_start_ms": 1000.0},
    ]
    images = [event for event in media_events if event["type"] == "sglang.input_image.append"]
    assert [event["sglang"] for event in images] == [{"t_ms": 250.0}, {"t_ms": 1250.0}]
    assert client.media_clock_ms == [1000, 2000]

    drain = await session.finish_input(timeout_s=1.0)
    assert drain.response_done and drain.input_committed and drain.drain_timeout is None
    collector = runner._event_collector_from_realtime(client.events)
    metrics = collect_duplex_session_metrics(collector, stream_start=0.0, session_id="sample-1")
    request = metrics.request_metrics[0]
    assert request["ttft_ms"] == 12.5
    assert request["ttfp_ms"] == 34.5
    assert request["measurement_origin"]["ttft"].startswith("model-turn request start")
    assert request["audio_continuity"]["audio_duration_s"] == pytest.approx(0.2)
    assert request["audio_continuity"]["audio_underrun_s"] == pytest.approx(0.15)
    assert "stage0_tokens" not in request

    await session.close(timeout_s=1.0)
    assert client.sent[-1]["type"] == "session.close"


@pytest.mark.asyncio
async def test_sglang_session_rejects_server_vad():
    session = SGLangReplaySession(_FakeRealtimeClient())
    with pytest.raises(ValueError, match="turn_detection='none'"):
        await session.configure(
            "served-model",
            ref_audio=None,
            instructions=None,
            extra_body=None,
            turn_detection="server_vad",
        )


@pytest.mark.asyncio
async def test_sglang_session_preserves_explicit_force_listen_count():
    client = _FakeRealtimeClient()
    session = SGLangReplaySession(client)

    await session.configure(
        "served-model",
        ref_audio=None,
        instructions=None,
        extra_body={"sampling": {"force_listen_count": 2}},
        turn_detection="none",
    )

    extension = _session_payload(client)["sglang"]
    assert isinstance(extension, Mapping)
    assert extension["sampling"] == {"force_listen_count": 2}
    assert session.session_configuration["requested_sampling"] == {"force_listen_count": 2}


@pytest.mark.asyncio
async def test_sglang_session_rejects_unknown_top_level_extra_body():
    client = _FakeRealtimeClient()
    session = SGLangReplaySession(client)

    with pytest.raises(ValueError, match="Unsupported top-level extra_body keys.*mystery_option"):
        await session.configure(
            "served-model",
            ref_audio=None,
            instructions=None,
            extra_body={"mystery_option": True},
            turn_detection="none",
        )
    assert not any(event["type"] == "session.update" for event in client.sent)


@pytest.mark.asyncio
async def test_sglang_session_rejects_sampling_not_granted_by_server():
    session = SGLangReplaySession(_FakeRealtimeClient(supported_sampling=["force_listen_count"]))

    with pytest.raises(ValueError, match="does not support requested sampling parameters: top_p"):
        await session.configure(
            "served-model",
            ref_audio=None,
            instructions=None,
            extra_body={"top_p": 0.8},
            turn_detection="none",
        )


@pytest.mark.asyncio
async def test_sglang_command_wait_ignores_uncorrelated_error():
    client = _FakeRealtimeClient(uncorrelated_update_error=True)
    session = SGLangReplaySession(client)

    await session.configure(
        "served-model",
        ref_audio=None,
        instructions=None,
        extra_body=None,
        turn_detection="none",
    )

    assert session.session_configuration["backend"] == "sglang"
    assert client.events.count("error") == 1


@pytest.mark.asyncio
async def test_sglang_finish_waits_for_response_done_after_drained():
    client = _FakeRealtimeClient(finish_mode="drained_before_done")
    session = SGLangReplaySession(client)
    await session.configure(
        "served-model",
        ref_audio=None,
        instructions=None,
        extra_body=None,
        turn_detection="none",
    )

    drain = await session.finish_input(timeout_s=1.0)

    assert drain.input_committed and drain.response_done
    assert drain.drain_timeout is None
    assert client.events.count("response.done") == 1


@pytest.mark.asyncio
async def test_sglang_finish_reports_unfinished_response_after_drained():
    client = _FakeRealtimeClient(finish_mode="drained_without_done")
    session = SGLangReplaySession(client)
    await session.configure(
        "served-model",
        ref_audio=None,
        instructions=None,
        extra_body=None,
        turn_detection="none",
    )

    drain = await session.finish_input(timeout_s=0.05)

    assert drain.input_committed and not drain.response_done
    assert drain.drain_timeout is not None
    assert drain.drain_timeout_state == {
        "response_ids": ["response-1"],
        "unfinished_response_ids": ["response-1"],
    }


@pytest.mark.asyncio
async def test_sglang_finish_times_out_when_no_response_was_created():
    client = _FakeRealtimeClient(finish_mode="no_response")
    session = SGLangReplaySession(client)
    await session.configure(
        "served-model",
        ref_audio=None,
        instructions=None,
        extra_body=None,
        turn_detection="none",
    )

    drain = await session.finish_input(timeout_s=0.05)

    assert drain.input_committed and not drain.response_done
    assert drain.drain_timeout is not None
    assert drain.drain_timeout_state == {"response_ids": [], "unfinished_response_ids": []}
