# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Omni-DuplexEval: optional session.stats must not fail a finished sample."""

import asyncio
import time

import pytest

from vllm_omni.benchmarks.duplex import omni_duplex_eval_runner as runner
from vllm_omni.benchmarks.duplex.omni_duplex_eval_dataset import DuplexSample
from vllm_omni.benchmarks.duplex.omni_duplex_eval_runner import PreparedSample
from vllm_omni.benchmarks.duplex_session_inputs import TURN_DETECTION_SERVER_VAD
from vllm_omni.experimental.fullduplex.client import RealtimeDuplexClient, RealtimeEventCollector

pytestmark = [pytest.mark.core_model, pytest.mark.benchmark, pytest.mark.cpu]

_STATS_TIMEOUT_S = 0.05
_GRACE_S = 0.05
_BUDGET_S = 5.0


class _FakeTransport:
    def abort(self) -> None:
        return None


class _FakeWebSocket:
    def __init__(self) -> None:
        self.transport = _FakeTransport()

    async def close(self) -> None:
        return None


class _CompletedSessionClient:
    """A session that finishes a response and closes cleanly."""

    def __init__(self, *args, **kwargs) -> None:
        del args, kwargs
        self.events = RealtimeEventCollector()
        self._ws = _FakeWebSocket()
        self.configured = False
        self.close_calls = 0
        self.events.add(
            {
                "type": "session.created",
                "session": {"capabilities": {"supports_session_stats": True}},
            }
        )
        self.events.add({"type": "response.created", "response": {"id": "r1"}})
        self.events.add({"type": "response.done", "response": {"id": "r1", "status": "completed"}})

    async def __aenter__(self) -> "_CompletedSessionClient":
        return self

    async def __aexit__(self, *_exc) -> bool:
        await self._ws.close()
        return False

    async def configure(self, *args, **kwargs) -> None:
        del args, kwargs
        self.configured = True

    async def stream_av_units(self, *args, **kwargs) -> int:
        del args, kwargs
        return 0

    async def acknowledge_playback(self) -> None:
        return None

    async def close_session(self, *, timeout_s: float = 20.0) -> None:
        del timeout_s
        self.close_calls += 1

    def raise_if_reader_stopped(self) -> None:
        return None


class _StatsErrorClient(_CompletedSessionClient):
    async def get_session_stats(self, *, timeout_s: float = 10.0, event_id: str | None = None) -> dict[str, object]:
        del timeout_s
        request_id = event_id or "evt_stats"
        self.events.add(
            {
                "type": "error",
                "related_event_id": request_id,
                "error": {"code": "session_stats_disabled", "message": "disabled"},
            }
        )
        raise ConnectionError("disabled")


class _StalledStatsSendClient(_CompletedSessionClient):
    async def send(self, event: dict[str, object]) -> None:
        if event.get("type") == "session.stats.get":
            await asyncio.Event().wait()

    async def get_session_stats(self, *, timeout_s: float = 10.0, event_id: str | None = None) -> dict[str, object]:
        return await RealtimeDuplexClient.get_session_stats(self, timeout_s=timeout_s, event_id=event_id)


def _sample() -> DuplexSample:
    return DuplexSample(
        id="s1",
        split="dev",
        family="pr",
        task_type=None,
        video=None,
        question_audio=None,
        video_duration=1.0,
    )


def _install_deadlines(monkeypatch) -> None:
    monkeypatch.setattr(runner, "_FINAL_CLOSE_TIMEOUT_S", _STATS_TIMEOUT_S)
    monkeypatch.setattr(runner, "_FORCE_CLOSE_GRACE_S", _GRACE_S)
    monkeypatch.setattr(runner, "_FINAL_ACK_TIMEOUT_S", _STATS_TIMEOUT_S)
    monkeypatch.setattr(runner, "_SESSION_STATS_TIMEOUT_S", _STATS_TIMEOUT_S)
    monkeypatch.setattr(runner, "_SERVER_TURN_SETTLE_S", 0.0)


async def _generate(monkeypatch, tmp_path, client_cls):
    created: list[object] = []

    def client_factory(*args, **kwargs):
        client = client_cls(*args, **kwargs)
        created.append(client)
        return client

    monkeypatch.setattr(runner, "RealtimeDuplexClient", client_factory)
    prepared = PreparedSample(pcm=b"", frames=(), duration=1.0, ref_audio=None, ref_audio_sha256=None)
    started = time.monotonic()
    result = await runner.generate_sample(
        _sample(),
        url="ws://localhost:8000/v1/realtime",
        model="test-model",
        ref_audio=None,
        output_root=tmp_path,
        turn_detection=TURN_DETECTION_SERVER_VAD,
        prepared=prepared,
    )
    return result, created[0], time.monotonic() - started


async def test_stats_error_keeps_a_successful_sample(monkeypatch, tmp_path) -> None:
    _install_deadlines(monkeypatch)
    result, client, elapsed = await _generate(monkeypatch, tmp_path, _StatsErrorClient)

    assert elapsed < _BUDGET_S
    assert client.configured is True
    assert client.close_calls == 1
    assert result.metadata["response_done"] is True
    assert result.error == ""
    assert result.metadata["server_errors"] == []
    assert result.metadata["session_stats_errors"]
    assert all("busy_rtf" not in metric for metric in result.request_metrics)
    assert all("stall_ratio" not in metric for metric in result.request_metrics)
    assert result.output.exists()
    assert result.output.with_name("s1.meta.json").exists()


async def test_stalled_stats_send_still_writes_artifacts(monkeypatch, tmp_path) -> None:
    _install_deadlines(monkeypatch)
    result, client, elapsed = await _generate(monkeypatch, tmp_path, _StalledStatsSendClient)

    assert elapsed < _BUDGET_S
    assert client.configured is True
    assert client.close_calls == 1
    assert result.metadata["response_done"] is True
    assert result.error == ""
    assert all("busy_rtf" not in metric for metric in result.request_metrics)
    assert result.output.exists()
    assert result.output.with_name("s1.meta.json").exists()
