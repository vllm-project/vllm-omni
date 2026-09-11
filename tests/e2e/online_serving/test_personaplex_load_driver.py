# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import asyncio
import base64
import json
import wave
from contextlib import asynccontextmanager
from typing import TypedDict

import numpy as np
import pytest
import websockets

from tests.e2e.online_serving import personaplex_realtime_duplex as driver

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _ServerState(TypedDict):
    index: int
    closed: bool
    frames: int
    close_requested: bool


def _args(tmp_path, *extra):
    wav_path = tmp_path / "input.wav"
    with wave.open(str(wav_path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(driver.SAMPLE_RATE_HZ)
        stream.writeframes(np.full(driver.FRAME_SAMPLES * 2, 2000, dtype="<i2").tobytes())
    return driver.parse_args(
        [
            "--model",
            "fixture-only",
            "--input-wav",
            str(wav_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--sessions",
            "2",
            "--load-frames",
            "2",
            "--tail-s",
            "0",
            "--drain-s",
            "0.04",
            "--timeout-s",
            "0.5",
            "--min-voiced-frames",
            "1",
            "--minimum-audio-chunks",
            "1",
            "--max-frame-deficit",
            "0",
            *extra,
        ]
    )


def _audio(response_id="r", samples=1920):
    return {
        "type": "response.output_audio.delta",
        "response_id": response_id,
        "delta": base64.b64encode(np.full(samples, 2000, dtype="<i2").tobytes()).decode("ascii"),
        "sample_rate_hz": driver.SAMPLE_RATE_HZ,
        "metadata": {
            "vllm_omni": {
                "runtime_impl": "scheduler_data_plane",
                "uses_model_runner_scheduler": True,
                "runner_kv_backed": True,
            }
        },
    }


@asynccontextmanager
async def _server(mode="success"):
    states: list[_ServerState] = []

    async def handle(ws):
        index = len(states)
        state: _ServerState = {"index": index, "closed": False, "frames": 0, "close_requested": False}
        states.append(state)
        response_id = "shared" if mode == "shared_response" else f"r-{index}"
        try:
            async for raw in ws:
                message = json.loads(raw)
                if message["type"] == "session.update":
                    if mode == "reject_second" and index == 1:
                        await ws.send(json.dumps({"type": "error", "error": {"code": "resource_exhausted"}}))
                        continue
                    if mode == "missing_capabilities":
                        await ws.send(
                            json.dumps(
                                {
                                    "type": "session.created",
                                    "session": {"resume_token": "must-not-be-exported"},
                                }
                            )
                        )
                        continue
                    await ws.send(
                        json.dumps(
                            {
                                "type": "session.created",
                                "session": {
                                    "resume_token": "must-not-be-exported",
                                    "capabilities": {
                                        "chunk_period_ms": 80,
                                        "supports_multi_session_same_replica": True,
                                    },
                                },
                            }
                        )
                    )
                    await ws.send(json.dumps({"type": "response.created", "response": {"id": response_id}}))
                elif message["type"] == "input_audio_buffer.append":
                    state["frames"] += 1
                    if mode == "disconnect":
                        await ws.close()
                        return
                    if mode == "bad_json":
                        await ws.send("{")
                    elif mode not in ("silent", "late_on_close"):
                        await ws.send(json.dumps(_audio(response_id)))
                elif message["type"] == "session.close":
                    state["close_requested"] = True
                    if mode == "late_on_close":
                        await ws.send(json.dumps(_audio(response_id, samples=driver.FRAME_SAMPLES * state["frames"])))
                    if mode == "error_on_close":
                        await ws.send(json.dumps({"type": "error", "error": {"code": "cleanup_failed"}}))
                    if mode != "no_close_ack":
                        await ws.send(json.dumps({"type": "session.closed"}))
        finally:
            state["closed"] = True

    async with websockets.serve(handle, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        yield f"ws://127.0.0.1:{port}/v1/realtime", states


def test_omitted_sessions_preserves_lifecycle_mode():
    args = driver.parse_args(["--model", "unused", "--input-wav", "unused.wav"])
    assert args.sessions is None


@pytest.mark.parametrize(
    "options",
    [
        ["--sessions", "0"],
        ["--sessions", "65"],
        ["--load-frames", "0"],
        ["--load-frames", "10001"],
        ["--drain-s", "nan"],
        ["--tail-s", "-1"],
        ["--timeout-s", "0"],
        ["--timeout-s", "inf"],
        ["--max-client-rtf", "-1"],
        ["--max-frame-deficit", "-1"],
        ["--minimum-audio-chunks", "0"],
        ["--min-voiced-frames", "0"],
        ["--voiced-frame-rms-threshold", "nan"],
    ],
)
def test_load_rejects_invalid_configuration(tmp_path, options):
    with pytest.raises(SystemExit):
        _args(tmp_path, *options)


def test_metric_clock_and_packet_frame_distinction():
    client = driver.RawRealtimeProbe("ws://unused")
    client.events.add({"type": "session.created", "session": {"resume_token": "secret"}}, received_at_s=7)
    for when in (10.4, 10.8, 11.6):
        event = _audio(samples=driver.FRAME_SAMPLES * 5)
        event["_client_received_at_s"] = -999.0
        client.events.add(event, received_at_s=when)
    report = driver._load_metrics(client, [(10, 10, 10.01)])
    intervals = report["client_audio_packet_interval_ms"]
    assert isinstance(intervals, dict)
    assert intervals["count"] == 2
    assert intervals["median"] == pytest.approx(600)
    assert intervals["p99"] == pytest.approx(796)
    assert intervals["max"] == pytest.approx(800)
    assert intervals["argmax_index"] == 1
    assert report["client_first_audio_after_stream_start_ms"] == pytest.approx(400)
    assert report["output_samples"] == 15 * driver.FRAME_SAMPLES
    assert "secret" not in json.dumps(report)
    assert "-999" not in json.dumps(report)


def test_empty_output_has_missing_not_zero_latency():
    report = driver._load_metrics(driver.RawRealtimeProbe("ws://unused"), [])
    assert report["client_stream_rtf"] is None
    intervals = report["client_audio_packet_interval_ms"]
    assert isinstance(intervals, dict)
    assert intervals["p99"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize("sessions", [1, 2, 4])
async def test_real_websocket_load_success(tmp_path, sessions):
    args = _args(tmp_path, "--sessions", str(sessions))
    async with _server() as (url, states):
        args.url = url
        result = await driver.run(args)
    assert result["ok"] is True
    assert result["passed_sessions"] == sessions
    rows = result["sessions"]
    assert isinstance(rows, list)
    assert len(rows) == sessions
    assert all(row["frame_deficit"] == 0 for row in rows)
    assert all(s["closed"] and s["close_requested"] for s in states)
    assert "must-not-be-exported" not in json.dumps(result)
    saved = json.loads((tmp_path / "out/load-result.json").read_text())
    assert saved == result
    origins = [row["input_timeline"][0]["planned_at_s"] for row in rows]
    assert len(set(origins)) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode",
    [
        "reject_second",
        "silent",
        "disconnect",
        "no_close_ack",
        "shared_response",
        "late_on_close",
        "error_on_close",
        "bad_json",
        "missing_capabilities",
    ],
)
async def test_real_websocket_failures_remain_in_results(tmp_path, mode):
    args = _args(tmp_path)
    async with _server(mode) as (url, states):
        args.url = url
        result = await asyncio.wait_for(driver.run(args), timeout=4)
    assert result["ok"] is False
    assert "must-not-be-exported" not in json.dumps(result)
    assert result["requested_sessions"] == 2
    rows = result["sessions"]
    assert isinstance(rows, list)
    assert len(rows) == 2
    assert all(s["closed"] for s in states)
    if mode == "reject_second":
        assert result["passed_sessions"] == 1
        rejected = next(row for row in rows if not row["ok"])
        assert rejected["input_frames"] == 0
        assert rejected["client_stream_rtf"] is None
        assert rejected["server_error_codes"] == ["resource_exhausted"]
    if mode == "silent":
        # A failed measurement must still release its admitted session.
        assert all(s["close_requested"] for s in states)
    if mode == "late_on_close":
        assert all(row["output_samples"] == 0 for row in rows)
    if mode == "shared_response":
        assert result["passed_sessions"] == 0


@pytest.mark.asyncio
async def test_existing_report_is_not_overwritten(tmp_path):
    args = _args(tmp_path)
    path = tmp_path / "out/load-result.json"
    path.parent.mkdir()
    path.write_text("previous measurement")
    with pytest.raises(FileExistsError):
        await driver.run(args)
    assert path.read_text() == "previous measurement"


@pytest.mark.asyncio
async def test_cancellation_closes_all_connections(tmp_path):
    args = _args(tmp_path, "--drain-s", "10")
    async with _server() as (url, states):
        args.url = url
        task = asyncio.create_task(driver.run(args))

        async def started():
            while len(states) < 2 or not all(state["frames"] for state in states):
                await asyncio.sleep(0.01)

        await asyncio.wait_for(started(), timeout=3)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=3)
    assert all(s["closed"] for s in states)


@pytest.mark.asyncio
async def test_client_rtf_ceiling_marks_slow_sessions_failed(tmp_path):
    args = _args(tmp_path, "--max-client-rtf", "0.001")
    async with _server() as (url, _):
        args.url = url
        result = await driver.run(args)
    assert result["passed_sessions"] == 0
    rows = result["sessions"]
    assert isinstance(rows, list)
    assert all("client_stream_rtf exceeds" in row["error"] for row in rows)


@pytest.mark.asyncio
async def test_stalled_send_does_not_cause_catchup_burst(monkeypatch):
    class Clock:
        now = 10.0

        def monotonic(self):
            return self.now

        async def sleep(self, delay):
            self.now += delay

    class Client(driver.RawRealtimeProbe):
        async def send(self, event):
            if event["audio_end_ms"] == 80:
                clock.now += 0.25

    clock = Clock()
    monkeypatch.setattr(driver, "time", clock)
    monkeypatch.setattr(driver.asyncio, "sleep", clock.sleep)
    monkeypatch.setattr(driver, "_check_load_connection", lambda _: None)
    sends: list[tuple[float, float, float]] = []
    await driver._paced_load_frames(
        Client("ws://unused"),
        np.zeros(driver.FRAME_SAMPLES * 3, dtype="<f4"),
        epoch=10.0,
        timeout_s=1.0,
        sends=sends,
    )
    assert len(sends) == 3
    assert sends[1][1] >= sends[0][2] + driver.FRAME_PERIOD_S
    assert sends[1][1] - sends[1][0] == pytest.approx(0.25)
    assert sends[2][1] >= sends[1][2] + driver.FRAME_PERIOD_S


@pytest.mark.parametrize("ok", [True, False])
def test_load_main_exit_status(tmp_path, monkeypatch, ok):
    args = _args(tmp_path)

    async def completed(_args):
        return {"mode": "paced_multi_session", "ok": ok}

    monkeypatch.setattr(driver, "parse_args", lambda: args)
    monkeypatch.setattr(driver, "run", completed)
    if ok:
        driver.main()
    else:
        with pytest.raises(SystemExit) as exc:
            driver.main()
        assert exc.value.code == 1


@pytest.mark.asyncio
async def test_handshake_rejection_is_retained_without_server_details(tmp_path, monkeypatch):
    from websockets.exceptions import InvalidHandshake

    async def rejected(_self):
        raise InvalidHandshake("credentials-in-remote-handshake-detail")

    monkeypatch.setattr(driver.RawRealtimeProbe, "__aenter__", rejected)
    result = await asyncio.wait_for(driver.run(_args(tmp_path)), timeout=3)
    assert result["ok"] is False
    rows = result["sessions"]
    assert isinstance(rows, list) and len(rows) == 2
    assert all(row["error"] == "admission: InvalidHandshake" for row in rows)
    assert "credentials-in-remote-handshake-detail" not in json.dumps(result)
