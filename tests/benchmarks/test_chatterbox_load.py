# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Exercise the load harness through a real local streaming HTTP server."""

import asyncio
import json
import sys
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
from aiohttp import ClientSession, ClientTimeout, web

from vllm_omni.entrypoints.openai.speech_text_splitter import SpeechTextSplitter

sys.path.insert(0, str(Path(__file__).parents[2] / "benchmarks" / "tts"))
import chatterbox_load

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("metrics_available", [False, True])
def test_streaming_waves_measure_overlap_and_keep_failures(tmp_path, metrics_available):
    async def exercise():
        active = 0
        peak = 0

        async def speech(request):
            nonlocal active, peak
            body = await request.json()
            active += 1
            peak = max(peak, active)
            response = web.StreamResponse(headers={"X-Audio-Sample-Rate": "24000", "X-Audio-Channels": "1"})
            await response.prepare(request)
            await asyncio.sleep(0.01)
            # One silent stream per wave proves HTTP 200 alone cannot pass.
            value = body["seed"] + 1 if body["seed"] % 4 != 3 else 0
            await response.write(np.full(240, value, dtype="<i2").tobytes())
            await asyncio.sleep(0.3)
            await response.write(np.full(240, value, dtype="<i2").tobytes())
            await response.write_eof()
            active -= 1
            return response

        async def metrics(request):
            if not metrics_available:
                raise web.HTTPNotFound()
            text = "\n".join(
                f'vllm:num_requests_{kind}{{model_name="test",replica="0",stage="{stage}"}} {value}'
                for stage in (0, 1)
                for kind, value in (("running", active), ("waiting", 0))
            )
            return web.Response(text=text + "\n")

        app = web.Application()
        app.router.add_get("/metrics", metrics)
        app.router.add_post("/v1/audio/speech", speech)
        runner = web.AppRunner(app)
        await runner.setup()
        server = await asyncio.get_running_loop().create_server(runner.server, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        args = SimpleNamespace(
            model="test",
            url=f"http://127.0.0.1:{port}/v1/audio/speech",
            concurrency=4,
            stage0_sequences_per_request=2,
            cycles=2,
            max_new_tokens=128,
            reference=tmp_path,
            timeout=5,
            save_audio=True,
        )
        try:
            report = await chatterbox_load.run(args, [(f"data:audio/wav;base64,{i}", str(i)) for i in range(8)])
        finally:
            server.close()
            await server.wait_closed()
            await runner.cleanup()
        assert peak == 4
        assert report["stage0_capacity_target"] == 8
        assert report["stage0_sequences_per_request"] == 2
        assert not report["passed"]
        for workload in report["workloads"]:
            assert workload["peak_outstanding_clients"] == 4
            assert workload["errors"] == 2
            assert workload["completed"] == 6
            assert workload["ttfa_seconds"]["max"] > 0
            assert workload["e2e_seconds"]["p50"] > workload["ttfa_seconds"]["p50"]
            assert workload["rtf"]["p50"] > 10
            assert workload["playback_deficit_seconds"]["p50"] > 0.2
            for row in workload["requests"]:
                if row["ok"]:
                    assert row["rtf"] == pytest.approx(row["e2e_seconds"] / row["audio_seconds"])
                    assert sum(chunk["pcm_bytes"] for chunk in row["chunk_arrivals"]) == 960
            assert len(workload["requests"]) == 8
            observed = workload["server_metrics"]
            assert observed["available"] == metrics_available
            if metrics_available:
                assert observed["samples"] >= 2
                assert observed["peaks"]["stage_0_running"] > 0
                assert not workload["capacity_confirmed"]
                assert observed["peaks"]["stage_1_waiting"] == 0
            else:
                assert observed["samples"] == 0
                assert observed["errors"] == ["Metrics HTTP 404", "Metrics HTTP 404"]
                assert not workload["capacity_confirmed"]
        assert not report["capacity_confirmed"]
        for workload in report["workloads"]:
            assert sum("_audio_pcm" in row for row in workload["requests"]) == 2
        chatterbox_load.save_representative_audio(report, tmp_path / "report.json")
        json.dumps(report, allow_nan=False)
        for workload in report["workloads"]:
            selected = [row for row in workload["requests"] if "audio_path" in row]
            assert len(selected) == 2
            for row in selected:
                assert row["seed"] % 4 == 0
                with wave.open(row["audio_path"], "rb") as audio:
                    assert audio.getframerate() == 24000
                    assert audio.getnchannels() == 1
                    assert audio.getnframes() == 480
        cold, warm, builtin = report["workloads"]
        assert len({r["reference_sha256"] for r in cold["requests"]}) == 8
        assert {r["reference_sha256"] for r in warm["requests"]} == {"0"}
        assert {r["reference_sha256"] for r in builtin["requests"]} == {None}

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "data,rate,channels",
    [(b"", 24000, 1), (b"x", 24000, 1), (b"\x00\x00", 24000, 1), (b"\x01\x00", 22050, 1), (b"\x01\x00", 24000, 2)],
)
def test_invalid_pcm_is_rejected(data, rate, channels):
    with pytest.raises(ValueError):
        chatterbox_load.validate_pcm(data, rate, channels)


def test_scheduler_metric_parser_sums_replicas_and_keeps_stages_separate():
    values = chatterbox_load.parse_scheduler_metrics(
        'vllm:num_requests_running{stage="0",replica="0"} 40\n'
        'vllm:num_requests_running{stage="0",replica="1"} 24\n'
        'vllm:num_requests_waiting{stage="0"} 7\n'
        'vllm:num_requests_running{stage="1"} 8\n'
        'vllm:num_requests_waiting{stage="1"} 56\n'
        "vllm_omni:num_requests_running 64\n"
        "unrelated_gauge 1000\n"
    )
    assert values == {
        "stage_0_running": 64,
        "stage_0_waiting": 7,
        "stage_1_running": 8,
        "stage_1_waiting": 56,
        "omni_running": 64,
    }


@pytest.mark.parametrize("extra_samples", [-1, 0, 1])
def test_reference_duration_strict_five_second_boundary(tmp_path, extra_samples):
    rate = 24000
    path = tmp_path / "speech.wav"
    samples = np.sin(np.arange(rate * 5 + extra_samples, dtype=np.float32) * (2 * np.pi * 200 / rate))
    sf.write(path, samples, rate, subtype="FLOAT")
    if extra_samples <= 0:
        with pytest.raises(ValueError, match="more than five seconds"):
            chatterbox_load.prepare_references(path, 1)
    else:
        references = chatterbox_load.prepare_references(path, 1)
        assert len(references) == 1
        assert references[0][0].startswith("data:audio/wav;base64,")


def test_sentence_websocket_load_through_actual_server(tmp_path):
    async def exercise():
        active = 0
        peak = 0

        async def speech(request):
            nonlocal active, peak
            websocket = web.WebSocketResponse()
            await websocket.prepare(request)
            config = await websocket.receive_json()
            assert config["split_granularity"] == "sentence"
            assert config["stream_audio"]
            active += 1
            peak = max(peak, active)
            sentence = 0
            splitter = SpeechTextSplitter("sentence")
            try:
                async for message in websocket:
                    event = json.loads(message.data)
                    if event["type"] == "input.text":
                        units = splitter.feed(event["text"])
                    elif event["type"] == "input.done":
                        units = splitter.flush()
                    elif event["type"] == "session.close":
                        break
                    else:
                        continue
                    for unit in units:
                        await asyncio.sleep(0.02)
                        await websocket.send_json(
                            {
                                "type": "audio.start",
                                "sentence_index": sentence,
                                "sentence_text": unit,
                                "sample_rate": 24000,
                                "channels": 1,
                            }
                        )
                        pcm = np.full(240, config["seed"] + sentence + 1, dtype="<i2").tobytes()
                        await websocket.send_bytes(pcm)
                        await websocket.send_json({"type": "audio.done", "sentence_index": sentence})
                        sentence += 1
                    if event["type"] == "input.done":
                        await websocket.send_json({"type": "session.done", "total_sentences": sentence})
            finally:
                active -= 1
                await websocket.close()
            return websocket

        app = web.Application()
        app.router.add_get("/v1/audio/speech/stream", speech)
        runner = web.AppRunner(app)
        await runner.setup()
        server = await asyncio.get_running_loop().create_server(runner.server, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        args = SimpleNamespace(
            model="test",
            url=f"http://127.0.0.1:{port}/v1/audio/speech/stream",
            transport="websocket",
            concurrency=4,
            stage0_sequences_per_request=1,
            cycles=1,
            max_new_tokens=128,
            reference=tmp_path,
            timeout=5,
            save_audio=True,
        )
        try:
            report = await chatterbox_load.run(args, [(f"data:audio/wav;base64,{i}", str(i)) for i in range(4)])
        finally:
            server.close()
            await server.wait_closed()
            await runner.cleanup()
        assert report["functional_passed"]
        assert not report["capacity_confirmed"]
        assert peak == 4
        for workload in report["workloads"]:
            assert workload["completed"] == 4
            assert workload["segments"]["completed"] == 20
            for row in workload["requests"]:
                assert len(row["segments"]) == 5
                assert row["segments"][1]["ttfa_seconds"] >= row["segments"][0]["ttfa_seconds"]
        chatterbox_load.save_representative_audio(report, tmp_path / "websocket.json")
        assert len(list(tmp_path.glob("websocket-*.wav"))) == 3
        json.dumps(report, allow_nan=False)

    asyncio.run(exercise())


def test_websocket_failure_preserves_server_close_reason():
    async def exercise():
        async def speech(request):
            websocket = web.WebSocketResponse()
            await websocket.prepare(request)
            await websocket.receive_json()
            await asyncio.sleep(0.02)
            await websocket.close(code=1011, message=b"keepalive ping timeout")
            return websocket

        app = web.Application()
        app.router.add_get("/speech", speech)
        runner = web.AppRunner(app)
        await runner.setup()
        server = await asyncio.get_running_loop().create_server(runner.server, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        gate = asyncio.Event()
        gate.set()
        state = {"active": 0, "peak": 0}
        payload = {
            "model": "test",
            "voice": "default",
            "input": "First sentence. Second sentence.",
            "seed": 0,
            "max_new_tokens": 128,
        }
        try:
            async with ClientSession(timeout=ClientTimeout(total=5)) as session:
                result = await chatterbox_load.websocket_request(
                    session, f"http://127.0.0.1:{port}/speech", payload, gate, state, None
                )
        finally:
            server.close()
            await server.wait_closed()
            await runner.cleanup()
        assert not result["ok"]
        assert result["websocket_message_type"] == "CLOSE"
        assert result["websocket_close_code"] == 1011
        assert result["websocket_close_reason"] == "keepalive ping timeout"
        assert "1011" in result["error"]
        assert "keepalive ping timeout" in result["error"]
        assert state["active"] == 0

    asyncio.run(exercise())
