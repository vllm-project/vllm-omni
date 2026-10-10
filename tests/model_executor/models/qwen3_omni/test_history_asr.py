# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import asyncio
import io
import wave
from email import policy
from email.message import EmailMessage
from email.parser import BytesParser

import httpx
import numpy as np
import pytest

from vllm_omni.engine.duplex.session.history_calibration import HeardTextSnapshot
from vllm_omni.model_executor.models.qwen3_omni.duplex import history_asr

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def snapshot():
    return HeardTextSnapshot(
        "response-test", "Hello world again. The rest is unheard.", np.zeros(24000, "<f4").tobytes(), 24000, 1000
    )


def install_transport(monkeypatch, handler):
    client = httpx.AsyncClient
    monkeypatch.setattr(
        history_asr.httpx,
        "AsyncClient",
        lambda **kwargs: client(transport=httpx.MockTransport(handler), **kwargs),
    )


@pytest.mark.asyncio
async def test_asr_sends_only_heard_audio_and_logs_no_transcript(monkeypatch, caplog):
    async def handle(request):
        message = BytesParser(policy=policy.default).parsebytes(
            b"Content-Type: " + request.headers["content-type"].encode() + b"\r\n\r\n" + request.content
        )
        assert isinstance(message, EmailMessage)
        parts = {part.get_param("name", header="content-disposition"): part for part in message.iter_parts()}
        assert set(parts) == {"file", "model", "response_format"}
        payload = parts["file"].get_payload(decode=True)
        assert isinstance(payload, bytes)
        with wave.open(io.BytesIO(payload)) as wav:
            assert wav.getframerate() == 24000
            assert wav.getnframes() == 20160  # 1000 ms playback less the 160 ms guard.
            assert wav.getsampwidth() == 2
            assert wav.getnchannels() == 1
        return httpx.Response(200, json={"text": "Hello world again"})

    install_transport(monkeypatch, handle)
    calibrate = history_asr.QwenAsrCalibration("http://asr.test/transcriptions", "model-test", 1)
    with caplog.at_level("INFO", logger=history_asr.logger.name):
        assert await calibrate(snapshot()) == len("Hello world")
    log = "\n".join(record.getMessage() for record in caplog.records if record.name == history_asr.logger.name)
    assert "status=accepted" in log
    assert all(name + "_ms=" in log for name in ("preprocess", "queue", "http", "match", "total"))
    assert "Hello" not in log


@pytest.mark.asyncio
async def test_timeout_while_waiting_for_slot_does_not_call_asr(monkeypatch, caplog):
    async def handle(request):
        pytest.fail("A queued task must not reach HTTP after cancellation")

    install_transport(monkeypatch, handle)
    calibrate = history_asr.QwenAsrCalibration("http://asr.test/transcriptions", "model-test", 1, max_concurrency=2)
    await calibrate.slots.acquire()
    await calibrate.slots.acquire()
    try:
        with caplog.at_level("INFO", logger=history_asr.logger.name), pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(calibrate(snapshot()), timeout=0.02)
    finally:
        calibrate.slots.release()
        calibrate.slots.release()
    assert "status=cancelled phase=queue" in caplog.text


@pytest.mark.asyncio
async def test_http_failure_is_propagated_and_identified(monkeypatch, caplog):
    async def handle(request):
        return httpx.Response(503)

    install_transport(monkeypatch, handle)
    calibrate = history_asr.QwenAsrCalibration("http://asr.test/transcriptions", "model-test", 1)
    with caplog.at_level("INFO", logger=history_asr.logger.name), pytest.raises(httpx.HTTPStatusError):
        await calibrate(snapshot())
    assert "status=HTTPStatusError phase=http" in caplog.text


@pytest.mark.asyncio
async def test_http_cancellation_reaches_transport_and_releases_slot(monkeypatch):
    entered, cancelled = asyncio.Event(), asyncio.Event()

    async def handle(request):
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    install_transport(monkeypatch, handle)
    calibrate = history_asr.QwenAsrCalibration("http://asr.test/transcriptions", "model-test", 1, max_concurrency=1)
    task = asyncio.create_task(calibrate(snapshot()))
    await asyncio.wait_for(entered.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cancelled.is_set()
    await asyncio.wait_for(calibrate.slots.acquire(), 1)
    calibrate.slots.release()
