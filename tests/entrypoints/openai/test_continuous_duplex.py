# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Continuous native streams complete at a delivery fence, not token EOS."""

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

from tests.entrypoints.openai_api.test_duplex_handler import (
    FakeChatService,
    FakeEngineClient,
    TimedWebSocket,
    _auto_response_context,
    _native_session_create,
    _pcm_f32_b64,
)
from vllm_omni.entrypoints.duplex.protocol import DuplexCapabilities
from vllm_omni.entrypoints.duplex.serving import OmniDuplexSessionHandler
from vllm_omni.model_executor.models.minicpmo_4_5.duplex.capabilities import minicpmo45_native_capabilities
from vllm_omni.model_executor.models.nemotron_voicechat.duplex.data_plane import NemotronVoiceChatDataPlaneSession
from vllm_omni.model_executor.models.personaplex.duplex.serving_adapter import PersonaPlexServingRuntimeAdapter

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.asyncio
@pytest.mark.parametrize("partial", [False, True])
@pytest.mark.parametrize("drop_output", [False, True])
async def test_personaplex_close_waits_for_pending_append_and_audio(mocker, partial, drop_output):
    entered, release = asyncio.Event(), asyncio.Event()
    outputs: asyncio.Queue[object] = asyncio.Queue()

    class Engine(FakeEngineClient):
        async def append_duplex_input_async(self, session_id, **kwargs):
            kwargs.pop("expected_epoch", None)
            await super().append_duplex_input_async(session_id, **kwargs)
            seq = len(self.appended)
            if seq == 2:
                entered.set()
                await release.wait()
                if not drop_output:
                    outputs.put_nowait(
                        [
                            SimpleNamespace(
                                request_id="req-personaplex-close",
                                multimodal_output={"audio": np.ones(1920, dtype=np.float32), "sr": 24000},
                            )
                        ]
                    )
            return {
                "ok": True,
                "stage_results": [
                    {
                        "result": {
                            "data_plane_append": True,
                            "request_id": "req-personaplex-close",
                            "response_stage_id": 1,
                            "seq": seq,
                        }
                    }
                ],
            }

        async def collect_duplex_data_plane_outputs_async(self, _request_id, *, response_stage_id=None, timeout=None):
            try:
                return await asyncio.wait_for(outputs.get(), timeout)
            except asyncio.TimeoutError:
                return []

    engine = Engine()
    adapter = PersonaPlexServingRuntimeAdapter(lambda *_: "audio")
    mocker.patch.object(adapter, "prepare_runtime_config", new=mocker.AsyncMock(return_value={}))
    handler = OmniDuplexSessionHandler(
        chat_service=FakeChatService(engine),
        serving_runtime_adapter=adapter,
        idle_timeout_s=5,
    )
    ws = TimedWebSocket(receive_timeout_s=5)
    create = _native_session_create("sid-personaplex-close", modalities=["audio", "text"])
    create["session"]["extra_body"] = {"auto_response": True, "duplex_control_timeout_s": 1.0}
    ws.put(create)
    for count in [1920, 960 if partial else 1920]:
        ws.put(
            {
                "type": "input_audio_buffer.append",
                "audio": _pcm_f32_b64(count),
                "format": "pcm_f32le",
                "sample_rate_hz": 24000,
            }
        )
    ws.put({"type": "session.close"})
    task = asyncio.create_task(handler.handle_session(ws))
    try:
        await asyncio.wait_for(entered.wait(), 3)
        assert not engine.closed
        assert not task.done()
        release.set()
        await asyncio.wait_for(task, 3)
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    assert len(engine.appended) == 2
    completed = [e for e in ws.sent if e.get("type") == "response.done" and e.get("status") == "completed"]
    if drop_output:
        assert not completed
        assert any(e.get("code") == "stream_drain_failed" for e in ws.sent)
    else:
        assert len(completed) == 1
        assert completed[0]["drain"]["audio_frames"] == 1
        assert completed[0]["drain"]["model_delay_frames"] == 1
        assert "audio.cancelled" not in ws.sent_types()
        assert "session.end" in ws.sent_types()


@pytest.mark.asyncio
@pytest.mark.parametrize("visible_response", [False, True])
async def test_personaplex_does_not_inherit_minicpm_silence_continuation(mocker, visible_response):
    handler, session = _auto_response_context("sid-personaplex-no-synthetic-input")
    handler._serving_runtime_adapter = PersonaPlexServingRuntimeAdapter(lambda *_: "audio")
    session.capabilities = handler._serving_runtime_adapter.capabilities(max_sessions=2)
    session.bind_request("personaplex-retained-request")
    if visible_response:
        session.begin_response()
    native = handler._runtime_session_state(session)
    native.silence_continuation_scheduler = mocker.AsyncMock(return_value=True)
    send = mocker.AsyncMock()
    start_drain = mocker.patch.object(handler, "_start_native_data_plane_stream_task", new=mocker.AsyncMock())

    await handler._arm_native_model_decision(
        send, session=session, expected_epoch=session.epoch, expected_model_turn_id=session.turn_id
    )
    await handler._maybe_continue_native_response(
        send, session=session, expected_epoch=session.epoch, expected_model_turn_id=session.turn_id
    )

    start_drain.assert_not_awaited()
    native.silence_continuation_scheduler.assert_not_awaited()
    assert native.continuation_units == 0
    send.assert_not_awaited()


def test_continuous_lifecycle_is_explicit_and_requires_delivery_tracking():
    assert DuplexCapabilities().response_lifecycle == "model_turn"
    with pytest.raises(ValueError, match="response lifecycle"):
        DuplexCapabilities(response_lifecycle="unknown")
    handler, session = _auto_response_context("sid-invalid-stream-contract")
    session.capabilities = minicpmo45_native_capabilities()
    session.capabilities.response_lifecycle = "continuous_stream"
    reason = handler._native_runtime_contract_error(session)
    assert reason is not None and "data_plane.drain_status is missing" in reason


@pytest.mark.asyncio
async def test_continuous_explicit_cancel_is_not_reported_as_completed():
    handler, session = _auto_response_context("sid-stream-cancel")
    session.capabilities = minicpmo45_native_capabilities()
    session.capabilities.response_lifecycle = "continuous_stream"
    session.bind_request("req-stream-cancel")
    response_id = session.begin_response()
    session.mark_audio_sent(80)
    ws = TimedWebSocket()
    assert await handler._cancel_active_response(session, None, ws.send_json, reason="client_cancelled")
    assert any(e.get("type") == "audio.cancelled" and e.get("response_id") == response_id for e in ws.sent)
    assert not any(e.get("type") == "response.done" and e.get("status") == "completed" for e in ws.sent)


@pytest.mark.asyncio
@pytest.mark.parametrize("drop_output", [False, True])
@pytest.mark.parametrize("samples", [640, 1280])
async def test_close_drains_accepted_append_or_reports_failure(drop_output, samples):
    request_id = "req-continuous-close"
    entered = asyncio.Event()
    release = asyncio.Event()
    outputs: asyncio.Queue[object] = asyncio.Queue()

    class ContinuousEngine(FakeEngineClient):
        async def append_duplex_input_async(self, session_id, **kwargs):
            kwargs.pop("expected_epoch", None)
            await super().append_duplex_input_async(session_id, **kwargs)
            entered.set()
            await release.wait()
            if not drop_output:
                outputs.put_nowait(
                    [
                        SimpleNamespace(
                            stage_id=0,
                            request_id=request_id,
                            outputs=[
                                SimpleNamespace(
                                    multimodal_output={"nvc_text_token_ids": [42]},
                                )
                            ],
                        ),
                        SimpleNamespace(
                            stage_id=2,
                            request_id=request_id,
                            outputs=[
                                SimpleNamespace(
                                    multimodal_output={"audio": np.ones(1764, dtype=np.float32), "sr": 22050},
                                )
                            ],
                        ),
                    ]
                )
            return {
                "ok": True,
                "stage_results": [
                    {
                        "result": {
                            "data_plane_append": True,
                            "request_id": request_id,
                            "response_stage_id": 2,
                            "seq": 1,
                        }
                    }
                ],
            }

        async def collect_duplex_data_plane_outputs_async(self, _request_id, *, response_stage_id=None, timeout=None):
            try:
                return await asyncio.wait_for(outputs.get(), timeout=timeout)
            except asyncio.TimeoutError:
                return []

    engine = ContinuousEngine()
    handler = OmniDuplexSessionHandler(chat_service=FakeChatService(engine), idle_timeout_s=5)
    projector = NemotronVoiceChatDataPlaneSession(lambda *_: "audio")
    projector.configure_runtime(
        {
            "nvc_text_bos_id": 1,
            "nvc_text_eos_id": 2,
            "nvc_text_pad_id": 12,
            "nvc_function_sotc_id": 20,
            "nvc_function_eotc_id": 21,
            "nvc_tokenizer_ref": "test",
        },
        tokenizer=SimpleNamespace(decode=lambda *_args, **_kwargs: "hello"),
    )

    def on_send(ws, event):
        if event.get("type") == "session.created":
            session = handler._registry.get("sid-continuous-close")
            session.capabilities.response_lifecycle = "continuous_stream"
            session.capabilities.chunk_period_ms = 80
            # The fake MiniCPM setup is only transport scaffolding. The
            # projector below owns this test's audio, so no voice weights load.
            session.config.modalities = ["text", "audio"]
            session.config.extra_body["duplex_control_timeout_s"] = 0.25
            handler._serving_runtime_adapter.data_plane = projector
            ws.put({"type": "input_audio_buffer.append", "audio": _pcm_f32_b64(samples), "format": "pcm_f32le"})
            if samples < 1280:
                ws.put({"type": "session.close"})

    ws = TimedWebSocket(on_send=on_send, receive_timeout_s=5)
    create = _native_session_create("sid-continuous-close")
    create["session"]["extra_body"]["auto_response"] = True
    ws.put(create)
    task = asyncio.create_task(handler.handle_session(ws))
    started = asyncio.create_task(entered.wait())
    try:
        await asyncio.wait({task, started}, timeout=2, return_when=asyncio.FIRST_COMPLETED)
        assert entered.is_set(), ws.sent
        if samples == 1280:
            ws.put({"type": "session.close"})
        await asyncio.sleep(0)
        assert engine.closed == []
        assert not task.done()
        release.set()
        await asyncio.wait_for(task, timeout=2)
    finally:
        started.cancel()
        await asyncio.gather(started, return_exceptions=True)
        release.set()
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    assert len(engine.appended) == 1
    assert "session.closed" in ws.sent_types()
    completed = [e for e in ws.sent if e.get("type") == "response.done" and e.get("status") == "completed"]
    if drop_output:
        assert completed == []
        assert any(e.get("code") == "stream_drain_failed" for e in ws.sent)
    else:
        assert len(completed) == 1
        assert completed[0]["drain"] == {"accepted_frames": 1, "text_frames": 1, "audio_frames": 1, "drained": True}
        assert ws.sent_types().index("response.output_audio.delta") < ws.sent_types().index("response.done")
        assert "audio.cancelled" not in ws.sent_types()
        assert "session.end" in ws.sent_types()


@pytest.mark.asyncio
async def test_continuous_response_does_not_autogenerate_silence(mocker):
    handler, session = _auto_response_context("sid-continuous-no-synthetic-input")
    session.capabilities.response_lifecycle = "continuous_stream"
    session.bind_request("req-continuous")
    session.begin_response()
    native = handler._runtime_session_state(session)
    native.silence_continuation_scheduler = mocker.AsyncMock()
    await handler._maybe_continue_native_response(None, session=session, expected_epoch=session.epoch)
    await handler._arm_native_model_decision(
        None, session=session, expected_epoch=session.epoch, expected_model_turn_id=0
    )
    native.silence_continuation_scheduler.assert_not_awaited()
    assert native.continuation_owner_id is None
