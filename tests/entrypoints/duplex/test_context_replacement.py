# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
from copy import deepcopy

import pytest

from tests.entrypoints.openai_api.test_duplex_handler import (
    FakeChatService,
    FakeEngineClient,
    TimedWebSocket,
    _native_session_create,
    _pcm_f32_b64,
)
from vllm_omni.entrypoints.duplex.serving import OmniDuplexSessionHandler
from vllm_omni.model_executor.models.minicpmo_4_5.duplex.serving_adapter import MiniCPMO45ServingRuntimeAdapter

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.asyncio
@pytest.mark.parametrize("reject_validation", [False, True])
async def test_context_switch_fences_late_output_and_validation_is_nondestructive(monkeypatch, reject_validation):
    sid = "replacement-wire"
    observed: list[tuple[str, int]] = []

    class Engine(FakeEngineClient):
        async def signal_duplex_turn_async(self, session_id, *, event, context=None, **kwargs):
            if not event.startswith("context."):
                return await super().signal_duplex_turn_async(session_id, event=event, **kwargs)
            session = handler._registry.get(sid)
            observed.append((event, session.epoch))
            value: dict[str, object]
            if event == "context.validate":
                session.begin_response(turn_id=0)
                if reject_validation:
                    raise ValueError("invalid history unit")
                value = {"supported": True, "base_input_seq": 1, "base_config_generation": 0}
            else:
                assert session.epoch == 1
                _, emitted = await handler._send_one_native_duplex_event(
                    ws.send_json,
                    {"is_listen": False, "audio_data": "stale", "text": "old generation"},
                    session=session,
                    expected_epoch=0,
                )
                assert emitted is False
                value = {
                    "supported": True,
                    "event": "context.replace",
                    "epoch": 1,
                    "context_version": 1,
                    "resource_generation": 1,
                    "retained_unit_ids": ["u0-1"],
                    "dropped_unit_ids": [],
                }
            return {"ok": True, "stage_results": [{"stage_id": 0, "replica_id": 0, "result": value}]}

    engine = Engine()
    handler = OmniDuplexSessionHandler(chat_service=FakeChatService(engine), config_timeout_s=0.1, idle_timeout_s=1)

    def prepare(item, current, *, epoch):
        assert epoch == 0
        return {**deepcopy(current), "duplex_context_version": 1, "gander_context_version": 1}, {
            "event_id": "edit",
            "base_version": 0,
            "edits": [],
        }

    monkeypatch.setattr(MiniCPMO45ServingRuntimeAdapter, "prepare_context_replacement", staticmethod(prepare))

    def close_after_result(ws, event):
        if event.get("type") in {"input.context.replaced", "error"}:
            ws.put({"type": "session.close"})

    ws = TimedWebSocket(on_send=close_after_result, receive_timeout_s=2)
    ws.put(_native_session_create(sid, modalities=["text"]))
    ws.put(
        {
            "type": "input_audio_buffer.append",
            "audio": _pcm_f32_b64(16000),
            "format": "pcm_f32le",
            "sample_rate_hz": 16000,
            "duration_ms": 1000,
            "is_speech": True,
        }
    )
    ws.put({"type": "input.context.replace", "context": {"kind": "history_edit", "event_id": "edit", "epoch": 0}})
    await handler.handle_session(ws)
    if reject_validation:
        assert observed == [("context.validate", 0)]
        assert not any(e.get("type") == "audio.cancelled" and e.get("reason") == "context_replaced" for e in ws.sent)
        assert "input.context.replaced" not in ws.sent_types()
    else:
        assert observed == [("context.validate", 0), ("context.replace", 1)]
        assert "audio.cancelled" in ws.sent_types()
        ack = next(e for e in ws.sent if e.get("type") == "input.context.replaced")
        assert ack["epoch"] == 1 and ack["context_version"] == 1
        assert all(e.get("audio_data") != "stale" for e in ws.sent)
        assert "error" not in ws.sent_types()


@pytest.mark.asyncio
async def test_context_fence_retires_old_audio_from_resume_journal():
    from vllm_omni.entrypoints.duplex.session_attachment import DuplexJournalGapError, DuplexSessionAttachmentRegistry

    async def sink(*args):
        pass

    registry = DuplexSessionAttachmentRegistry(
        replay_ttl_s=60, replay_max_bytes_per_session=10000, disconnect_grace_s=30
    )
    created = await registry.create("s", incarnation=0, send=sink, close=sink)
    await registry.send_event("s", {"type": "response.audio.delta", "delta": "old audio"})
    boundary = await registry.invalidate_replay("s")
    ack = await registry.send_event("s", {"type": "input.context.replaced", "epoch": 1})
    assert ack.sequence > boundary
    with pytest.raises(DuplexJournalGapError):
        await registry.resume(
            "s",
            incarnation=0,
            resume_token=created.resume_token.plaintext,
            last_received_server_event_seq=0,
            send=sink,
            close=sink,
        )


@pytest.mark.parametrize("reason", ["context_replaced", "model_interrupt"])
def test_realtime_context_cancellation_emits_explicit_playback_clear(reason):
    from vllm_omni.entrypoints.duplex.realtime_session import NativeRealtimeSessionProtocol

    protocol = NativeRealtimeSessionProtocol(TimedWebSocket())
    protocol.encode_outbound_event({"type": "response.created", "response_id": "r", "epoch": 0})
    events = protocol.encode_outbound_event(
        {"type": "audio.cancelled", "response_id": "r", "epoch": 0, "reason": reason}
    )
    assert any(e.get("type") == "output_audio_buffer.cleared" and e.get("response_id") == "r" for e in events)


@pytest.mark.asyncio
async def test_context_snapshot_waits_for_preceding_input_submission():
    import asyncio

    entered, release = asyncio.Event(), asyncio.Event()
    inspections = []

    class Engine(FakeEngineClient):
        async def append_duplex_input_async(self, *args, **kwargs):
            entered.set()
            await release.wait()
            kwargs.pop("expected_epoch", None)
            return await super().append_duplex_input_async(*args, **kwargs)

        async def signal_duplex_turn_async(self, session_id, *, event, **kwargs):
            if event == "context.inspect":
                inspections.append(release.is_set())
                return {"ok": True, "stage_results": [{"result": {"supported": True, "epoch": 0, "units": []}}]}
            return await super().signal_duplex_turn_async(session_id, event=event, **kwargs)

    handler = OmniDuplexSessionHandler(chat_service=FakeChatService(Engine()), config_timeout_s=0.1, idle_timeout_s=2)

    def on_send(ws, e):
        if e.get("type") == "input.context.snapshot":
            ws.put({"type": "session.close"})

    ws = TimedWebSocket(on_send=on_send, receive_timeout_s=2)
    ws.put(_native_session_create("snapshot-order"))
    ws.put(
        {
            "type": "input_audio_buffer.append",
            "audio": _pcm_f32_b64(16000),
            "format": "pcm_f32le",
            "sample_rate_hz": 16000,
            "duration_ms": 1000,
            "is_speech": True,
        }
    )
    ws.put({"type": "input.context.get"})
    task = asyncio.create_task(handler.handle_session(ws))
    try:
        await asyncio.wait_for(entered.wait(), 1)
        for _ in range(5):
            await asyncio.sleep(0)
        assert not inspections
        release.set()
        await asyncio.wait_for(task, 3)
        assert inspections == [True]
        assert "error" not in ws.sent_types()
    finally:
        release.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_resume_accepts_context_before_new_audio(monkeypatch):
    from tests.entrypoints.openai_api.test_duplex_handler import _native_realtime_session_update

    handler = OmniDuplexSessionHandler(
        chat_service=FakeChatService(FakeEngineClient()), config_timeout_s=0.1, idle_timeout_s=1
    )
    sid = "resume-context"
    first = TimedWebSocket(receive_timeout_s=0.01)
    first.put(_native_realtime_session_update(sid))
    await handler.handle_realtime_session(first)
    created = next(e for e in first.sent if e.get("type") == "session.created")
    retained = handler._registry.get(sid)
    handler._runtime_session_state(retained).native_context_locked = True
    prepared = []

    def prepare(item, current, *, epoch):
        prepared.append(epoch)
        return dict(current), None

    monkeypatch.setattr(MiniCPMO45ServingRuntimeAdapter, "prepare_context_input", staticmethod(prepare))
    second = TimedWebSocket(receive_timeout_s=0.1)
    second.query_params = {"model": "openbmb/MiniCPM-o-4_5", "native_duplex": "1", "resume": "1"}
    second.put(
        {
            "type": "session.resume",
            "session_id": sid,
            "incarnation": created["incarnation"],
            "resume_token": created["resume_token"],
            "last_received_server_event_seq": created.get("server_event_seq", 0),
        }
    )
    second.put(
        {"type": "input.context.append", "context": {"kind": "runtime_event", "event_id": "after-resume", "epoch": 0}}
    )
    second.put({"type": "session.close"})
    await handler.handle_realtime_session(second)
    assert prepared == [0]
    assert not any(e.get("code") == "context_not_initialized" for e in second.sent)


@pytest.mark.asyncio
async def test_invalid_replacement_waits_for_committed_append_receipt(monkeypatch):
    import asyncio

    entered, release = asyncio.Event(), asyncio.Event()
    resumed_audio = asyncio.Event()
    validations = []
    cancelled: list[bool] = []

    class Engine(FakeEngineClient):
        calls = 0

        async def append_duplex_input_async(self, *args, **kwargs):
            self.calls += 1
            if self.calls == 3:
                assert not cancelled, "uncertain previous operation"
                resumed_audio.set()
            if self.calls == 1:
                kwargs.pop("expected_epoch", None)
                return await super().append_duplex_input_async(*args, **kwargs)
            entered.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                cancelled.append(True)
                raise
            kwargs.pop("expected_epoch", None)
            return await super().append_duplex_input_async(*args, **kwargs)

        async def signal_duplex_turn_async(self, session_id, *, event, **kwargs):
            if event == "context.validate":
                validations.append(release.is_set())
                raise ValueError("unknown history unit")
            return await super().signal_duplex_turn_async(session_id, event=event, **kwargs)

    def prepare(item, current, *, epoch):
        return {**current, "duplex_context_version": 1}, {"base_version": 0, "edits": []}

    monkeypatch.setattr(MiniCPMO45ServingRuntimeAdapter, "prepare_context_replacement", staticmethod(prepare))
    handler = OmniDuplexSessionHandler(chat_service=FakeChatService(Engine()), config_timeout_s=0.1, idle_timeout_s=2)

    def on_send(ws, event):
        if event.get("type") == "error":
            ws.put(
                {
                    "type": "input_audio_buffer.append",
                    "audio": _pcm_f32_b64(16000),
                    "format": "pcm_f32le",
                    "sample_rate_hz": 16000,
                    "duration_ms": 1000,
                    "is_speech": True,
                }
            )

    ws = TimedWebSocket(receive_timeout_s=2, on_send=on_send)
    sid = "receipt-barrier"
    ws.put(_native_session_create(sid))
    ws.put(
        {
            "type": "input_audio_buffer.append",
            "audio": _pcm_f32_b64(16000),
            "format": "pcm_f32le",
            "sample_rate_hz": 16000,
            "duration_ms": 1000,
            "is_speech": True,
        }
    )
    ws.put(
        {
            "type": "input_audio_buffer.append",
            "audio": _pcm_f32_b64(16000),
            "format": "pcm_f32le",
            "sample_rate_hz": 16000,
            "duration_ms": 1000,
            "is_speech": True,
        }
    )
    task = asyncio.create_task(handler.handle_session(ws))
    try:
        await asyncio.wait_for(entered.wait(), 1)
        ws.put(
            {"type": "input.context.replace", "context": {"kind": "history_edit", "event_id": "bad-edit", "epoch": 0}}
        )
        for _ in range(10):
            await asyncio.sleep(0)
        assert not validations and not cancelled
        release.set()
        await asyncio.wait_for(resumed_audio.wait(), 2)
        ws.put({"type": "session.close"})
        await asyncio.wait_for(task, 3)
        assert validations == [True]
        assert not cancelled
    finally:
        release.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
