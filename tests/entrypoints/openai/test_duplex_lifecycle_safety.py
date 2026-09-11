# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Weight-free regressions for the shared duplex serving lifecycle."""

import asyncio
import json
from contextlib import suppress

import pytest

from vllm_omni.config.stage_config import DuplexSessionRuntimeConfig
from vllm_omni.engine.duplex.lease import DuplexLeaseActivity
from vllm_omni.engine.duplex.messages import DuplexFence, DuplexSessionLifecycleMessage
from vllm_omni.entrypoints.duplex.protocol import DuplexSessionConfig
from vllm_omni.entrypoints.duplex.realtime_session import NativeRealtimeSessionProtocol
from vllm_omni.entrypoints.duplex.serving import OmniDuplexSessionHandler
from vllm_omni.entrypoints.duplex.session_attachment import (
    DuplexEventJournal,
    DuplexSessionAttachmentRegistry,
    InvalidResumeTokenError,
)
from vllm_omni.model_executor.models.minicpmo_4_5.duplex.serving_adapter import MiniCPMO45ServingRuntimeAdapter
from vllm_omni.model_executor.models.nemotron_voicechat.duplex.serving_adapter import (
    NemotronVoiceChatServingRuntimeAdapter,
)
from vllm_omni.model_executor.models.personaplex.duplex.serving_adapter import PersonaPlexServingRuntimeAdapter

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_attachment_io_timeout_rejects_unbounded_values(value):
    with pytest.raises(ValueError, match="finite and positive"):
        DuplexSessionRuntimeConfig(attachment_io_timeout_s=value)


def test_handler_uses_server_owned_attachment_deadline(shared_handler):
    handler, _, _ = shared_handler
    assert handler._attachment_registry._transport_timeout_s == handler._duplex_session_config.attachment_io_timeout_s


async def _noop(*args, **kwargs):
    pass


def _registry():
    return DuplexSessionAttachmentRegistry(replay_ttl_s=60, replay_max_bytes_per_session=8192)


def test_journal_freezes_real_projector_events_and_accounts_snapshot_bytes():
    protocol = NativeRealtimeSessionProtocol({})
    journal = DuplexEventJournal(max_bytes=4096, ttl_s=60)
    initial = protocol.encode_outbound_event({"type": "response.created", "response_id": "r", "modalities": ["text"]})
    entries = [journal.record(event) for event in initial]
    original = json.loads(json.dumps([dict(entry.payload) for entry in entries]))
    for event in protocol.encode_outbound_event(
        {"type": "response.text.delta", "response_id": "r", "delta": "x" * 1024}
    ):
        journal.record(event)
    replay = journal.replay_after(0)
    assert [dict(entry.payload) for entry in replay[: len(entries)]] == original
    assert next(entry.payload["item"]["content"] for entry in entries if "item" in entry.payload) == []
    assert journal.retained_bytes == sum(
        len(json.dumps(dict(entry.payload), separators=(",", ":"), ensure_ascii=False).encode()) for entry in replay
    )
    assert journal.retained_bytes <= 4096


def test_journal_replay_consumer_cannot_mutate_retained_snapshot():
    journal = DuplexEventJournal(max_bytes=4096, ttl_s=60)
    entry = journal.record({"nested": [{"text": "原始"}]})
    entry.payload["nested"][0]["text"] = "changed"
    assert journal.replay_after(0)[0].payload["nested"][0]["text"] == "原始"


@pytest.mark.asyncio
async def test_takeover_revokes_blocked_send_without_cancelling_event_producer():
    registry = _registry()
    started = asyncio.Event()
    wire = []

    async def blocked_send(payload):
        started.set()
        await asyncio.Future()

    async def new_send(payload):
        wire.append(payload)

    created = await registry.create("s", incarnation=0, send=blocked_send, close=_noop)
    producer = asyncio.create_task(registry.send_event("s", {"type": "before"}))
    try:
        await asyncio.wait_for(started.wait(), 1)
        resumed = await asyncio.wait_for(
            registry.resume(
                "s",
                incarnation=0,
                resume_token=created.resume_token.plaintext,
                last_received_server_event_seq=0,
                send=new_send,
                close=_noop,
                activation_payload_factory=lambda token, generation: {"type": "session.resumed"},
            ),
            1,
        )
        assert resumed.attachment_generation == 2
        assert (await asyncio.wait_for(producer, 1)).sequence == 1
        await registry.send_event("s", {"type": "after"})
        assert [event["type"] for event in wire] == ["session.resumed", "before", "after"]
    finally:
        producer.cancel()
        with suppress(asyncio.CancelledError):
            await producer
        await registry.close("s")


@pytest.mark.asyncio
async def test_cancelled_resume_preserves_last_delivered_credential():
    registry = _registry()
    started = asyncio.Event()

    async def blocked_send(payload):
        started.set()
        await asyncio.Future()

    created = await registry.create("s", incarnation=0, send=_noop, close=_noop)
    await registry.detach("s", attachment_generation=1)
    task = asyncio.create_task(
        registry.resume(
            "s",
            incarnation=0,
            resume_token=created.resume_token.plaintext,
            last_received_server_event_seq=0,
            send=blocked_send,
            close=_noop,
            activation_payload_factory=lambda token, generation: {"type": "session.resumed"},
        )
    )
    await asyncio.wait_for(started.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    resumed = await registry.resume(
        "s",
        incarnation=0,
        resume_token=created.resume_token.plaintext,
        last_received_server_event_seq=0,
        send=_noop,
        close=_noop,
    )
    assert resumed.attachment_generation == 3
    with pytest.raises(InvalidResumeTokenError):
        await registry.authenticate_resume(
            "s",
            incarnation=0,
            resume_token=created.resume_token.plaintext,
            last_received_server_event_seq=0,
        )
    await registry.close("s")


@pytest.fixture(
    params=[MiniCPMO45ServingRuntimeAdapter, PersonaPlexServingRuntimeAdapter, NemotronVoiceChatServingRuntimeAdapter]
)
def shared_handler(request):
    from tests.entrypoints.openai_api.test_duplex_handler import FakeChatService, FakeEngineClient

    engine = FakeEngineClient()
    adapter = request.param(lambda *args: None)
    handler = OmniDuplexSessionHandler(
        chat_service=FakeChatService(engine),
        serving_runtime_adapter=adapter,
        duplex_session_config=DuplexSessionRuntimeConfig(max_sessions=2),
    )
    handler._registry._capabilities = adapter.capabilities(max_sessions=2)
    return handler, adapter, engine


@pytest.mark.asyncio
@pytest.mark.parametrize("blocked_callback", ["send", "close"])
async def test_lifecycle_reclaims_peer_without_waiting_for_network(shared_handler, blocked_callback):
    handler, adapter, engine = shared_handler
    started, release = asyncio.Event(), asyncio.Event()

    async def blocked(*args):
        started.set()
        await release.wait()

    for sid in ("a", "b"):
        session = handler._registry.create(DuplexSessionConfig(model="test"), session_id=sid)
        adapter.session_states[sid] = adapter.create_session_state()
        await handler._attachment_registry.create(
            sid,
            incarnation=session.incarnation,
            send=blocked if sid == "a" and blocked_callback == "send" else _noop,
            close=blocked if sid == "a" and blocked_callback == "close" else _noop,
        )
        engine.duplex_lifecycle_events.put_nowait(
            DuplexSessionLifecycleMessage(
                fence=DuplexFence(sid, incarnation=session.incarnation),
                session_id=sid,
                event="expired",
                reason="test",
                lease_generation=1,
                submitted_request_ids=[],
                reserved_request_ids=[],
            )
        )
    handler._ensure_lifecycle_listener()
    listener = handler._lifecycle_task
    try:
        await asyncio.wait_for(started.wait(), 1)
        await asyncio.wait_for(engine.duplex_lifecycle_events.join(), 1)
        assert handler._registry.active_count() == 0
        assert not adapter.session_states
    finally:
        release.set()
        if listener is not None:
            listener.cancel()
            with suppress(asyncio.CancelledError):
                await listener
        notices = getattr(handler, "_lifecycle_notifications", ())
        if notices:
            await asyncio.wait_for(asyncio.gather(*notices), 1)


@pytest.mark.asyncio
async def test_finished_response_state_is_released_and_deleted_history_stays_deleted():
    protocol = NativeRealtimeSessionProtocol({})
    for i in range(600):
        rid = f"r-{i}"
        protocol.encode_outbound_event({"type": "response.created", "response_id": rid, "modalities": ["text"]})
        protocol.encode_outbound_event({"type": "response.text.delta", "response_id": rid, "delta": "x" * 1024})
        done = protocol.encode_outbound_event({"type": "response.done", "response_id": rid})
        assert next(p for p in done if p["type"] == "response.done")["response"]["output"][0]["content"]
        await protocol._to_duplex_event({"type": "conversation.item.delete", "item_id": f"item_{rid}"})
        protocol._pending_outbound.get_nowait()
    assert not protocol._response_states
    assert not protocol._conversation_items
    assert len(protocol._completed_response_ids) <= 256
    # A response older than the bounded terminal cache must not be recreated by a late delta.
    assert protocol.encode_outbound_event({"type": "response.text.delta", "response_id": "r-0", "delta": "late"}) == []
    assert protocol.encode_outbound_event({"type": "response.done", "response_id": "r-0"}) == []
    assert not protocol._response_states
    assert not protocol._conversation_items


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_during", ["runtime", "activation"])
@pytest.mark.parametrize(
    "shared_handler", [MiniCPMO45ServingRuntimeAdapter, NemotronVoiceChatServingRuntimeAdapter], indirect=True
)
async def test_resume_cancellation_keeps_engine_cas_and_credentials_consistent(shared_handler, mocker, cancel_during):
    handler, adapter, engine = shared_handler
    session = handler._registry.create(DuplexSessionConfig(model="test"), session_id="resume")
    created = await handler._attachment_registry.create(
        "resume", incarnation=session.incarnation, send=_noop, close=_noop
    )
    await handler._attachment_registry.detach("resume", attachment_generation=1)
    started, release = asyncio.Event(), asyncio.Event()
    generations = []
    original_resume = engine.resume_duplex_session_async

    async def resume_runtime(sid, **kwargs):
        generations.append(kwargs["expected_lease_generation"])
        assert kwargs["expected_lease_generation"] == len(generations) - 1
        if cancel_during == "runtime" and len(generations) == 1:
            started.set()
            await release.wait()
        return await original_resume(sid, **kwargs)

    async def activation(payload):
        if cancel_during == "activation":
            started.set()
            await release.wait()

    mocker.patch.object(engine, "resume_duplex_session_async", side_effect=resume_runtime)
    event = {
        "session_id": "resume",
        "incarnation": session.incarnation,
        "resume_token": created.resume_token.plaintext,
        "last_received_server_event_seq": 0,
    }
    task = asyncio.create_task(
        handler._resume_session_handshake(
            event,
            send_json=_noop,
            realtime_protocol=NativeRealtimeSessionProtocol({}),
            attachment_send=activation,
            attachment_close=_noop,
        )
    )
    try:
        await asyncio.wait_for(started.wait(), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        release.set()
        resumed = await asyncio.wait_for(
            handler._resume_session_handshake(
                event,
                send_json=_noop,
                realtime_protocol=NativeRealtimeSessionProtocol({}),
                attachment_send=_noop,
                attachment_close=_noop,
            ),
            1,
        )
        assert resumed is not None
        assert handler._lease_generations["resume"] == 2
        assert generations == [0, 1]
        assert engine.touched == [("resume", DuplexLeaseActivity.DETACH)]
    finally:
        release.set()
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
        await handler._attachment_registry.close("resume")


@pytest.mark.asyncio
async def test_invalid_resume_cannot_revoke_a_live_attachment():
    registry = _registry()
    created = await registry.create("s", incarnation=0, send=_noop, close=_noop)
    with pytest.raises(InvalidResumeTokenError):
        await registry.resume(
            "s", incarnation=0, resume_token="invalid", last_received_server_event_seq=0, send=_noop, close=_noop
        )
    assert await registry.is_current_attachment("s", created.attachment_generation)
    await registry.close("s")


@pytest.mark.asyncio
async def test_resume_delivery_timeout_keeps_recovery_credential():
    registry = DuplexSessionAttachmentRegistry(
        replay_ttl_s=60, replay_max_bytes_per_session=8192, transport_timeout_s=0.01
    )
    created = await registry.create("s", incarnation=0, send=_noop, close=_noop)

    async def blocked(payload):
        await asyncio.Future()

    with pytest.raises(TimeoutError):
        await registry.resume(
            "s",
            incarnation=0,
            resume_token=created.resume_token.plaintext,
            last_received_server_event_seq=0,
            send=blocked,
            close=_noop,
            activation_payload_factory=lambda *_: {"type": "session.resumed"},
        )
    assert (
        await registry.resume(
            "s",
            incarnation=0,
            resume_token=created.resume_token.plaintext,
            last_received_server_event_seq=0,
            send=_noop,
            close=_noop,
        )
    ).attachment_generation == 3
    await registry.close("s")


def test_completed_audio_retains_truncation_metadata_without_response_body_copy():
    protocol = NativeRealtimeSessionProtocol({})
    protocol.encode_outbound_event({"type": "response.created", "response_id": "r"})
    protocol.encode_outbound_event(
        {
            "type": "response.output_audio.delta",
            "response_id": "r",
            "text": "hello world",
            "audio_duration_ms": 1000,
            "audio": "AAAAAA==",
            "format": "pcm16",
            "sample_rate_hz": 24000,
            "audio_text_marks": [{"audio_end_ms": 500, "text_chars": 5}, {"audio_end_ms": 1000, "text_chars": 11}],
        }
    )
    protocol.encode_outbound_event({"type": "response.done", "response_id": "r"})
    item = protocol._conversation_items["item_r"]
    # This is the retained conversation item, not the released assembly state.
    assert item["content"][0]["audio_duration_ms"] == 1000
    assert len(item["content"][0]["audio_text_marks"]) == 2
    protocol.encode_outbound_event(
        {"type": "conversation.item.truncated", "item_id": "item_r", "content_index": 0, "audio_end_ms": 500}
    )
    assert item["content"][0]["transcript"] == "hello"
    assert not protocol._response_states
    assert protocol.encode_outbound_event({"type": "response.text.delta", "response_id": "r", "delta": "late"}) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("shared_handler", [PersonaPlexServingRuntimeAdapter], indirect=True)
async def test_unadvertised_resume_is_rejected_without_engine_or_attachment_changes(shared_handler):
    handler, adapter, engine = shared_handler
    session = handler._registry.create(DuplexSessionConfig(model="test"), session_id="s")
    created = await handler._attachment_registry.create("s", incarnation=session.incarnation, send=_noop, close=_noop)
    errors = []

    async def send_error(payload):
        errors.append(payload)

    result = await handler._resume_session_handshake(
        {"session_id": "s", "incarnation": session.incarnation, "resume_token": created.resume_token.plaintext},
        send_json=send_error,
        realtime_protocol=NativeRealtimeSessionProtocol({}),
        attachment_send=_noop,
        attachment_close=_noop,
    )
    assert result is None
    assert errors[0]["code"] == "unsupported_session_resume"
    assert not engine.resumed
    assert await handler._attachment_registry.is_current_attachment("s", 1)
    await handler._attachment_registry.close("s")


@pytest.mark.asyncio
async def test_lifecycle_peer_progresses_while_response_task_cancellation_drains(shared_handler):
    from vllm_omni.entrypoints.duplex.websocket import DuplexSessionTasks

    handler, adapter, engine = shared_handler
    started, cancelling, release, peer_closed = (asyncio.Event() for _ in range(4))

    async def delayed_cancellation():
        started.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            cancelling.set()
            await release.wait()

    async def close_peer(reason):
        peer_closed.set()

    response = asyncio.create_task(delayed_cancellation())
    await asyncio.wait_for(started.wait(), 1)
    handler._session_tasks["a"] = DuplexSessionTasks(active_response_task=response)
    for sid in ("a", "b"):
        session = handler._registry.create(DuplexSessionConfig(model="test"), session_id=sid)
        adapter.session_states[sid] = adapter.create_session_state()
        await handler._attachment_registry.create(
            sid, incarnation=session.incarnation, send=_noop, close=close_peer if sid == "b" else _noop
        )
        engine.duplex_lifecycle_events.put_nowait(
            DuplexSessionLifecycleMessage(
                fence=DuplexFence(sid, incarnation=session.incarnation),
                session_id=sid,
                event="expired",
                reason="test",
                lease_generation=1,
                submitted_request_ids=[],
                reserved_request_ids=[],
            )
        )
    handler._ensure_lifecycle_listener()
    try:
        await asyncio.wait_for(cancelling.wait(), 1)
        await asyncio.wait_for(peer_closed.wait(), 1)
        assert handler._registry.get("b") is None
        assert "b" not in adapter.session_states
        assert not response.done()
    finally:
        release.set()
        await asyncio.wait_for(engine.duplex_lifecycle_events.join(), 1)
        if handler._lifecycle_notifications:
            await asyncio.wait_for(asyncio.gather(*handler._lifecycle_notifications), 1)


@pytest.mark.asyncio
async def test_takeover_is_not_held_by_transport_that_acknowledges_cancel_late():
    registry = _registry()
    started, cancelling, release = (asyncio.Event() for _ in range(3))

    async def old_send(payload):
        started.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            cancelling.set()
            await release.wait()

    created = await registry.create("s", incarnation=0, send=old_send, close=_noop)
    producer = asyncio.create_task(registry.send_event("s", {"type": "old"}))
    try:
        await asyncio.wait_for(started.wait(), 1)
        resumed = await asyncio.wait_for(
            registry.resume(
                "s",
                incarnation=0,
                resume_token=created.resume_token.plaintext,
                last_received_server_event_seq=0,
                send=_noop,
                close=_noop,
                activation_payload_factory=lambda *_: {"type": "session.resumed"},
            ),
            1,
        )
        await asyncio.wait_for(cancelling.wait(), 1)
        assert (await producer).sequence == 1
        assert await registry.is_current_attachment("s", resumed.attachment_generation)
        assert registry._transport_tasks  # late callbacks remain owned, not fire-and-forget
    finally:
        release.set()
        producer.cancel()
        with suppress(asyncio.CancelledError):
            await producer
        if registry._transport_tasks:
            await asyncio.wait_for(asyncio.gather(*registry._transport_tasks, return_exceptions=True), 1)
        await registry.close("s")


@pytest.mark.asyncio
async def test_close_during_resume_cannot_resurrect_attachment():
    registry = _registry()
    started = asyncio.Event()

    async def blocked_send(payload):
        started.set()
        await asyncio.Future()

    created = await registry.create("s", incarnation=0, send=_noop, close=_noop)
    task = asyncio.create_task(
        registry.resume(
            "s",
            incarnation=0,
            resume_token=created.resume_token.plaintext,
            last_received_server_event_seq=0,
            send=blocked_send,
            close=_noop,
            activation_payload_factory=lambda *_: {"type": "session.resumed"},
        )
    )
    await asyncio.wait_for(started.wait(), 1)
    await registry.close("s")
    with pytest.raises(ConnectionError):
        await asyncio.wait_for(task, 1)
    assert not await registry.is_current_attachment("s", 2)
    with pytest.raises(KeyError):
        await registry.authenticate_resume(
            "s", incarnation=0, resume_token=created.resume_token.plaintext, last_received_server_event_seq=0
        )


def test_asgi_websocket_takeover_replays_snapshot_and_preserves_session():
    """Exercise real ASGI WebSockets; only the weight-bearing engine is a fake."""
    from fastapi import FastAPI, WebSocket
    from fastapi.testclient import TestClient

    from tests.entrypoints.openai_api.test_duplex_handler import (
        FakeChatService,
        FakeEngineClient,
        _native_realtime_session_update,
    )

    engine = FakeEngineClient()
    handler = OmniDuplexSessionHandler(chat_service=FakeChatService(engine))
    app = FastAPI()

    @app.websocket("/v1/realtime")
    async def realtime(websocket: WebSocket):
        await handler.handle_realtime_session(websocket)

    async def publish_response():
        protocol = handler._realtime_protocols["wire"]
        for event in (
            {"type": "response.created", "response_id": "r", "modalities": ["text"]},
            {"type": "response.text.delta", "response_id": "r", "delta": "hello"},
            {"type": "response.done", "response_id": "r"},
        ):
            for payload in protocol.encode_outbound_event(event):
                await handler._attachment_registry.send_event("wire", payload)

    with TestClient(app) as client:
        with client.websocket_connect("/v1/realtime") as first:
            first.send_json(_native_realtime_session_update("wire"))
            created = first.receive_json()
            assert created["type"] == "session.created"
            client.portal.call(publish_response)
            with client.websocket_connect("/v1/realtime?resume=1") as second:
                second.send_json(
                    {
                        "type": "session.resume",
                        "session_id": "wire",
                        "incarnation": created["incarnation"],
                        "resume_token": created["resume_token"],
                        "last_received_server_event_seq": created.get("server_event_seq", 0),
                    }
                )
                activation = second.receive_json()
                assert activation["type"] == "session.resumed"
                assert activation["resume_token"] != created["resume_token"]
                replay = []
                for _ in range(20):
                    event = second.receive_json()
                    replay.append(event)
                    if event["type"] == "response.done":
                        break
                assert replay[-1]["type"] == "response.done"
                assert next(e for e in replay if e["type"] == "response.output_item.added")["item"]["content"] == []
                assert replay[-1]["response"]["output"][0]["content"][0]["text"] == "hello"
                second.send_json({"type": "session.close"})
                for _ in range(10):
                    if second.receive_json()["type"] == "session.closed":
                        break
                else:
                    pytest.fail("session.close did not complete")
        assert engine.opened == ["wire"]
        assert engine.resumed == [("wire", 0)]
        assert engine.closed == [("wire", "session_close")]
        assert handler._registry.active_count() == 0


@pytest.mark.asyncio
async def test_expired_connection_late_finally_cannot_remove_reopened_session():
    from tests.entrypoints.openai_api.test_duplex_handler import (
        FakeChatService,
        FakeEngineClient,
        TimedWebSocket,
        _native_realtime_session_update,
    )

    engine = FakeEngineClient()
    handler = OmniDuplexSessionHandler(chat_service=FakeChatService(engine))
    old_created, new_created, close_started, release_close = (asyncio.Event() for _ in range(4))

    class DelayedClose(TimedWebSocket):
        async def close(self, code=1000, reason=None):
            close_started.set()
            await release_close.wait()
            await super().close(code, reason)

    first = DelayedClose(
        on_send=lambda _, p: old_created.set() if p["type"] == "session.created" else None, receive_timeout_s=10
    )
    second = TimedWebSocket(
        on_send=lambda _, p: new_created.set() if p["type"] == "session.created" else None, receive_timeout_s=10
    )
    first.put(_native_realtime_session_update("reuse"))
    old_task = asyncio.create_task(handler.handle_realtime_session(first))
    new_task = None
    try:
        await asyncio.wait_for(old_created.wait(), 1)
        old_session = handler._registry.get("reuse")
        engine.duplex_lifecycle_events.put_nowait(
            DuplexSessionLifecycleMessage(
                fence=engine.opened_fences[0],
                session_id="reuse",
                event="expired",
                reason="test",
                lease_generation=1,
                submitted_request_ids=[],
                reserved_request_ids=[],
            )
        )
        await asyncio.wait_for(engine.duplex_lifecycle_events.join(), 1)
        await asyncio.wait_for(close_started.wait(), 1)
        second.put(_native_realtime_session_update("reuse"))
        new_task = asyncio.create_task(handler.handle_realtime_session(second))
        await asyncio.wait_for(new_created.wait(), 1)
        new_session = handler._registry.get("reuse")
        assert new_session is not old_session
        assert new_session.incarnation > old_session.incarnation
        release_close.set()
        await asyncio.wait_for(old_task, 1)
        assert handler._registry.get("reuse") is new_session
        assert "reuse" in handler._session_tasks
        assert "reuse" in handler._serving_runtime_adapter.session_states
        second.put({"type": "session.close"})
        await asyncio.wait_for(new_task, 1)
        assert engine.closed == [("reuse", "session_close")]
    finally:
        release_close.set()
        for task in (old_task, new_task):
            if task is not None:
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
        if handler._lifecycle_notifications:
            await asyncio.wait_for(asyncio.gather(*handler._lifecycle_notifications), 1)


@pytest.mark.asyncio
async def test_replay_cancellation_can_recover_with_delivered_rotated_token():
    registry = _registry()
    replay_started = asyncio.Event()
    delivered = []

    async def partial_send(payload):
        if payload["type"] == "session.resumed":
            delivered.append(payload["resume_token"])
        else:
            replay_started.set()
            await asyncio.Future()

    created = await registry.create("s", incarnation=0, send=_noop, close=_noop)
    await registry.send_event("s", {"type": "history"})
    task = asyncio.create_task(
        registry.resume(
            "s",
            incarnation=0,
            resume_token=created.resume_token.plaintext,
            last_received_server_event_seq=0,
            send=partial_send,
            close=_noop,
            activation_payload_factory=lambda token, _: {"type": "session.resumed", "resume_token": token.plaintext},
        )
    )
    await asyncio.wait_for(replay_started.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert delivered
    resumed = await registry.resume(
        "s", incarnation=0, resume_token=delivered[0], last_received_server_event_seq=0, send=_noop, close=_noop
    )
    assert resumed.attachment_generation == 3
    with pytest.raises(InvalidResumeTokenError):
        await registry.authenticate_resume(
            "s", incarnation=0, resume_token=created.resume_token.plaintext, last_received_server_event_seq=0
        )
    await registry.close("s")


@pytest.mark.asyncio
async def test_terminal_send_failure_still_closes_retired_transport():
    registry = _registry()
    closed = []

    async def broken_send(payload):
        raise ConnectionError("peer gone")

    async def close(reason):
        closed.append(reason)

    await registry.create("s", incarnation=0, send=broken_send, close=close)
    attachment = await registry.close("s")
    await registry.retire_attachment(attachment, "expired", {"type": "session.expired"})
    assert closed == ["expired"]


@pytest.mark.asyncio
async def test_resume_does_not_require_python_311_timeout(monkeypatch):
    monkeypatch.delattr(asyncio, "timeout", raising=False)
    sent = []

    async def send(payload):
        sent.append(payload)

    async def close(reason):
        pass

    registry = DuplexSessionAttachmentRegistry(replay_ttl_s=60, replay_max_bytes_per_session=4096)
    created = await registry.create("resume-310", incarnation=0, send=send, close=close)
    await registry.send_event("resume-310", {"type": "test.event"})
    resumed = await registry.resume(
        "resume-310",
        incarnation=0,
        resume_token=created.resume_token.plaintext,
        last_received_server_event_seq=0,
        send=send,
        close=close,
        activation_payload_factory=lambda token, generation: {"type": "session.resumed", "generation": generation},
    )
    assert resumed.attachment_generation == 2
    assert [item["type"] for item in sent] == ["test.event", "session.resumed", "test.event"]
    await registry.close("resume-310")


@pytest.mark.asyncio
async def test_slow_send_times_out_and_preserves_replay_for_takeover():
    registry = DuplexSessionAttachmentRegistry(
        replay_ttl_s=60, replay_max_bytes_per_session=8192, transport_timeout_s=0.02
    )
    closed = asyncio.Event()

    async def blocked_send(payload):
        await asyncio.Future()

    async def close(reason):
        assert reason == "send_timeout"
        closed.set()

    created = await registry.create("slow", incarnation=0, send=blocked_send, close=close)
    try:
        with pytest.raises(TimeoutError, match="send"):
            await asyncio.wait_for(registry.send_event("slow", {"type": "audio"}), 1)
        await asyncio.wait_for(closed.wait(), 1)
        assert not await registry.is_current_attachment("slow", 1)
        assert await registry.is_current_attachment("slow", 1, include_revoked=True)
        wire = []

        async def send(payload):
            wire.append(payload)

        resumed = await registry.resume(
            "slow",
            incarnation=0,
            resume_token=created.resume_token.plaintext,
            last_received_server_event_seq=0,
            send=send,
            close=_noop,
            activation_payload_factory=lambda token, generation: {"type": "session.resumed"},
        )
        assert resumed.attachment_generation == 2
        assert [event["type"] for event in wire] == ["session.resumed", "audio"]
        await registry.send_event("slow", {"type": "after"})
        assert [event["type"] for event in wire] == ["session.resumed", "audio", "after"]
    finally:
        await registry.close("slow")


@pytest.mark.asyncio
async def test_slow_send_deadline_does_not_wait_for_transport_cancellation():
    registry = DuplexSessionAttachmentRegistry(
        replay_ttl_s=60, replay_max_bytes_per_session=8192, transport_timeout_s=0.02
    )
    release = asyncio.Event()
    cancelled = asyncio.Event()

    async def stubborn_send(payload):
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            cancelled.set()
            await release.wait()

    await registry.create("slow", incarnation=0, send=stubborn_send, close=_noop)
    await registry.create("peer", incarnation=0, send=_noop, close=_noop)
    try:
        with pytest.raises(TimeoutError, match="send"):
            await asyncio.wait_for(registry.send_event("slow", {"type": "audio"}), 1)
        await asyncio.wait_for(cancelled.wait(), 1)
        await asyncio.wait_for(registry.send_event("peer", {"type": "healthy"}), 1)
        await asyncio.wait_for(registry.close("slow"), 1)
    finally:
        release.set()
        await registry.close("slow")
        await registry.close("peer")
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_send_timeout_does_not_skip_session_owner_cleanup():
    from tests.entrypoints.openai_api.test_duplex_handler import (
        FakeChatService,
        FakeEngineClient,
        TimedWebSocket,
        _native_realtime_session_update,
    )

    engine = FakeEngineClient()
    handler = OmniDuplexSessionHandler(
        chat_service=FakeChatService(engine),
        duplex_session_config=DuplexSessionRuntimeConfig(attachment_io_timeout_s=0.02),
    )
    blocked = asyncio.Event()

    class SlowWebSocket(TimedWebSocket):
        async def send_json(self, payload):
            if payload.get("type") == "session.updated":
                blocked.set()
                await asyncio.Future()
            await super().send_json(payload)

    websocket = SlowWebSocket(receive_timeout_s=10)
    websocket.put(_native_realtime_session_update("slow-cleanup"))
    websocket.put({"type": "session.update", "session": {"instructions": "updated"}})
    task = asyncio.create_task(handler.handle_realtime_session(websocket))
    try:
        await asyncio.wait_for(blocked.wait(), 2)
        await asyncio.wait_for(task, 2)
        assert handler._registry.get("slow-cleanup") is None
        assert "slow-cleanup" not in handler._session_tasks
        assert "slow-cleanup" not in handler._serving_runtime_adapter.session_states
        assert len(engine.closed) == 1
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
