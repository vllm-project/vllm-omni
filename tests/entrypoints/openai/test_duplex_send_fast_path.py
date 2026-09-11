# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Protect encoded delivery, task isolation, revocation and cooperative progress."""

import asyncio
import contextvars
import json
from contextlib import suppress

import pytest

from vllm_omni.entrypoints.duplex.session_attachment import (
    DuplexSessionAttachmentRegistry,
    JournalEntry,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


async def _noop(*args):
    pass


@pytest.fixture(params=["default", "scheduled"])
def registry(request, monkeypatch):
    if request.param == "scheduled":
        # Exercise the older-Python path without changing the global loop policy.
        monkeypatch.setattr(asyncio, "eager_task_factory", None, raising=False)
    return DuplexSessionAttachmentRegistry(replay_ttl_s=60, replay_max_bytes_per_session=1024 * 1024)


@pytest.mark.asyncio
async def test_encoded_delivery_and_resume_reuse_the_immutable_snapshot(registry, mocker):
    initial, replay, control = [], [], []

    async def text_send(payload):
        initial.append(payload)

    async def replay_send(payload):
        replay.append(payload)

    async def dict_send(payload):
        control.append(payload)

    created = await registry.create("s", incarnation=0, send=dict_send, close=_noop, send_text=text_send)
    payload = {"type": "response.text.delta", "nested": [{"text": "中文😀"}]}
    entry = await registry.send_event("s", payload)
    assert isinstance(payload["nested"], list)
    payload["nested"][0]["text"] = "mutated"
    assert initial == [entry.encoded_payload.decode("utf-8")]
    assert not control
    assert len(initial[0].encode("utf-8")) == entry.encoded_bytes

    def no_decode(_entry):
        raise AssertionError("encoded replay must not decode the journal payload")

    mocker.patch.object(JournalEntry, "payload", property(no_decode))
    await registry.resume(
        "s",
        incarnation=0,
        resume_token=created.resume_token.plaintext,
        last_received_server_event_seq=0,
        send=dict_send,
        close=_noop,
        send_text=replay_send,
        activation_payload_factory=lambda *_: {"type": "session.resumed"},
    )
    assert replay == initial
    assert control == [{"type": "session.resumed"}]
    assert json.loads(replay[0])["nested"][0]["text"] == "中文😀"
    await registry.close("s")


@pytest.mark.asyncio
async def test_send_context_is_isolated_from_the_producer(registry):
    context = contextvars.ContextVar("duplex_send_context", default="producer")

    async def send(payload):
        context.set("transport")

    await registry.create("s", incarnation=0, send=send, close=_noop)
    await registry.send_event("s", {"type": "event"})
    assert context.get() == "producer"
    await registry.close("s")


@pytest.mark.asyncio
async def test_send_failure_is_observed_and_leaves_no_pending_work(registry):
    async def fail(payload):
        raise ValueError("send failed")

    await registry.create("s", incarnation=0, send=fail, close=_noop)
    with pytest.raises(ValueError, match="send failed"):
        await registry.send_event("s", {"type": "event"})
    attachment = await registry.close("s")
    await asyncio.sleep(0)
    assert not registry._transport_tasks
    assert not attachment.pending_sends
    assert not attachment.pending_completions


@pytest.mark.asyncio
@pytest.mark.parametrize("encoded", [False, True])
async def test_revocation_releases_producer_even_if_transport_delays_cancellation(registry, encoded):
    started, cancelled, release = (asyncio.Event() for _ in range(3))

    async def send(payload):
        started.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            cancelled.set()
            await release.wait()

    await registry.create("s", incarnation=0, send=send, close=_noop, send_text=send if encoded else None)
    producer = asyncio.create_task(registry.send_event("s", {"type": "event"}))
    try:
        await asyncio.wait_for(started.wait(), 1)
        attachment = await registry.close("s")
        assert (await asyncio.wait_for(producer, 1)).sequence == 1
        await asyncio.wait_for(cancelled.wait(), 1)
        assert registry._transport_tasks
        assert not attachment.pending_completions
    finally:
        release.set()
        producer.cancel()
        with suppress(asyncio.CancelledError):
            await producer
        if registry._transport_tasks:
            await asyncio.wait_for(asyncio.gather(*registry._transport_tasks, return_exceptions=True), 1)


@pytest.mark.asyncio
@pytest.mark.parametrize("encoded", [False, True])
async def test_caller_cancellation_reclaims_pending_send(registry, encoded):
    started = asyncio.Event()

    async def send(payload):
        started.set()
        await asyncio.Future()

    await registry.create("s", incarnation=0, send=send, close=_noop, send_text=send if encoded else None)
    producer = asyncio.create_task(registry.send_event("s", {"type": "event"}))
    await asyncio.wait_for(started.wait(), 1)
    producer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await producer
    attachment = await registry.close("s")
    if registry._transport_tasks:
        await asyncio.wait_for(asyncio.gather(*registry._transport_tasks, return_exceptions=True), 1)
    assert not attachment.pending_completions


@pytest.mark.asyncio
async def test_inline_send_burst_allows_control_task_progress(registry):
    sent = []
    observed = []

    async def send(payload):
        sent.append(payload["server_event_seq"])

    async def control():
        observed.append(len(sent))

    await registry.create("s", incarnation=0, send=send, close=_noop)
    peer = asyncio.create_task(control())
    for _ in range(128):
        await registry.send_event("s", {"type": "event"})
    assert peer.done()
    assert observed[0] < 128
    assert sent == list(range(1, 129))
    await registry.close("s")


@pytest.mark.asyncio
async def test_cancellation_after_send_failure_keeps_notification_non_exceptional(registry, mocker):
    started, release = asyncio.Event(), asyncio.Event()

    async def send(payload):
        started.set()
        await release.wait()
        raise ValueError("transport failed")

    await registry.create("s", incarnation=0, send=send, close=_noop)
    producer = asyncio.create_task(registry.send_event("s", {"type": "event"}))
    await asyncio.wait_for(started.wait(), 1)
    attachment = registry._sessions["s"].attachment
    completion = next(iter(attachment.pending_completions))
    send_task = next(iter(attachment.pending_sends))
    observed = mocker.spy(send_task, "exception")
    # Registered after sent(): publish completion, then cancel its recipient
    # before the recipient gets its next event-loop turn.
    send_task.add_done_callback(lambda _: producer.cancel())
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await producer
    assert observed.call_count > 0
    assert completion.result() is True
    await registry.close("s")


@pytest.mark.asyncio
@pytest.mark.parametrize("error_type", [ValueError, asyncio.CancelledError])
async def test_revocation_wins_over_a_completed_old_send_error(registry, error_type):
    started, release = asyncio.Event(), asyncio.Event()

    async def send(payload):
        started.set()
        await release.wait()
        raise error_type()

    await registry.create("s", incarnation=0, send=send, close=_noop)
    producer = asyncio.create_task(registry.send_event("s", {"type": "event"}))
    await asyncio.wait_for(started.wait(), 1)
    attachment = registry._sessions["s"].attachment
    send_task = next(iter(attachment.pending_sends))
    # The old send ends, then takeover revokes it before its producer wakes.
    send_task.add_done_callback(lambda _: attachment.revoke())
    release.set()
    assert (await asyncio.wait_for(producer, 1)).sequence == 1
    await registry.close("s")


@pytest.mark.asyncio
async def test_inline_send_error_after_revocation_does_not_fail_the_producer(registry):
    async def send(payload):
        registry._sessions["s"].attachment.revoke()
        raise ConnectionError("old socket closed during takeover")

    await registry.create("s", incarnation=0, send=send, close=_noop)
    assert (await registry.send_event("s", {"type": "event"})).sequence == 1
    await registry.close("s")


@pytest.mark.asyncio
async def test_custom_task_factory_is_not_bypassed(registry):
    loop = asyncio.get_running_loop()
    previous = loop.get_task_factory()
    created = []

    def factory(loop, coro, **kwargs):
        task = asyncio.Task(coro, loop=loop, **kwargs)
        created.append(task)
        return task

    await registry.create("s", incarnation=0, send=_noop, close=_noop)
    loop.set_task_factory(factory)
    try:
        await registry.send_event("s", {"type": "event"})
        assert len(created) == 1  # no additional Event.wait task per message
    finally:
        loop.set_task_factory(previous)
        await registry.close("s")


@pytest.mark.asyncio
async def test_future_returning_transport_is_supported(registry):
    def send(payload):
        result = asyncio.get_running_loop().create_future()
        result.set_result(None)
        return result

    await registry.create("s", incarnation=0, send=send, close=_noop)
    assert (await registry.send_event("s", {"type": "event"})).sequence == 1
    await registry.close("s")


def test_asgi_live_and_replay_do_not_decode_journal_entries(mocker):
    from tests.entrypoints.openai.test_duplex_lifecycle_safety import (
        test_asgi_websocket_takeover_replays_snapshot_and_preserves_session,
    )

    def no_decode(_entry):
        raise AssertionError("the real WebSocket path must use the encoded journal snapshot")

    mocker.patch.object(JournalEntry, "payload", property(no_decode))
    test_asgi_websocket_takeover_replays_snapshot_and_preserves_session()


@pytest.mark.parametrize("loop_kind", ["asyncio", "uvloop"])
def test_event_loop_compatibility(loop_kind):
    if loop_kind == "uvloop":
        uvloop = pytest.importorskip("uvloop")
        loop = uvloop.new_event_loop()
    else:
        loop = asyncio.new_event_loop()

    async def exercise():
        reg = DuplexSessionAttachmentRegistry(replay_ttl_s=60, replay_max_bytes_per_session=4096)
        await reg.create("s", incarnation=0, send=_noop, close=_noop)
        await reg.send_event("s", {"type": "event"})
        await reg.close("s")
        started = asyncio.Event()

        async def blocked(payload):
            started.set()
            await asyncio.Future()

        await reg.create("blocked", incarnation=0, send=blocked, close=_noop)
        producer = asyncio.create_task(reg.send_event("blocked", {"type": "event"}))
        await asyncio.wait_for(started.wait(), 1)
        await reg.close("blocked")
        assert (await asyncio.wait_for(producer, 1)).sequence == 1
        if reg._transport_tasks:
            await asyncio.wait_for(asyncio.gather(*reg._transport_tasks, return_exceptions=True), 1)
        await asyncio.sleep(0)

    try:
        loop.run_until_complete(exercise())
    finally:
        loop.close()
