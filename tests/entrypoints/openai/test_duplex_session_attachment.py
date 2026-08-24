# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import asyncio
import base64
import json

import pytest

from vllm_omni.experimental.fullduplex.openai.session_attachment import (
    DuplexEventJournal,
    DuplexJournalGapError,
    DuplexJournalOverflowError,
    DuplexResumeCredential,
    DuplexSessionAttachmentRegistry,
    InvalidResumeTokenError,
    ResumeToken,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def _decoded_token_bytes(token: ResumeToken) -> bytes:
    padding = "=" * (-len(token.plaintext) % 4)
    return base64.urlsafe_b64decode(token.plaintext + padding)


def test_resume_credential_uses_256_bit_token_digest_and_redacted_repr(monkeypatch) -> None:
    token = ResumeToken.generate()
    credential = DuplexResumeCredential.from_token(token)
    calls = []

    import vllm_omni.experimental.fullduplex.openai.session_attachment as attachment_module

    original_compare = attachment_module.hmac.compare_digest

    def recording_compare(left, right):
        calls.append((left, right))
        return original_compare(left, right)

    monkeypatch.setattr(attachment_module.hmac, "compare_digest", recording_compare)

    assert len(_decoded_token_bytes(token)) == 32
    assert credential.verify(token.plaintext) is True
    assert credential.verify("invalid-token") is False
    assert len(calls) == 2
    assert token.plaintext not in repr(token)
    assert token.plaintext not in repr(credential)
    assert not hasattr(credential, "plaintext")
    assert len(credential.token_digest) == 32


def test_resume_credential_rotation_revokes_old_token() -> None:
    first = ResumeToken.generate()
    credential = DuplexResumeCredential.from_token(first)

    second = credential.rotate()

    assert credential.verify(first.plaintext) is False
    assert credential.verify(second.plaintext) is True
    assert second.plaintext != first.plaintext


def test_event_journal_sequences_acknowledges_and_replays_exact_payloads() -> None:
    clock = _Clock()
    journal = DuplexEventJournal(max_bytes=4096, ttl_s=60.0, clock=clock)

    first = journal.record({"type": "response.audio.delta", "delta": "AAAA"})
    second = journal.record({"type": "response.done", "response_id": "resp-1"})

    assert first.sequence == 1
    assert second.sequence == 2
    assert first.payload["server_event_seq"] == 1
    assert second.payload["server_event_seq"] == 2
    assert first.encoded_bytes == len(
        json.dumps(dict(first.payload), separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    )
    assert [entry.sequence for entry in journal.replay_after(0)] == [1, 2]
    assert journal.acknowledge(1) == 1
    assert [entry.sequence for entry in journal.replay_after(1)] == [2]
    assert journal.acknowledge(1) == 0


def test_event_journal_ttl_pruning_reports_replay_gap() -> None:
    clock = _Clock()
    journal = DuplexEventJournal(max_bytes=4096, ttl_s=5.0, clock=clock)
    journal.record({"type": "one"})
    clock.advance(3.0)
    journal.record({"type": "two"})
    clock.advance(3.0)

    assert journal.prune() == 1
    with pytest.raises(DuplexJournalGapError, match="older than retained"):
        journal.replay_after(0)
    assert [entry.payload["type"] for entry in journal.replay_after(1)] == ["two"]


def test_event_journal_overflow_is_explicit_and_does_not_evict_silently() -> None:
    clock = _Clock()
    journal = DuplexEventJournal(max_bytes=100, ttl_s=60.0, clock=clock)
    first = journal.record({"type": "small", "value": "x"})

    with pytest.raises(DuplexJournalOverflowError, match="byte limit"):
        journal.record({"type": "large", "value": "x" * 200})

    assert journal.overflowed is True
    assert journal.retained_bytes == first.encoded_bytes
    with pytest.raises(DuplexJournalGapError, match="overflowed"):
        journal.replay_after(0)


@pytest.mark.asyncio
async def test_registry_resume_rotates_token_replays_and_atomically_replaces_attachment() -> None:
    clock = _Clock()
    registry = DuplexSessionAttachmentRegistry(
        replay_ttl_s=60.0,
        replay_max_bytes_per_session=4096,
        clock=clock,
    )
    sends_a = []
    closes_a = []
    sends_b = []
    closes_b = []

    async def send_a(payload):
        sends_a.append(payload)

    async def close_a(reason):
        closes_a.append(reason)

    async def send_b(payload):
        sends_b.append(payload)

    async def close_b(reason):
        closes_b.append(reason)

    created = await registry.create("sid", incarnation=2, send=send_a, close=close_a)
    await registry.send_event("sid", {"type": "event-1"})
    await registry.send_event("sid", {"type": "event-2"})

    resumed = await registry.resume(
        "sid",
        incarnation=2,
        resume_token=created.resume_token.plaintext,
        last_received_server_event_seq=1,
        send=send_b,
        close=close_b,
    )

    assert resumed.attachment_generation == 2
    assert resumed.resume_token.plaintext != created.resume_token.plaintext
    assert [entry.sequence for entry in resumed.replay_entries] == [2]
    assert resumed.replaced_attachment is not None
    assert resumed.replaced_attachment.generation == 1
    assert await registry.is_current_attachment("sid", created.attachment_generation) is False
    assert await registry.is_current_attachment("sid", resumed.attachment_generation) is True
    assert await registry.detach("sid", attachment_generation=1) is False
    assert await registry.detach("sid", attachment_generation=2) is True
    assert [(payload["type"], payload["server_event_seq"]) for payload in sends_a] == [
        ("event-1", 1),
        ("event-2", 2),
    ]
    assert closes_a == [] and sends_b == [] and closes_b == []


@pytest.mark.asyncio
async def test_registry_resume_sends_activation_then_replay_before_new_live_events() -> None:
    registry = DuplexSessionAttachmentRegistry(
        replay_ttl_s=60.0,
        replay_max_bytes_per_session=4096,
    )
    wire = []
    activation_started = asyncio.Event()
    release_activation = asyncio.Event()

    async def old_send(payload):
        del payload

    async def close(reason):
        del reason

    async def new_send(payload):
        wire.append(payload["type"])
        if payload["type"] == "session.resumed":
            activation_started.set()
            await release_activation.wait()

    created = await registry.create("sid-order", incarnation=0, send=old_send, close=close)
    await registry.detach("sid-order", attachment_generation=1)
    await registry.send_event("sid-order", {"type": "replayed"})
    resume_task = asyncio.create_task(
        registry.resume(
            "sid-order",
            incarnation=0,
            resume_token=created.resume_token.plaintext,
            last_received_server_event_seq=0,
            send=new_send,
            close=close,
            activation_payload_factory=lambda _token, _generation: {"type": "session.resumed"},
        )
    )
    await activation_started.wait()
    live_task = asyncio.create_task(registry.send_event("sid-order", {"type": "live"}))

    release_activation.set()
    await asyncio.gather(resume_task, live_task)

    assert wire == ["session.resumed", "replayed", "live"]


@pytest.mark.asyncio
async def test_registry_resume_delivery_failure_keeps_one_shot_old_token_recovery() -> None:
    registry = DuplexSessionAttachmentRegistry(
        replay_ttl_s=60.0,
        replay_max_bytes_per_session=4096,
    )

    async def send(payload):
        del payload

    async def failing_send(payload):
        del payload
        raise RuntimeError("transport lost before rotated token arrived")

    async def close(reason):
        del reason

    created = await registry.create("sid-recovery", incarnation=0, send=send, close=close)
    await registry.detach("sid-recovery", attachment_generation=1)

    with pytest.raises(RuntimeError, match="transport lost"):
        await registry.resume(
            "sid-recovery",
            incarnation=0,
            resume_token=created.resume_token.plaintext,
            last_received_server_event_seq=0,
            send=failing_send,
            close=close,
            activation_payload_factory=lambda token, generation: {
                "type": "session.resumed",
                "resume_token": token.plaintext,
                "attachment_generation": generation,
            },
        )

    recovered = await registry.resume(
        "sid-recovery",
        incarnation=0,
        resume_token=created.resume_token.plaintext,
        last_received_server_event_seq=0,
        send=send,
        close=close,
    )

    assert recovered.attachment_generation == 3
    with pytest.raises(InvalidResumeTokenError):
        await registry.resume(
            "sid-recovery",
            incarnation=0,
            resume_token=created.resume_token.plaintext,
            last_received_server_event_seq=0,
            send=send,
            close=close,
        )


@pytest.mark.asyncio
async def test_registry_concurrent_resume_allows_exactly_one_rotated_token_winner() -> None:
    registry = DuplexSessionAttachmentRegistry(
        replay_ttl_s=60.0,
        replay_max_bytes_per_session=4096,
    )

    async def send(payload):
        del payload

    async def close(reason):
        del reason

    created = await registry.create("sid-race", incarnation=0, send=send, close=close)

    async def attempt():
        try:
            return await registry.resume(
                "sid-race",
                incarnation=0,
                resume_token=created.resume_token.plaintext,
                last_received_server_event_seq=0,
                send=send,
                close=close,
            )
        except InvalidResumeTokenError as exc:
            return exc

    results = await asyncio.gather(attempt(), attempt())

    assert sum(not isinstance(result, Exception) for result in results) == 1
    assert sum(isinstance(result, InvalidResumeTokenError) for result in results) == 1


@pytest.mark.asyncio
async def test_registry_keeps_sessions_journals_and_incarnations_isolated() -> None:
    registry = DuplexSessionAttachmentRegistry(
        replay_ttl_s=60.0,
        replay_max_bytes_per_session=4096,
    )

    async def send(payload):
        del payload

    async def close(reason):
        del reason

    created_a = await registry.create("sid-a", incarnation=1, send=send, close=close)
    created_b = await registry.create("sid-b", incarnation=4, send=send, close=close)
    await registry.detach("sid-a", attachment_generation=1)
    await registry.detach("sid-b", attachment_generation=1)
    await registry.send_event("sid-a", {"type": "a-only"})
    await registry.send_event("sid-b", {"type": "b-only"})

    resumed_a = await registry.resume(
        "sid-a",
        incarnation=1,
        resume_token=created_a.resume_token.plaintext,
        last_received_server_event_seq=0,
        send=send,
        close=close,
    )
    resumed_b = await registry.resume(
        "sid-b",
        incarnation=4,
        resume_token=created_b.resume_token.plaintext,
        last_received_server_event_seq=0,
        send=send,
        close=close,
    )

    assert [entry.payload["type"] for entry in resumed_a.replay_entries] == ["a-only"]
    assert [entry.payload["type"] for entry in resumed_b.replay_entries] == ["b-only"]
    with pytest.raises(ValueError, match="incarnation mismatch"):
        await registry.resume(
            "sid-a",
            incarnation=2,
            resume_token=resumed_a.resume_token.plaintext,
            last_received_server_event_seq=1,
            send=send,
            close=close,
        )


@pytest.mark.asyncio
async def test_registry_repr_never_contains_plaintext_tokens() -> None:
    registry = DuplexSessionAttachmentRegistry(
        replay_ttl_s=60.0,
        replay_max_bytes_per_session=4096,
    )

    async def send(payload):
        del payload

    async def close(reason):
        del reason

    created = await registry.create("sid-repr", incarnation=0, send=send, close=close)
    await registry.detach("sid-repr", attachment_generation=1)
    entry = await registry.send_event(
        "sid-repr",
        {"type": "session.resumed", "resume_token": created.resume_token.plaintext},
    )

    assert entry is not None
    assert created.resume_token.plaintext not in repr(created)
    assert created.resume_token.plaintext not in repr(entry)
    assert created.resume_token.plaintext not in repr(registry)
