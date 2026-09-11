# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import math
import secrets
import time
from collections import deque
from collections.abc import Awaitable, Callable, Iterator, Mapping
from dataclasses import dataclass, field
from itertools import count
from types import MappingProxyType


class InvalidResumeTokenError(RuntimeError):
    pass


class DuplexJournalGapError(RuntimeError):
    pass


class DuplexJournalOverflowError(RuntimeError):
    pass


@dataclass(frozen=True)
class ResumeToken:
    plaintext: str = field(repr=False)

    @classmethod
    def generate(cls) -> ResumeToken:
        return cls(secrets.token_urlsafe(32))


@dataclass
class DuplexResumeCredential:
    token_digest: bytes

    @classmethod
    def from_token(cls, token: ResumeToken) -> DuplexResumeCredential:
        return cls(token_digest=cls._digest(token.plaintext))

    @staticmethod
    def _digest(plaintext: str) -> bytes:
        return hashlib.sha256(plaintext.encode("utf-8")).digest()

    def verify(self, plaintext: str) -> bool:
        return hmac.compare_digest(self.token_digest, self._digest(plaintext))

    def rotate(self) -> ResumeToken:
        token = ResumeToken.generate()
        self.token_digest = self._digest(token.plaintext)
        return token


@dataclass(frozen=True)
class DuplexTransportAttachment:
    generation: int
    send: Callable[[dict[str, object]], Awaitable[None]] = field(repr=False)
    close: Callable[[str], Awaitable[None]] = field(repr=False)
    send_text: Callable[[str], Awaitable[None]] | None = field(default=None, repr=False, kw_only=True)
    revoked: asyncio.Event = field(default_factory=asyncio.Event, repr=False, compare=False)
    pending_sends: set[asyncio.Future[None]] = field(default_factory=set, repr=False, compare=False)
    pending_completions: set[asyncio.Future[bool]] = field(default_factory=set, repr=False, compare=False)
    _inline_sends: Iterator[int] = field(default_factory=count, repr=False, compare=False)

    def revoke(self) -> None:
        """Wake pending producers without awaiting transport cancellation."""
        self.revoked.set()
        for completion in self.pending_completions:
            if not completion.done():
                completion.set_result(False)


@dataclass(frozen=True)
class JournalEntry:
    """An immutable wire snapshot; readers receive independently decoded payloads."""

    sequence: int
    created_monotonic: float
    encoded_payload: bytes = field(repr=False)

    @property
    def encoded_bytes(self) -> int:
        return len(self.encoded_payload)

    @property
    def payload(self) -> Mapping[str, object]:
        return MappingProxyType(json.loads(self.encoded_payload))


class DuplexEventJournal:
    def __init__(
        self,
        *,
        max_bytes: int,
        ttl_s: float,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if max_bytes <= 0:
            raise ValueError("journal max_bytes must be positive")
        if ttl_s <= 0:
            raise ValueError("journal ttl_s must be positive")
        self._max_bytes = max_bytes
        self._ttl_s = ttl_s
        self._clock = clock or time.monotonic
        self._entries: deque[JournalEntry] = deque()
        self._next_sequence = 1
        self._dropped_through = 0
        self._retained_bytes = 0
        self._overflowed = False

    @property
    def retained_bytes(self) -> int:
        return self._retained_bytes

    @property
    def overflowed(self) -> bool:
        return self._overflowed

    @property
    def last_sequence(self) -> int:
        return self._next_sequence - 1

    def record(self, payload: Mapping[str, object]) -> JournalEntry:
        if self._overflowed:
            raise DuplexJournalOverflowError("duplex event journal already exceeded its byte limit")
        self.prune()
        sequence = self._next_sequence
        sequenced_payload = dict(payload)
        sequenced_payload["server_event_seq"] = sequence
        encoded_payload = json.dumps(sequenced_payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        encoded_bytes = len(encoded_payload)
        if self._retained_bytes + encoded_bytes > self._max_bytes:
            self._overflowed = True
            raise DuplexJournalOverflowError(
                f"duplex event journal byte limit exceeded: {self._retained_bytes + encoded_bytes} > {self._max_bytes}"
            )
        entry = JournalEntry(
            sequence=sequence,
            created_monotonic=self._clock(),
            encoded_payload=encoded_payload,
        )
        self._entries.append(entry)
        self._retained_bytes += encoded_bytes
        self._next_sequence += 1
        return entry

    def acknowledge(self, sequence: int) -> int:
        if sequence < 0:
            raise ValueError("acknowledged sequence must not be negative")
        if sequence > self.last_sequence:
            raise ValueError(f"acknowledged sequence {sequence} is newer than journal head {self.last_sequence}")
        removed = 0
        while self._entries and self._entries[0].sequence <= sequence:
            entry = self._entries.popleft()
            self._retained_bytes -= entry.encoded_bytes
            removed += 1
        self._dropped_through = max(self._dropped_through, sequence)
        return removed

    def prune(self, now: float | None = None) -> int:
        effective_now = self._clock() if now is None else now
        cutoff = effective_now - self._ttl_s
        removed = 0
        while self._entries and self._entries[0].created_monotonic <= cutoff:
            entry = self._entries.popleft()
            self._retained_bytes -= entry.encoded_bytes
            self._dropped_through = max(self._dropped_through, entry.sequence)
            removed += 1
        return removed

    def replay_after(self, sequence: int) -> tuple[JournalEntry, ...]:
        if sequence < 0:
            raise ValueError("replay sequence must not be negative")
        self.prune()
        if self._overflowed:
            raise DuplexJournalGapError("duplex event journal overflowed; replay is incomplete")
        if sequence < self._dropped_through:
            raise DuplexJournalGapError(
                f"requested sequence {sequence} is older than retained journal boundary {self._dropped_through}"
            )
        if sequence > self.last_sequence:
            raise ValueError(f"requested sequence {sequence} is newer than journal head {self.last_sequence}")
        return tuple(entry for entry in self._entries if entry.sequence > sequence)


@dataclass(frozen=True)
class DuplexSessionAttachmentCreated:
    session_id: str
    incarnation: int
    attachment_generation: int
    resume_token: ResumeToken = field(repr=False)


@dataclass(frozen=True)
class DuplexSessionResumeResult:
    session_id: str
    incarnation: int
    attachment_generation: int
    resume_token: ResumeToken = field(repr=False)
    replay_entries: tuple[JournalEntry, ...] = ()
    replaced_attachment: DuplexTransportAttachment | None = None


@dataclass
class _DuplexSessionAttachmentState:
    session_id: str
    incarnation: int
    credential: DuplexResumeCredential
    journal: DuplexEventJournal
    attachment: DuplexTransportAttachment | None
    attachment_generation: int
    outbound_lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)
    resume_lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)
    handshake_lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)
    recovery_token_digest: bytes | None = field(default=None, repr=False)
    grace_task: asyncio.Task[None] | None = field(default=None, repr=False)


class DuplexSessionAttachmentRegistry:
    def __init__(
        self,
        *,
        replay_ttl_s: float,
        replay_max_bytes_per_session: int,
        disconnect_grace_s: float = 30.0,
        clock: Callable[[], float] | None = None,
        transport_timeout_s: float = 5.0,
    ) -> None:
        if replay_ttl_s <= 0:
            raise ValueError("replay_ttl_s must be positive")
        if replay_max_bytes_per_session <= 0:
            raise ValueError("replay_max_bytes_per_session must be positive")
        if disconnect_grace_s <= 0:
            raise ValueError("disconnect_grace_s must be positive")
        if not math.isfinite(transport_timeout_s) or transport_timeout_s <= 0:
            raise ValueError("transport_timeout_s must be finite and positive")
        self._replay_ttl_s = replay_ttl_s
        self._replay_max_bytes_per_session = replay_max_bytes_per_session
        self._disconnect_grace_s = disconnect_grace_s
        self._clock = clock or time.monotonic
        self._sessions: dict[str, _DuplexSessionAttachmentState] = {}
        self._lock = asyncio.Lock()
        self._transport_timeout_s = transport_timeout_s
        self._transport_tasks: set[asyncio.Future[None]] = set()

    def handshake_lock(self, session_id: str) -> asyncio.Lock:
        """Serialize engine lease CAS and attachment activation for one session."""
        return self._require(session_id).handshake_lock

    def _track_transport(self, operation: Awaitable[None], *, eager: bool = False) -> asyncio.Future[None]:
        # Python 3.12 can finish nonblocking writes inline in an isolated Task
        # context. Respect custom loop factories; older Python keeps the normal
        # scheduled path. Never install a process-wide eager task factory.
        factory = getattr(asyncio, "eager_task_factory", None) if eager else None
        loop = asyncio.get_running_loop()
        if factory is not None and loop.get_task_factory() is None and asyncio.iscoroutine(operation):
            task = factory(loop, operation)
        else:
            task = asyncio.ensure_future(operation)
        if task.done():
            if not task.cancelled():
                task.exception()
            return task
        self._transport_tasks.add(task)

        def completed(done: asyncio.Future[None]) -> None:
            self._transport_tasks.discard(done)
            if not done.cancelled():
                done.exception()

        task.add_done_callback(completed)
        return task

    async def _send_attachment(
        self, attachment: DuplexTransportAttachment, payload: dict[str, object] | JournalEntry
    ) -> bool:
        """Release the ordered producer on revocation, without cancelling that producer.

        A transport callback never owns the session's producer task. Even a
        callback slow to acknowledge cancellation cannot hold the outbound lock
        after its attachment is revoked. Its late completion cannot mutate a
        replacement attachment. All callback tasks remain tracked until done.
        """
        if attachment.revoked.is_set():
            return False
        if isinstance(payload, JournalEntry):
            operation = (
                attachment.send_text(payload.encoded_payload.decode("utf-8"))
                if attachment.send_text is not None
                else attachment.send(dict(payload.payload))
            )
        else:
            operation = attachment.send(payload)
        send = self._track_transport(operation, eager=True)
        if send.done():
            if attachment.revoked.is_set():
                return False
            send.result()
            # A buffered producer can otherwise monopolize the loop when all
            # writes complete inline. Bound the burst without yielding per event.
            if next(attachment._inline_sends) % 32 == 31:
                await asyncio.sleep(0)
            return not attachment.revoked.is_set()

        completion = asyncio.get_running_loop().create_future()
        attachment.pending_sends.add(send)
        attachment.pending_completions.add(completion)

        def sent(done: asyncio.Future[None]) -> None:
            attachment.pending_sends.discard(done)
            if not completion.done():
                # This is only a wakeup, not an error carrier. Check revocation
                # before reading the send result, including same-tick races.
                completion.set_result(True)

        send.add_done_callback(sent)
        try:
            done, _ = await asyncio.wait((completion,), timeout=self._transport_timeout_s)
            if not done:
                attachment.revoke()
                self._track_transport(self.retire_attachment(attachment, "send_timeout"))
                raise TimeoutError("duplex attachment send timed out")
            if attachment.revoked.is_set():
                return False
            send.result()
            return True
        finally:
            attachment.pending_completions.discard(completion)
            if not send.done():
                send.cancel()

    async def retire_attachment(
        self, attachment: DuplexTransportAttachment, reason: str, payload: dict[str, object] | None = None
    ) -> None:
        """Best-effort terminal notification and close, each with a bounded wait."""
        attachment.revoke()

        async def bounded(operation: Awaitable[None]) -> None:
            task = self._track_transport(operation)
            try:
                await asyncio.wait((task,), timeout=self._transport_timeout_s)
            finally:
                if not task.done():
                    task.cancel()

        try:
            # Never start a second write while a revoked callback is draining.
            if payload is not None and not any(not task.done() for task in attachment.pending_sends):
                await bounded(attachment.send(payload))
        finally:
            await bounded(attachment.close(reason))

    def __repr__(self) -> str:
        return f"{type(self).__name__}(session_ids={sorted(self._sessions)})"

    async def create(
        self,
        session_id: str,
        *,
        incarnation: int,
        send: Callable[[dict[str, object]], Awaitable[None]],
        close: Callable[[str], Awaitable[None]],
        send_text: Callable[[str], Awaitable[None]] | None = None,
    ) -> DuplexSessionAttachmentCreated:
        async with self._lock:
            if session_id in self._sessions:
                raise ValueError(f"duplex attachment session already exists: {session_id}")
            token = ResumeToken.generate()
            generation = 1
            self._sessions[session_id] = _DuplexSessionAttachmentState(
                session_id=session_id,
                incarnation=incarnation,
                credential=DuplexResumeCredential.from_token(token),
                journal=DuplexEventJournal(
                    max_bytes=self._replay_max_bytes_per_session,
                    ttl_s=self._replay_ttl_s,
                    clock=self._clock,
                ),
                attachment=DuplexTransportAttachment(
                    generation=generation,
                    send=send,
                    close=close,
                    send_text=send_text,
                ),
                attachment_generation=generation,
            )
            return DuplexSessionAttachmentCreated(
                session_id=session_id,
                incarnation=incarnation,
                attachment_generation=generation,
                resume_token=token,
            )

    async def send_event(
        self,
        session_id: str,
        payload: Mapping[str, object],
        *,
        journal: bool = True,
    ) -> JournalEntry | None:
        """Sequence and dispatch one event to the current attachment.

        The per-session lock keeps wire order equal to journal order without
        serializing unrelated sessions. A detached session still records
        replayable events, but has no transport side effect.
        """
        async with self._lock:
            state = self._require(session_id)
        async with state.outbound_lock:
            async with self._lock:
                if self._sessions.get(session_id) is not state:
                    raise KeyError(f"unknown duplex attachment session: {session_id}")
                entry = state.journal.record(payload) if journal else None
                attachment = state.attachment
            if attachment is not None:
                await self._send_attachment(attachment, entry if entry is not None else dict(payload))
            return entry

    async def invalidate_replay(self, session_id: str) -> int:
        """Retire old output history at a context fence; stale resumes require resync."""
        async with self._lock:
            state = self._require(session_id)
        async with state.outbound_lock:
            async with self._lock:
                if self._sessions.get(session_id) is not state:
                    raise KeyError(session_id)
                boundary = state.journal.last_sequence
                state.journal.acknowledge(boundary)
                return boundary

    async def acknowledge(self, session_id: str, sequence: int) -> int:
        async with self._lock:
            return self._require(session_id).journal.acknowledge(sequence)

    async def detach(
        self,
        session_id: str,
        *,
        attachment_generation: int,
        on_grace_expired: Callable[[], Awaitable[None]] | None = None,
    ) -> bool:
        async with self._lock:
            state = self._sessions.get(session_id)
            if state is None or state.attachment_generation != attachment_generation:
                return False
            if state.attachment is not None:
                state.attachment.revoke()
            state.attachment = None
            if state.grace_task is not None:
                state.grace_task.cancel()
            state.grace_task = (
                asyncio.create_task(
                    self._run_disconnect_grace(
                        state,
                        attachment_generation=attachment_generation,
                        callback=on_grace_expired,
                    )
                )
                if on_grace_expired is not None
                else None
            )
            return True

    async def is_current_attachment(
        self, session_id: str, attachment_generation: int, *, include_revoked: bool = False
    ) -> bool:
        async with self._lock:
            state = self._sessions.get(session_id)
            return (
                state is not None
                and state.attachment is not None
                and (include_revoked or not state.attachment.revoked.is_set())
                and state.attachment_generation == attachment_generation
            )

    async def is_attached(self, session_id: str) -> bool:
        """Whether a live attachment still owns this session's transport."""
        async with self._lock:
            state = self._sessions.get(session_id)
            return state is not None and state.attachment is not None and not state.attachment.revoked.is_set()

    async def authenticate_resume(
        self,
        session_id: str,
        *,
        incarnation: int,
        resume_token: str,
        last_received_server_event_seq: int,
    ) -> None:
        """Validate transport credentials before any engine resume control."""
        async with self._lock:
            state = self._require(session_id)
            self._validate_resume_identity(
                state,
                incarnation=incarnation,
                resume_token=resume_token,
            )
            state.journal.replay_after(last_received_server_event_seq)

    async def resume(
        self,
        session_id: str,
        *,
        incarnation: int,
        resume_token: str,
        last_received_server_event_seq: int,
        send: Callable[[dict[str, object]], Awaitable[None]],
        close: Callable[[str], Awaitable[None]],
        activation_payload_factory: Callable[[ResumeToken, int], Mapping[str, object]] | None = None,
        send_text: Callable[[str], Awaitable[None]] | None = None,
    ) -> DuplexSessionResumeResult:
        async with self._lock:
            state = self._require(session_id)
        async with state.resume_lock:
            # Authenticate before disturbing the old connection. Revocation
            # wakes its send independently of the transport's cancellation.
            async with self._lock:
                if self._sessions.get(session_id) is not state:
                    raise KeyError(f"unknown duplex attachment session: {session_id}")
                used_recovery = self._validate_resume_identity(
                    state,
                    incarnation=incarnation,
                    resume_token=resume_token,
                )
                state.journal.replay_after(last_received_server_event_seq)
                accepted_token_digest = (
                    state.recovery_token_digest if used_recovery else bytes(state.credential.token_digest)
                )
                replaced = state.attachment
                if replaced is not None:
                    replaced.revoke()
            attachment = None
            try:
                async with state.outbound_lock:
                    async with self._lock:
                        if self._sessions.get(session_id) is not state:
                            raise KeyError(f"unknown duplex attachment session: {session_id}")
                        replay_entries = state.journal.replay_after(last_received_server_event_seq)
                        if state.grace_task is not None:
                            state.grace_task.cancel()
                            state.grace_task = None
                        state.recovery_token_digest = None
                        rotated_token = state.credential.rotate()
                        state.attachment_generation += 1
                        attachment_generation = state.attachment_generation
                        attachment = DuplexTransportAttachment(
                            generation=attachment_generation, send=send, close=close, send_text=send_text
                        )
                        state.attachment = attachment
                    if activation_payload_factory is not None:
                        # One deadline covers the entire activation/replay,
                        # not a fresh allowance for every historical event.
                        async def activate_and_replay() -> None:
                            activated = await self._send_attachment(
                                attachment, dict(activation_payload_factory(rotated_token, attachment_generation))
                            )
                            if not activated:
                                raise ConnectionError("duplex attachment revoked during resume")
                            for entry in replay_entries:
                                if not await self._send_attachment(attachment, entry):
                                    raise ConnectionError("duplex attachment revoked during resume")

                        await asyncio.wait_for(activate_and_replay(), timeout=self._transport_timeout_s)
                    return DuplexSessionResumeResult(
                        session_id=session_id,
                        incarnation=incarnation,
                        attachment_generation=attachment_generation,
                        resume_token=rotated_token,
                        replay_entries=replay_entries,
                        replaced_attachment=replaced,
                    )
            except BaseException:
                # No await in rollback: cancellation (including a repeated
                # cancel) cannot interrupt credential/generation convergence.
                if self._sessions.get(session_id) is state and state.attachment is (attachment or replaced):
                    state.attachment = None
                    state.recovery_token_digest = accepted_token_digest
                for retired in (attachment, replaced):
                    if retired is not None:
                        retired.revoke()
                        self._track_transport(self.retire_attachment(retired, "resume_failed"))
                raise

    async def close(self, session_id: str) -> DuplexTransportAttachment | None:
        async with self._lock:
            state = self._sessions.pop(session_id, None)
            if state is not None and state.grace_task is not None:
                state.grace_task.cancel()
            if state is not None and state.attachment is not None:
                state.attachment.revoke()
            return state.attachment if state is not None else None

    async def _run_disconnect_grace(
        self,
        state: _DuplexSessionAttachmentState,
        *,
        attachment_generation: int,
        callback: Callable[[], Awaitable[None]] | None,
    ) -> None:
        try:
            await asyncio.sleep(self._disconnect_grace_s)
        except asyncio.CancelledError:
            return
        async with self._lock:
            if (
                self._sessions.get(state.session_id) is not state
                or state.attachment is not None
                or state.attachment_generation != attachment_generation
            ):
                return
            state.grace_task = None
        if callback is not None:
            await callback()

    def _require(self, session_id: str) -> _DuplexSessionAttachmentState:
        state = self._sessions.get(session_id)
        if state is None:
            raise KeyError(f"unknown duplex attachment session: {session_id}")
        return state

    @staticmethod
    def _validate_resume_identity(
        state: _DuplexSessionAttachmentState,
        *,
        incarnation: int,
        resume_token: str,
    ) -> bool:
        if incarnation != state.incarnation:
            raise ValueError(f"duplex attachment incarnation mismatch: expected {state.incarnation}, got {incarnation}")
        if state.credential.verify(resume_token):
            return False
        recovery_digest = state.recovery_token_digest
        if (
            state.attachment is None
            and recovery_digest is not None
            and hmac.compare_digest(recovery_digest, DuplexResumeCredential._digest(resume_token))
        ):
            return True
        raise InvalidResumeTokenError(f"invalid resume token for duplex session {state.session_id}")


__all__ = [
    "DuplexEventJournal",
    "DuplexJournalGapError",
    "DuplexJournalOverflowError",
    "DuplexResumeCredential",
    "DuplexSessionAttachmentCreated",
    "DuplexSessionAttachmentRegistry",
    "DuplexSessionResumeResult",
    "DuplexTransportAttachment",
    "InvalidResumeTokenError",
    "JournalEntry",
    "ResumeToken",
]
