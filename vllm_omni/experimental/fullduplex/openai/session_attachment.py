# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import secrets
import time
from collections import deque
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
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


@dataclass(frozen=True)
class JournalEntry:
    sequence: int
    created_monotonic: float
    encoded_bytes: int
    payload: Mapping[str, object] = field(repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", MappingProxyType(dict(self.payload)))


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
        encoded_bytes = len(
            json.dumps(
                sequenced_payload,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
        )
        if self._retained_bytes + encoded_bytes > self._max_bytes:
            self._overflowed = True
            raise DuplexJournalOverflowError(
                f"duplex event journal byte limit exceeded: {self._retained_bytes + encoded_bytes} > {self._max_bytes}"
            )
        entry = JournalEntry(
            sequence=sequence,
            created_monotonic=self._clock(),
            encoded_bytes=encoded_bytes,
            payload=sequenced_payload,
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
    ) -> None:
        if replay_ttl_s <= 0:
            raise ValueError("replay_ttl_s must be positive")
        if replay_max_bytes_per_session <= 0:
            raise ValueError("replay_max_bytes_per_session must be positive")
        if disconnect_grace_s <= 0:
            raise ValueError("disconnect_grace_s must be positive")
        self._replay_ttl_s = replay_ttl_s
        self._replay_max_bytes_per_session = replay_max_bytes_per_session
        self._disconnect_grace_s = disconnect_grace_s
        self._clock = clock or time.monotonic
        self._sessions: dict[str, _DuplexSessionAttachmentState] = {}
        self._lock = asyncio.Lock()

    def __repr__(self) -> str:
        return f"{type(self).__name__}(session_ids={sorted(self._sessions)})"

    async def create(
        self,
        session_id: str,
        *,
        incarnation: int,
        send: Callable[[dict[str, object]], Awaitable[None]],
        close: Callable[[str], Awaitable[None]],
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
                wire_payload = dict(entry.payload) if entry is not None else dict(payload)
            if attachment is not None:
                await attachment.send(wire_payload)
            return entry

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

    async def is_current_attachment(self, session_id: str, attachment_generation: int) -> bool:
        async with self._lock:
            state = self._sessions.get(session_id)
            return (
                state is not None
                and state.attachment is not None
                and state.attachment_generation == attachment_generation
            )

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
    ) -> DuplexSessionResumeResult:
        async with self._lock:
            state = self._require(session_id)
        async with state.outbound_lock:
            async with self._lock:
                if self._sessions.get(session_id) is not state:
                    raise KeyError(f"unknown duplex attachment session: {session_id}")
                used_recovery = self._validate_resume_identity(
                    state,
                    incarnation=incarnation,
                    resume_token=resume_token,
                )
                replay_entries = state.journal.replay_after(last_received_server_event_seq)
                accepted_token_digest = (
                    state.recovery_token_digest if used_recovery else bytes(state.credential.token_digest)
                )
                if state.grace_task is not None:
                    state.grace_task.cancel()
                    state.grace_task = None
                state.recovery_token_digest = None
                rotated_token = state.credential.rotate()
                replaced = state.attachment
                state.attachment_generation += 1
                attachment_generation = state.attachment_generation
                state.attachment = DuplexTransportAttachment(
                    generation=attachment_generation,
                    send=send,
                    close=close,
                )
            if activation_payload_factory is not None:
                try:
                    await send(dict(activation_payload_factory(rotated_token, attachment_generation)))
                    for entry in replay_entries:
                        await send(dict(entry.payload))
                except Exception:
                    async with self._lock:
                        if (
                            self._sessions.get(session_id) is state
                            and state.attachment_generation == attachment_generation
                        ):
                            state.attachment = None
                            state.recovery_token_digest = accepted_token_digest
                    raise
            return DuplexSessionResumeResult(
                session_id=session_id,
                incarnation=incarnation,
                attachment_generation=attachment_generation,
                resume_token=rotated_token,
                replay_entries=replay_entries,
                replaced_attachment=replaced,
            )

    async def close(self, session_id: str) -> DuplexTransportAttachment | None:
        async with self._lock:
            state = self._sessions.pop(session_id, None)
            if state is not None and state.grace_task is not None:
                state.grace_task.cancel()
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
