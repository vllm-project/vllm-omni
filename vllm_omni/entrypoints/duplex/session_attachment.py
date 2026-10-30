# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import secrets
import time
from collections import deque
from collections.abc import Awaitable, Callable, Mapping
from contextlib import AbstractContextManager, nullcontext
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
    attachment_generation: int
    resume_token: ResumeToken = field(repr=False)


@dataclass(frozen=True)
class DuplexDetachedAttachment:
    """The transport a detach dropped, with the engine lease generation it was serving."""

    session_id: str
    attachment_generation: int
    lease_generation: int


@dataclass(frozen=True)
class DuplexSessionResumeResult:
    session_id: str
    attachment_generation: int
    resume_token: ResumeToken = field(repr=False)
    replay_entries: tuple[JournalEntry, ...] = ()
    replaced_attachment: DuplexTransportAttachment | None = None


@dataclass
class _DuplexSessionAttachmentState:
    session_id: str
    credential: DuplexResumeCredential
    journal: DuplexEventJournal
    attachment: DuplexTransportAttachment | None
    attachment_generation: int
    #: The engine lease generation the attached connection is serving: the one
    #: its open or resume produced, or one an abandoned resume handed to it.
    #: Read together with the attachment under the lock, so a detach always
    #: gives back the lease of the connection it drops, never a newer one.
    lease_generation: int = 0
    #: Resumes between ``begin_resume`` and ``end_resume``: their engine lease
    #: resume is in flight or landed, and they have not activated yet. While
    #: one is pending, the next activation is the connection that will serve,
    #: so an orphaned generation waits for it instead of going to the socket
    #: that is about to be replaced.
    pending_resumes: int = 0
    #: An engine lease generation nobody serves yet (its resume was abandoned,
    #: or its provisional attachment was rolled back), waiting for the next
    #: activation to absorb it. Generations only move forward.
    orphaned_lease_generation: int | None = None
    outbound_lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)
    recovery_token_digest: bytes | None = field(default=None, repr=False)


class DuplexSessionAttachmentRegistry:
    def __init__(
        self,
        *,
        replay_ttl_s: float,
        replay_max_bytes_per_session: int,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if replay_ttl_s <= 0:
            raise ValueError("replay_ttl_s must be positive")
        if replay_max_bytes_per_session <= 0:
            raise ValueError("replay_max_bytes_per_session must be positive")
        self._replay_ttl_s = replay_ttl_s
        self._replay_max_bytes_per_session = replay_max_bytes_per_session
        self._clock = clock or time.monotonic
        self._sessions: dict[str, _DuplexSessionAttachmentState] = {}
        self._lock = asyncio.Lock()

    def __repr__(self) -> str:
        return f"{type(self).__name__}(session_ids={sorted(self._sessions)})"

    async def create(
        self,
        session_id: str,
        *,
        send: Callable[[dict[str, object]], Awaitable[None]],
        close: Callable[[str], Awaitable[None]],
        lease_generation: int = 0,
    ) -> DuplexSessionAttachmentCreated:
        async with self._lock:
            if session_id in self._sessions:
                raise ValueError(f"duplex attachment session already exists: {session_id}")
            token = ResumeToken.generate()
            generation = 1
            self._sessions[session_id] = _DuplexSessionAttachmentState(
                session_id=session_id,
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
                lease_generation=lease_generation,
            )
            return DuplexSessionAttachmentCreated(
                session_id=session_id,
                attachment_generation=generation,
                resume_token=token,
            )

    async def send_event(
        self,
        session_id: str,
        payload: Mapping[str, object],
        *,
        journal: bool = True,
        on_accepted: Callable[[], None] | None = None,
        event_guard: Callable[[], AbstractContextManager[bool]] | None = None,
    ) -> JournalEntry | None:
        """Sequence and dispatch one event to the current attachment.

        The per-session lock keeps wire order equal to journal order without
        serializing unrelated sessions. A detached session still records
        replayable events, but has no transport side effect.

        ``on_accepted`` runs synchronously once the event is journaled, or
        after a successful transport send when journaling is disabled. A later
        send failure cannot undo acceptance into the journal. Detached,
        non-journaled events do not invoke it. The callback must not raise.
        """
        async with self._lock:
            state = self._require(session_id)
        async with state.outbound_lock:
            async with self._lock:
                if self._sessions.get(session_id) is not state:
                    raise KeyError(f"unknown duplex attachment session: {session_id}")
                # The producer may invalidate held audio while this coroutine
                # waits for an attachment lock. Commit order before sequencing;
                # release the cross-thread guard before awaiting network I/O.
                with event_guard() if event_guard is not None else nullcontext(True) as valid:
                    if not valid:
                        return None
                    entry = state.journal.record(payload) if journal else None
                    attachment = state.attachment
                    wire_payload = dict(entry.payload) if entry is not None else dict(payload)
            if entry is not None and on_accepted is not None:
                on_accepted()
            if attachment is not None:
                await attachment.send(wire_payload)
                if entry is None and on_accepted is not None:
                    on_accepted()
            return entry

    async def acknowledge(self, session_id: str, sequence: int) -> int:
        async with self._lock:
            return self._require(session_id).journal.acknowledge(sequence)

    async def detach(self, session_id: str, *, attachment_generation: int | None = None) -> bool:
        """Drop the current transport; the engine lease owns the disconnect grace.

        ``attachment_generation`` names the connection asking to detach, so a
        socket that already lost a takeover cannot detach the winner. ``None``
        means "whichever connection is attached right now" and is for callers
        that only know the session (the outbound pump, whose send just failed).

        Returns whether this call is the one that detached: an already-detached
        session answers ``False`` so a second disconnect signal for the same
        socket cannot restart the engine's disconnect grace window.
        """
        return await self.release_attachment(session_id, attachment_generation=attachment_generation) is not None

    async def release_attachment(
        self, session_id: str, *, attachment_generation: int | None = None
    ) -> DuplexDetachedAttachment | None:
        """``detach`` that also hands back the lease generation the dropped connection was serving.

        Captured under the same lock as the attachment: the shared session
        handle may already carry a newer generation from a resume that has
        not activated yet, and the engine detach that follows must be fenced
        on the dropped connection's own lease, not on that one.
        """
        async with self._lock:
            state = self._sessions.get(session_id)
            if state is None or state.attachment is None:
                return None
            if attachment_generation is not None and state.attachment_generation != attachment_generation:
                return None
            state.attachment = None
            return DuplexDetachedAttachment(
                session_id=session_id,
                attachment_generation=state.attachment_generation,
                lease_generation=state.lease_generation,
            )

    async def begin_resume(self, session_id: str) -> None:
        """Claim that a resume of ``session_id`` is about to land and activate.

        Held from before the engine lease resume until ``end_resume``. While
        any claim is open, a lease generation that loses its owner is parked
        for the next activation (see ``settle_lease_generation``) instead of
        being handed to the currently attached connection, which that
        activation is about to replace.
        """
        async with self._lock:
            self._require(session_id).pending_resumes += 1

    async def end_resume(self, session_id: str) -> int | None:
        """Release a ``begin_resume`` claim, whether or not the resume activated.

        Returns a lease generation the caller must detach itself: one that
        was parked for an activation that never came, with nobody attached
        to take it over. ``None`` when nothing is owed.
        """
        async with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                return None
            state.pending_resumes = max(0, state.pending_resumes - 1)
            return self._settle_orphan_locked(state)

    async def settle_lease_generation(self, session_id: str, lease_generation: int) -> int | None:
        """Find an owner for the engine lease generation of a resume its connection never served.

        The engine already bumped its lease to ``lease_generation``. The owner
        is, in order: the resume waiting to activate (it will serve the session
        and absorbs the generation when it activates); else the connection
        attached right now (it keeps serving, and its own disconnect must be
        able to detach this lease); else nobody, in which case the generation
        is returned and the caller puts it back into disconnect grace.
        Generations only move forward, so an older orphan never displaces a
        newer one an owner already holds.
        """
        async with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                # The session is gone from the registry: the engine session
                # closed with it, nothing is left to own.
                return None
            state.orphaned_lease_generation = max(state.orphaned_lease_generation or 0, lease_generation)
            return self._settle_orphan_locked(state)

    @staticmethod
    def _settle_orphan_locked(state: _DuplexSessionAttachmentState) -> int | None:
        orphan = state.orphaned_lease_generation
        if orphan is None:
            return None
        if state.pending_resumes > 0:
            # The next activation absorbs it (see ``resume``).
            return None
        state.orphaned_lease_generation = None
        if state.attachment is not None:
            state.lease_generation = max(state.lease_generation, orphan)
            return None
        return orphan

    async def has_attachment(self, session_id: str) -> bool:
        """Whether some connection is attached right now, whoever it is.

        A resume that failed to activate has no generation of its own, so it
        cannot ask ``is_current_attachment``. What it needs to know before
        rolling the engine lease back into its disconnect grace is only whether
        it would be rolling back somebody else's live attachment.
        """
        async with self._lock:
            state = self._sessions.get(session_id)
            return state is not None and state.attachment is not None

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
        resume_token: str,
        last_received_server_event_seq: int,
    ) -> None:
        """Validate transport credentials before any engine resume control."""
        async with self._lock:
            state = self._require(session_id)
            self._validate_resume_identity(state, resume_token=resume_token)
            state.journal.replay_after(last_received_server_event_seq)

    async def resume(
        self,
        session_id: str,
        *,
        resume_token: str,
        last_received_server_event_seq: int,
        send: Callable[[dict[str, object]], Awaitable[None]],
        close: Callable[[str], Awaitable[None]],
        activation_payload_factory: Callable[[ResumeToken, int], Mapping[str, object]] | None = None,
        lease_generation: int | None = None,
    ) -> DuplexSessionResumeResult:
        async with self._lock:
            state = self._require(session_id)
        async with state.outbound_lock:
            async with self._lock:
                if self._sessions.get(session_id) is not state:
                    raise KeyError(f"unknown duplex attachment session: {session_id}")
                used_recovery = self._validate_resume_identity(state, resume_token=resume_token)
                replay_entries = state.journal.replay_after(last_received_server_event_seq)
                accepted_token_digest = (
                    state.recovery_token_digest if used_recovery else bytes(state.credential.token_digest)
                )
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
                # The activating connection serves the newest lease the engine
                # holds: its own resume's generation, or a newer one whose
                # resume was abandoned while this one was pending.
                state.lease_generation = max(
                    state.lease_generation,
                    lease_generation if lease_generation is not None else 0,
                    state.orphaned_lease_generation or 0,
                )
                state.orphaned_lease_generation = None
            if activation_payload_factory is not None:
                try:
                    await send(dict(activation_payload_factory(rotated_token, attachment_generation)))
                    for entry in replay_entries:
                        await send(dict(entry.payload))
                except (Exception, asyncio.CancelledError):
                    # CancelledError derives from BaseException, so an
                    # ``except Exception`` here would let a cancellation during
                    # activation or replay skip the rollback: the dead
                    # attachment would stay current while the token it rotated
                    # past is gone, leaving the session attached to an unusable
                    # transport with no way back. Both failures take the same
                    # path deliberately, so they cannot drift apart.
                    await self._rollback_resume(
                        session_id,
                        state=state,
                        attachment_generation=attachment_generation,
                        accepted_token_digest=accepted_token_digest,
                    )
                    raise
            return DuplexSessionResumeResult(
                session_id=session_id,
                attachment_generation=attachment_generation,
                resume_token=rotated_token,
                replay_entries=replay_entries,
                replaced_attachment=replaced,
            )

    async def _rollback_resume(
        self,
        session_id: str,
        *,
        state: _DuplexSessionAttachmentState,
        attachment_generation: int,
        accepted_token_digest: bytes | None,
    ) -> None:
        """Undo a resume whose activation never reached the client.

        Generation-checked: a later resume that already took over must keep its
        attachment. Restoring ``recovery_token_digest`` is what lets the caller
        retry with the token it presented, which ``resume`` had rotated past.

        The lease the provisional attachment was serving loses its owner here.
        That is not only its own resume's generation: a resume abandoned while
        this one was activating may have handed it a newer one. It is parked as
        the orphan, so the caller's settlement (which only knows its own
        generation) and ``end_resume`` give back the newest lease, never a
        stale one the engine would refuse.
        """
        async with self._lock:
            if self._sessions.get(session_id) is state and state.attachment_generation == attachment_generation:
                state.attachment = None
                state.recovery_token_digest = accepted_token_digest
                state.orphaned_lease_generation = max(state.orphaned_lease_generation or 0, state.lease_generation)

    async def close(self, session_id: str) -> DuplexTransportAttachment | None:
        async with self._lock:
            state = self._sessions.pop(session_id, None)
            return state.attachment if state is not None else None

    def _require(self, session_id: str) -> _DuplexSessionAttachmentState:
        state = self._sessions.get(session_id)
        if state is None:
            raise KeyError(f"unknown duplex attachment session: {session_id}")
        return state

    @staticmethod
    def _validate_resume_identity(state: _DuplexSessionAttachmentState, *, resume_token: str) -> bool:
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
