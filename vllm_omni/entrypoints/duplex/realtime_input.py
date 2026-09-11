# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Thin OpenAI Realtime wire envelope for one websocket connection.

Everything that needed session state now lives engine-side
(``vllm_omni.engine.duplex.realtime_commands`` for the command mapping,
``vllm_omni.engine.duplex.realtime_events`` for the projection state). What is
left here is the per-connection handshake policy (query-param defaults,
autostart / resume-only rules, ``session.resume`` parsing), the wire defaults
used to translate appends, and error rendering through the typed
:class:`~vllm_omni.engine.duplex.events.ErrorEvent`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from vllm_omni.engine.duplex.commands import DuplexCommand, DuplexCommandError
from vllm_omni.engine.duplex.events import error_event
from vllm_omni.engine.duplex.realtime_commands import (
    RealtimeInputDefaults,
    translate_realtime_command,
)

if TYPE_CHECKING:
    from starlette.websockets import WebSocket

__all__ = ["RealtimeEnvelope", "RealtimeHandshake", "ResumeRequest", "parse_resume_request"]

_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}

#: Client events the envelope handles itself (never translated into commands).
ENVELOPE_EVENT_TYPES = frozenset({"session.resume", "session.event_ack", "conversation.item.retrieve"})


@dataclass(frozen=True, slots=True)
class RealtimeHandshake:
    """What the first client message asks for."""

    #: ``"open"`` (session object in ``session_payload``) or ``"resume"`` (resume payload).
    kind: str
    session_payload: dict[str, object] = field(default_factory=dict)
    resume_payload: dict[str, object] = field(default_factory=dict)
    #: A client event that arrived before the session existed and must run right after opening.
    pending_command_payload: dict[str, object] | None = None


@dataclass(frozen=True, slots=True)
class ResumeRequest:
    """A validated ``session.resume`` client event."""

    session_id: str
    resume_token: str
    last_received_server_event_seq: int


def parse_resume_request(event: Mapping[str, object]) -> ResumeRequest | None:
    """Validate the shape of ``session.resume``; ``None`` when any field is missing or malformed."""
    session_id = event.get("session_id")
    resume_token = event.get("resume_token")
    last_received = event.get("last_received_server_event_seq", 0)
    if (
        not isinstance(session_id, str)
        or not session_id
        or not isinstance(resume_token, str)
        or not resume_token
        or not isinstance(last_received, int)
        or last_received < 0
    ):
        return None
    return ResumeRequest(
        session_id=session_id,
        resume_token=resume_token,
        last_received_server_event_seq=last_received,
    )


@dataclass
class RealtimeEnvelope:
    """Per-connection Realtime wire policy (no session state)."""

    default_model: str | None = None
    #: ``?resume=1`` / ``?autostart=0``: the first message must be ``session.resume``/``session.update``.
    resume_only: bool = False
    opened: bool = False
    autostarted_default_session: bool = False
    defaults: RealtimeInputDefaults = field(default_factory=RealtimeInputDefaults)

    @classmethod
    def from_query_params(cls, query_params: Mapping[str, str] | WebSocket) -> RealtimeEnvelope:
        if hasattr(query_params, "query_params"):
            query_params = query_params.query_params
        getter = query_params.get if hasattr(query_params, "get") else None
        resume_only = getter("resume") if getter is not None else None
        autostart = getter("autostart") if getter is not None else None
        model = getter("model") if getter is not None else None
        return cls(
            default_model=model if isinstance(model, str) and model else None,
            resume_only=(
                str(resume_only).strip().lower() in _TRUE_VALUES or str(autostart).strip().lower() in _FALSE_VALUES
            ),
        )

    # ---- handshake ----

    def default_session_payload(self) -> dict[str, object]:
        return {"model": self.default_model}

    def initial_open_payload(self) -> dict[str, object] | None:
        """Session object to open with *before* any client message (``?model=`` autostart), else None."""
        if self.opened or self.resume_only or self.autostarted_default_session or not self.default_model:
            return None
        self.opened = True
        self.autostarted_default_session = True
        return self.default_session_payload()

    def first_message(self, payload: Mapping[str, object]) -> RealtimeHandshake:
        """Classify the first client message (call only while ``not opened``).

        ``session.update`` opens with its session object; ``session.resume``
        resumes; any other event autostarts the default session and is then
        treated as a command (``pending_command_payload``).
        """
        event_type = payload.get("type")
        self.opened = True
        if event_type == "session.resume":
            return RealtimeHandshake(kind="resume", resume_payload=dict(payload))
        if event_type == "session.update":
            session = payload.get("session")
            session_payload = dict(session) if isinstance(session, dict) else dict(payload)
            session_payload.pop("type", None)
            self.note_session_payload(session_payload)
            return RealtimeHandshake(kind="open", session_payload=session_payload)
        self.autostarted_default_session = True
        return RealtimeHandshake(
            kind="open",
            session_payload=self.default_session_payload(),
            pending_command_payload=dict(payload),
        )

    def note_session_payload(self, session_payload: Mapping[str, object]) -> None:
        """Track wire defaults declared by a session object (open or session.update)."""
        self.defaults = self.defaults.with_session_payload(session_payload)

    # ---- commands ----

    def is_envelope_event(self, payload: Mapping[str, object]) -> bool:
        return payload.get("type") in ENVELOPE_EVENT_TYPES

    def translate(self, payload: Mapping[str, object]) -> DuplexCommand:
        """Wire event -> command (raises :class:`DuplexCommandError`)."""
        if payload.get("type") == "session.update":
            session = payload.get("session")
            if isinstance(session, dict):
                self.note_session_payload(session)
        return translate_realtime_command(payload, defaults=self.defaults)

    # ---- outbound helpers ----

    @staticmethod
    def error_payload(
        code: str,
        message: str,
        *,
        event_id: object | None = None,
        param: object | None = None,
    ) -> dict[str, object]:
        """Wire JSON of a transport-level error (derived from the typed ``ErrorEvent``)."""
        return error_event(code, message, event_id=event_id, param=param).to_realtime()

    @staticmethod
    def command_error_payload(exc: DuplexCommandError) -> dict[str, object]:
        return error_event(exc.code, str(exc), event_id=exc.event_id).to_realtime()
