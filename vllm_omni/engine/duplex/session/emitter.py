# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Everything one duplex session sends to its client.

Emission is the hub of the session runner: near enough every path through the
runner ends in an event, which is why this is a collaborator rather than a
module of functions. It owns three things the rest of the runner should not
have to think about --- the Realtime projection state, the epoch filter that
drops output from a superseded turn, and the domain transitions an outbound
terminal event applies to the session before it is projected.

The one thing it cannot own is what happens *after* a terminal promotes
deferred overlap audio: that re-enters the mailbox of the runner, so the
runner injects it as ``promote_deferred_overlap``.
"""

from __future__ import annotations

from collections.abc import Callable

from vllm_omni.engine.duplex.config import DuplexSessionState
from vllm_omni.engine.duplex.events import (
    DOMAIN_TERMINAL_EVENTS,
    MODEL_OUTPUT_EVENTS,
    DuplexEvent,
    error_event,
)
from vllm_omni.engine.duplex.realtime_events import (
    RealtimeProjectionState,
    discard_pending_input_audio,
    project_internal_event,
)
from vllm_omni.engine.duplex.session import overlap_policy
from vllm_omni.engine.duplex.session.context import DuplexSessionContext


class SessionEmitter:
    """Outbound events for one session."""

    def __init__(
        self,
        ctx: DuplexSessionContext,
        *,
        promote_deferred_overlap: Callable[[dict[str, object], bool], None],
    ) -> None:
        self._ctx = ctx
        self._promote_deferred_overlap = promote_deferred_overlap
        self._projector: RealtimeProjectionState | None = ctx.session.projector

    # ------------------------------------------------------------------ #
    # Projection state                                                   #
    # ------------------------------------------------------------------ #

    @property
    def projector(self) -> RealtimeProjectionState | None:
        """The Realtime projection state, or ``None`` before anything needs it."""
        return self._projector

    @projector.setter
    def projector(self, value: RealtimeProjectionState | None) -> None:
        self._projector = value
        self._ctx.session.projector = value

    def require_projector(self) -> RealtimeProjectionState:
        """The projection state, created on first use.

        A session torn down before ``start()`` never builds one, so the lazy
        path here is reachable rather than defensive.
        """
        projector = self._projector
        if projector is None:
            session = self._ctx.session
            projector = RealtimeProjectionState(
                session_id=session.session_id,
                model=session.config.model,
                initial_session_update=True,
            )
            self.projector = projector
        return projector

    # ------------------------------------------------------------------ #
    # Sending                                                            #
    # ------------------------------------------------------------------ #

    def emit_events(self, events: list[DuplexEvent]) -> None:
        for event in events:
            self._ctx.manager.emit(self._ctx.session, event)

    def emit_error(
        self,
        code: str,
        message: str,
        *,
        event_id: object | None = None,
        retryable: bool | None = None,
    ) -> None:
        """Send one typed ``error`` event (``event_id`` is the client event it answers)."""
        extra = {} if retryable is None else {"retryable": retryable}
        self.emit_events([error_event(code, message, event_id=event_id, extra=extra)])

    def emit(self, payload: dict[str, object]) -> None:
        """Apply the domain effects of an internal event, then project it to typed events and send them.

        Only events with domain effects (response / cancel / close terminals) or
        Realtime projection state (response items, content parts) still travel as
        internal dictionaries; stateless events are constructed typed at the
        emit site.
        """
        accepted, deferred_overlap_payload = self._apply_outbound_session_event(payload)
        if not accepted:
            return
        self.emit_events(project_internal_event(self.require_projector(), payload))
        if deferred_overlap_payload is not None and not self._ctx.run.closing:
            precreate_response = self._ctx.model_state.deferred_precreate_response
            self._ctx.model_state.deferred_precreate_response = False
            self._promote_deferred_overlap(deferred_overlap_payload, precreate_response)

    # ------------------------------------------------------------------ #
    # Filtering and domain effects                                       #
    # ------------------------------------------------------------------ #

    def auto_responds(self) -> bool:
        """Whether the session answers committed input without a ``response.create``."""
        extra = getattr(self._ctx.session.config, "extra_body", None)
        if not isinstance(extra, dict):
            return False
        return extra.get("auto_response") is True or extra.get("full_duplex") is True

    def is_stale_model_output(self, payload: dict[str, object]) -> bool:
        """Whether ``payload`` belongs to a turn the session has already moved past."""
        session = self._ctx.session
        event_type = payload.get("type")
        if event_type in DOMAIN_TERMINAL_EVENTS:
            return False
        if event_type not in MODEL_OUTPUT_EVENTS:
            return False
        if self._ctx.run.closing and event_type != "response.listen":
            return True
        if session.state == DuplexSessionState.CLOSED and event_type != "response.listen":
            return True
        epoch = payload.get("epoch")
        return isinstance(epoch, int) and epoch != session.epoch

    def _apply_outbound_session_event(self, payload: dict[str, object]) -> tuple[bool, dict[str, object] | None]:
        """Apply domain transitions before an event is projected (moved from serving)."""
        session = self._ctx.session
        model_state = self._ctx.model_state
        payload_type = payload.get("type")
        is_terminal = payload_type in DOMAIN_TERMINAL_EVENTS
        if is_terminal:
            payload_epoch = payload.get("epoch")
            if isinstance(payload_epoch, int) and payload_epoch != session.epoch:
                return False, None
            if payload_type in {"response.done", "response.listen"} and (
                self._ctx.run.closing or session.state == DuplexSessionState.CLOSED
            ):
                return False, None
        elif self.is_stale_model_output(payload):
            return False, None

        if payload_type == "session.closed":
            self._ctx.run.close_reason = self._ctx.run.close_reason or str(payload.get("reason") or "closed")
            session.mark_closing()

        if not is_terminal:
            return True, None

        terminal_status = payload.get("status")
        terminal_status_details = payload.get("status_details")
        if terminal_status is None and isinstance(terminal_status_details, dict):
            terminal_status = terminal_status_details.get("type")
        response_terminal = payload_type == "response.done" or (
            payload_type == "response.listen" and session.active_response_id is not None
        )
        can_promote_overlap = response_terminal and terminal_status not in {"cancelled", "failed"}
        deferred_overlap_payload: dict[str, object] | None = None
        continuous_input_crosses_terminal = (
            can_promote_overlap
            and self.auto_responds()
            and model_state.input_since_commit
            and not model_state.deferred_response_create
        )
        if continuous_input_crosses_terminal:
            session.reset_overlap_speech()
            return True, None
        realtime_input_still_open = (
            can_promote_overlap and model_state.input_since_commit and not model_state.deferred_response_create
        )
        if realtime_input_still_open:
            session.reset_overlap_speech()
            return True, None
        if can_promote_overlap and session.overlap_speech_ms > 0:
            has_deferred_overlap = (
                model_state.audio_buffer.has_pending() or model_state.committed_audio_payload is not None
            )
            should_promote_overlap = (
                session.state == DuplexSessionState.OPEN
                and has_deferred_overlap
                and session.overlap_speech_ms > session.config.overlap_short_ack_ms
            )
            if should_promote_overlap:
                flushed_reserved_bytes = model_state.audio_buffer.pending_byte_count
                deferred_overlap_payload = model_state.audio_buffer.flush(
                    chunk_period_ms=session.capabilities.chunk_period_ms or 1000
                )
                if model_state.committed_audio_payload is not None:
                    if deferred_overlap_payload is not None:
                        deferred_overlap_payload = overlap_policy.merge_audio_payloads(
                            model_state.committed_audio_payload,
                            deferred_overlap_payload,
                        )
                    else:
                        deferred_overlap_payload = model_state.committed_audio_payload
                if self.auto_responds() and deferred_overlap_payload is not None:
                    deferred_overlap_payload = dict(deferred_overlap_payload)
                    deferred_overlap_payload["force_listen"] = False
                if deferred_overlap_payload is not None:
                    model_state.retain_committed_audio(
                        deferred_overlap_payload,
                        operation_id=model_state.committed_audio_operation_id,
                        reserved_bytes=flushed_reserved_bytes,
                    )
                model_state.input_since_commit = deferred_overlap_payload is not None
                # Realtime path: defer the promoted response to the next terminal.
                model_state.deferred_response_create = True
                model_state.deferred_precreate_response = False
                deferred_overlap_payload = None
            else:
                had_pending_overlap_audio = model_state.audio_buffer.has_pending()
                model_state.audio_buffer.clear()
                model_state.input_since_commit = False
                model_state.speech_since_commit = False
                if had_pending_overlap_audio and self._projector is not None:
                    self.emit_events(discard_pending_input_audio(self._projector, session.overlap_speech_ms))
                if payload_type in {"audio.cancelled", "input.cancelled", "session.closed"}:
                    session.release_input_bytes(model_state.clear_committed_audio())

        session.reset_overlap_speech()
        if (
            can_promote_overlap
            and model_state.deferred_response_create
            and model_state.committed_audio_payload is not None
        ):
            deferred_overlap_payload = model_state.committed_audio_payload
            model_state.deferred_response_create = False
            model_state.input_since_commit = False
            model_state.speech_since_commit = False
        return True, deferred_overlap_payload
