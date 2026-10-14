# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Queue envelopes exchanged between ``DuplexOmniEngine`` and ``DuplexOrchestrator``.

Sessions live inside the engine (``DuplexSessionRunner`` on the orchestrator
loop). The API layer only opens/closes/resumes/touches sessions through
correlated RPC and pushes ``DuplexCommand`` objects one-way. Open binds a
bounded in-process output buffer; closure notifications and errors without
a live session travel back as ``DuplexSessionEventMessage``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from vllm_omni.engine.messages import EngineQueueMessage

if TYPE_CHECKING:
    from vllm_omni.engine.duplex.commands import DuplexCommand
    from vllm_omni.engine.duplex.config import DuplexCapabilities, DuplexSessionConfig
    from vllm_omni.engine.duplex.delivery import DuplexOutputBuffer
    from vllm_omni.engine.duplex.events import DuplexEvent
    from vllm_omni.engine.duplex.fallback import DuplexFallbackRequest


class DuplexSessionError(RuntimeError):
    """A session control operation was rejected by the engine."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "internal_error",
        retryable: bool = False,
        session_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable
        self.session_id = session_id


class OpenDuplexSessionMessage(EngineQueueMessage, kw_only=True):
    type: Literal["open_duplex_session"] = "open_duplex_session"
    control_id: str
    session_id: str
    #: Already normalized by ``DuplexOmni``; the queue is in-process, so the object crosses as is.
    session_config: DuplexSessionConfig
    #: In-process, bounded output shared with the single consumer; no media in the engine-wide queue.
    output_buffer: DuplexOutputBuffer


class CloseDuplexSessionMessage(EngineQueueMessage, kw_only=True):
    type: Literal["close_duplex_session"] = "close_duplex_session"
    control_id: str
    session_id: str
    reason: str = "client_close"


class ResumeDuplexSessionMessage(EngineQueueMessage, kw_only=True):
    type: Literal["resume_duplex_session"] = "resume_duplex_session"
    control_id: str
    session_id: str
    expected_lease_generation: int


class TouchDuplexSessionMessage(EngineQueueMessage, kw_only=True):
    type: Literal["touch_duplex_session"] = "touch_duplex_session"
    control_id: str
    session_id: str
    activity: str
    #: For a detach: the lease generation the caller opened or resumed
    #: against. The manager refuses to detach a newer lease, so a connection
    #: giving up its own resume can never put the lease a later resume owns
    #: into disconnect grace. ``None`` detaches whatever generation is current.
    expected_lease_generation: int | None = None


class DuplexSessionCommandMessage(EngineQueueMessage, kw_only=True):
    """One-way session command; rejections come back as ``ErrorEvent``."""

    type: Literal["duplex_session_command"] = "duplex_session_command"
    session_id: str
    command: DuplexCommand


class DuplexControlResultMessage(EngineQueueMessage, kw_only=True):
    type: Literal["duplex_control_result"] = "duplex_control_result"
    control_id: str
    operation: str
    session_id: str
    ok: bool
    lease_generation: int | None = None
    capabilities: DuplexCapabilities | None = None
    #: The Realtime ``session`` object of the session (wire payload, already a dict).
    public_session: dict[str, object] | None = None
    error_code: str | None = None
    error_message: str | None = None
    error_retryable: bool = False

    @property
    def rpc_correlation_key(self) -> tuple[str, str]:
        return ("duplex", self.control_id)


class DuplexSessionEventMessage(EngineQueueMessage, kw_only=True):
    """A typed session event on the engine output queue."""

    type: Literal["duplex_session_event"] = "duplex_session_event"
    session_id: str
    event: DuplexEvent


class DuplexSessionFallbackRequestMessage(EngineQueueMessage, kw_only=True):
    """Internal request from the engine to the API-side chat service."""

    type: Literal["duplex_session_fallback_request"] = "duplex_session_fallback_request"
    session_id: str
    request: DuplexFallbackRequest


class DuplexSessionFallbackStartedMessage(EngineQueueMessage, kw_only=True):
    type: Literal["duplex_session_fallback_started"] = "duplex_session_fallback_started"
    session_id: str
    request_id: str
    response_id: str
    epoch: int


class DuplexSessionFallbackOutputMessage(EngineQueueMessage, kw_only=True):
    type: Literal["duplex_session_fallback_output"] = "duplex_session_fallback_output"
    session_id: str
    request_id: str
    response_id: str
    epoch: int
    output: dict[str, object]


class DuplexSessionFallbackFailedMessage(EngineQueueMessage, kw_only=True):
    type: Literal["duplex_session_fallback_failed"] = "duplex_session_fallback_failed"
    session_id: str
    request_id: str
    response_id: str
    epoch: int
    error: str
    error_code: str


class DuplexSessionFallbackCancelMessage(EngineQueueMessage, kw_only=True):
    type: Literal["duplex_session_fallback_cancel"] = "duplex_session_fallback_cancel"
    session_id: str
    request_id: str
    response_id: str
    epoch: int
    reason: str = "cancelled"


__all__ = [
    "CloseDuplexSessionMessage",
    "DuplexControlResultMessage",
    "DuplexSessionCommandMessage",
    "DuplexSessionError",
    "DuplexSessionEventMessage",
    "DuplexSessionFallbackCancelMessage",
    "DuplexSessionFallbackFailedMessage",
    "DuplexSessionFallbackOutputMessage",
    "DuplexSessionFallbackRequestMessage",
    "DuplexSessionFallbackStartedMessage",
    "OpenDuplexSessionMessage",
    "ResumeDuplexSessionMessage",
    "TouchDuplexSessionMessage",
]
