# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Queue contracts owned by the experimental duplex control plane."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from vllm_omni.engine.messages import EngineQueueMessage


@dataclass(frozen=True, slots=True)
class DuplexFence:
    """Identity fence carried by experimental duplex control messages."""

    session_id: str
    epoch: int = 0
    turn_id: int = 0
    response_seq: int = 0
    incarnation: int = 0


class OpenDuplexSessionMessage(EngineQueueMessage, kw_only=True):
    type: Literal["open_duplex_session"] = "open_duplex_session"
    control_id: str
    fence: DuplexFence
    session_id: str
    session_mode: str = "duplex"
    capabilities: dict[str, object]
    session_config: dict[str, object] | None = None
    runtime_config: dict[str, object] | None = None


class AppendDuplexInputMessage(EngineQueueMessage, kw_only=True):
    type: Literal["append_duplex_input"] = "append_duplex_input"
    control_id: str
    operation_id: str | None = None
    fence: DuplexFence
    session_id: str
    expected_epoch: int | None = None
    mode: str
    payload: object
    final: bool = False


class SignalDuplexTurnMessage(EngineQueueMessage, kw_only=True):
    type: Literal["signal_duplex_turn"] = "signal_duplex_turn"
    control_id: str
    fence: DuplexFence
    session_id: str
    event: str
    next_fence: DuplexFence | None = None
    session_config: dict[str, object] | None = None
    runtime_config: dict[str, object] | None = None


class CloseDuplexSessionMessage(EngineQueueMessage, kw_only=True):
    type: Literal["close_duplex_session"] = "close_duplex_session"
    control_id: str
    fence: DuplexFence
    session_id: str
    reason: str = "client_close"


class TouchDuplexSessionMessage(EngineQueueMessage, kw_only=True):
    type: Literal["touch_duplex_session"] = "touch_duplex_session"
    control_id: str
    fence: DuplexFence
    session_id: str
    activity: str


class ResumeDuplexSessionMessage(EngineQueueMessage, kw_only=True):
    type: Literal["resume_duplex_session"] = "resume_duplex_session"
    control_id: str
    fence: DuplexFence
    session_id: str
    expected_lease_generation: int


class DuplexSessionLifecycleMessage(EngineQueueMessage, kw_only=True):
    type: Literal["duplex_session_lifecycle"] = "duplex_session_lifecycle"
    fence: DuplexFence
    session_id: str
    event: str
    reason: str
    lease_generation: int
    submitted_request_ids: list[str]
    reserved_request_ids: list[str]


class DuplexControlError(EngineQueueMessage, kw_only=True):
    code: str
    message: str
    retryable: bool = False


class DuplexControlResultMessage(EngineQueueMessage, kw_only=True):
    type: Literal["duplex_control_result"] = "duplex_control_result"
    control_id: str
    fence: DuplexFence
    operation: str
    session_id: str
    ok: bool
    stage_results: list[dict[str, object]]
    unsupported_count: int = 0
    error_count: int = 0
    error: DuplexControlError | None = None
    accepted_fence: DuplexFence | None = None
    lease_generation: int | None = None

    @property
    def rpc_correlation_key(self) -> tuple[str, str]:
        return ("duplex", self.control_id)


__all__ = [
    "AppendDuplexInputMessage",
    "CloseDuplexSessionMessage",
    "DuplexControlError",
    "DuplexControlResultMessage",
    "DuplexFence",
    "DuplexSessionLifecycleMessage",
    "OpenDuplexSessionMessage",
    "ResumeDuplexSessionMessage",
    "SignalDuplexTurnMessage",
    "TouchDuplexSessionMessage",
]
