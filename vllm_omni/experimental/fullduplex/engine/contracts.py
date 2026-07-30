# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Model-neutral contracts for the experimental duplex engine plugin.

This module contains only immutable data transfer objects and narrow protocols.
Duplex control algorithms, session implementations, model policy, and Realtime
serving remain in sibling experimental modules.
"""

from __future__ import annotations

import base64
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Protocol

from vllm_omni.engine.messages import EngineQueueMessage
from vllm_omni.experimental.fullduplex.engine.messages import DuplexFence


class SessionMode(str, Enum):
    TURN = "turn"
    DUPLEX = "duplex"


class DuplexInputMode(str, Enum):
    APPEND_TOKENS = "append_tokens"
    APPEND_AUDIO_CHUNK = "append_audio_chunk"
    REPLACE_LATEST_CHUNK = "replace_latest_chunk"
    REENCODE_CONTEXT = "reencode_context"
    ROLLBACK_TO_CHECKPOINT = "rollback_to_checkpoint"
    TURN_COMMIT_ONLY = "turn_commit_only"


class DuplexOutputAction(str, Enum):
    DIRECT_RESPONSE = "direct_response"


@dataclass
class DuplexRuntimeCapabilities:
    input_modes: set[DuplexInputMode] = field(default_factory=lambda: {DuplexInputMode.TURN_COMMIT_ONLY})
    implementation_level: str = "serving_session_adapter"


@dataclass(frozen=True)
class DuplexAppendPlan:
    prompt: dict[str, Any]


@dataclass(frozen=True)
class DuplexOutputDecision:
    action: DuplexOutputAction
    metadata: Mapping[str, Any] = field(default_factory=dict)
    final_output_type: str = "text"

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


class DuplexRuntimeExtension(Protocol):
    """Pure model policy invoked by the experimental duplex control plane."""

    def configure_sampling_params(
        self,
        *,
        runtime_config: dict[str, Any],
        defaults: tuple[object, ...],
    ) -> tuple[object, ...]: ...

    def plan_append(
        self,
        *,
        request_id: str,
        fence: DuplexFence,
        session_config: dict[str, Any],
        runtime_config: dict[str, Any],
        seq: int,
        turn_seq: int,
        mode: DuplexInputMode,
        payload: object,
        final: bool,
        sampling_params: object,
    ) -> DuplexAppendPlan: ...

    def decide_output(
        self,
        *,
        stage_id: int,
        final_stage_id: int,
        segment_finished: bool,
        segment_token_ids: tuple[int, ...],
        segment_output_metadata: dict[str, Any],
        output: object,
    ) -> DuplexOutputDecision | None: ...


@dataclass(frozen=True)
class DuplexRequestIdentity:
    session_id: str
    fence: DuplexFence


@dataclass(frozen=True)
class DuplexStageRequestContext:
    request_id: str
    session_id: str
    fence: DuplexFence
    stage_id: int
    final_stage_id: int
    config_generation: int
    sampling_params: tuple[object, ...]
    session_config: Mapping[str, Any] = field(default_factory=dict)
    runtime_config: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "sampling_params", tuple(self.sampling_params))
        object.__setattr__(self, "session_config", MappingProxyType(dict(self.session_config)))
        object.__setattr__(self, "runtime_config", MappingProxyType(dict(self.runtime_config)))

    @property
    def stage_sampling_params(self) -> object:
        return self.sampling_params[self.stage_id]


@dataclass(frozen=True)
class DuplexStageSubmission:
    context: DuplexStageRequestContext
    prompt: Mapping[str, Any]
    already_submitted: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "prompt", MappingProxyType(dict(self.prompt)))


@dataclass(frozen=True)
class DuplexStageSubmissionResult:
    request_id: str
    stage_id: int
    replica_id: int


@dataclass(frozen=True)
class DuplexOutputContext:
    identity: DuplexRequestIdentity
    final_stage_id: int
    segment_finished: bool
    segment_token_ids: tuple[int, ...] = ()
    segment_output_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "segment_token_ids", tuple(self.segment_token_ids))
        object.__setattr__(
            self,
            "segment_output_metadata",
            MappingProxyType(dict(self.segment_output_metadata)),
        )


class DuplexStagePort(Protocol):
    @property
    def stage_count(self) -> int: ...

    def sampling_defaults(self) -> tuple[object, ...]: ...

    def ensure_request(self, context: DuplexStageRequestContext) -> None: ...

    async def submit(self, submission: DuplexStageSubmission) -> DuplexStageSubmissionResult: ...

    async def cleanup(self, request_ids: list[str], *, abort: bool = False) -> None: ...


class DuplexControlPlanePort(Protocol):
    @property
    def sessions(self) -> object: ...

    def accepts(self, message: object) -> bool: ...

    def dispatch(self, message: object) -> None: ...

    async def shutdown(self) -> None: ...

    def close_sessions_for_request_ids(self, request_ids: list[str]) -> dict[str, list[str]]: ...

    def finalize_closed_sessions(self, session_ids: Iterable[str]) -> None: ...

    def session_for_identity(self, identity: DuplexRequestIdentity | None) -> object | None: ...

    def decide_output(
        self,
        stage_id: int,
        output: object,
        context: DuplexOutputContext | None,
    ) -> DuplexOutputDecision | None: ...


class CorrelatedRpcTransport(Protocol):
    def execute(
        self,
        key: tuple[str, str],
        message: EngineQueueMessage,
        *,
        timeout: float | None,
        timeout_message: str,
        block_on_submit: bool = False,
    ) -> EngineQueueMessage: ...


def duplex_data_plane_request_info(result: dict[str, object]) -> tuple[str | None, int | None]:
    stage_results = result.get("stage_results")
    if not isinstance(stage_results, list):
        return None, None
    for item in stage_results:
        if not isinstance(item, dict):
            continue
        inner = item.get("result")
        if not isinstance(inner, dict) or inner.get("data_plane_append") is not True:
            continue
        request_id = inner.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            continue
        response_stage_id = inner.get("response_stage_id")
        return request_id, response_stage_id if isinstance(response_stage_id, int) else None
    return None, None


def duplex_resource_request_id(fence: DuplexFence, role: str) -> str:
    if not role or any(
        character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in role
    ):
        raise ValueError(f"invalid duplex resource role: {role!r}")
    encoded_session_id = base64.urlsafe_b64encode(fence.session_id.encode("utf-8")).decode("ascii").rstrip("=")
    return f"duplex-s.{encoded_session_id}.i.{fence.incarnation}.e.{fence.epoch}.r.{role}"


def duplex_resource_request_belongs_to_session(request_id: str, session_id: str) -> bool:
    """Return whether a current-format resource request belongs to a session."""
    parts = request_id.split(".")
    if len(parts) != 8 or parts[0] != "duplex-s" or parts[2] != "i" or parts[4] != "e" or parts[6] != "r":
        return False
    try:
        int(parts[3])
        int(parts[5])
    except ValueError:
        return False
    role = parts[7]
    if not role or any(
        character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in role
    ):
        return False
    encoded_session_id = base64.urlsafe_b64encode(session_id.encode("utf-8")).decode("ascii").rstrip("=")
    return parts[1] == encoded_session_id


__all__ = [
    "CorrelatedRpcTransport",
    "DuplexAppendPlan",
    "DuplexControlPlanePort",
    "DuplexInputMode",
    "DuplexOutputAction",
    "DuplexOutputContext",
    "DuplexOutputDecision",
    "DuplexRequestIdentity",
    "DuplexRuntimeCapabilities",
    "DuplexRuntimeExtension",
    "DuplexStagePort",
    "DuplexStageRequestContext",
    "DuplexStageSubmission",
    "DuplexStageSubmissionResult",
    "SessionMode",
    "duplex_data_plane_request_info",
    "duplex_resource_request_belongs_to_session",
    "duplex_resource_request_id",
]
