# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Model-neutral value types shared by the duplex engine components.

Immutable DTOs plus the ``DuplexStagePort`` base class that ``DuplexOrchestrator``
implements for the session manager/runner.
"""

from __future__ import annotations

import base64
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType


@dataclass(frozen=True, slots=True)
class DuplexFence:
    """Engine-internal session identity used for stage request ids and stale filtering."""

    session_id: str
    epoch: int = 0
    turn_id: int = 0


class DuplexOutputAction(str, Enum):
    DIRECT_RESPONSE = "direct_response"


@dataclass(frozen=True)
class DuplexAppendPlan:
    prompt: dict[str, object]


@dataclass(frozen=True)
class DuplexOutputDecision:
    action: DuplexOutputAction
    metadata: Mapping[str, object] = field(default_factory=dict)
    final_output_type: str = "text"

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


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
    session_config: Mapping[str, object] = field(default_factory=dict)
    runtime_config: Mapping[str, object] = field(default_factory=dict)

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
    prompt: Mapping[str, object]
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
    segment_output_metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "segment_token_ids", tuple(self.segment_token_ids))
        object.__setattr__(
            self,
            "segment_output_metadata",
            MappingProxyType(dict(self.segment_output_metadata)),
        )


class DuplexStagePort(ABC):
    """Narrow stage-management surface the session runner/manager use (implemented by DuplexOrchestrator)."""

    @property
    @abstractmethod
    def stage_count(self) -> int: ...

    @abstractmethod
    def sampling_defaults(self) -> tuple[object, ...]: ...

    @abstractmethod
    def ensure_request(self, context: DuplexStageRequestContext) -> None: ...

    @abstractmethod
    async def submit(self, submission: DuplexStageSubmission) -> DuplexStageSubmissionResult: ...

    @abstractmethod
    async def cleanup(self, request_ids: list[str], *, abort: bool = False) -> None: ...

    @abstractmethod
    async def abort_requests(self, request_ids: list[str]) -> None: ...


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
    return f"duplex-s.{encoded_session_id}.e.{fence.epoch}.r.{role}"


def duplex_resource_request_belongs_to_session(request_id: str, session_id: str) -> bool:
    """Return whether a current-format resource request belongs to a session."""
    parts = request_id.split(".")
    if len(parts) != 6 or parts[0] != "duplex-s" or parts[2] != "e" or parts[4] != "r":
        return False
    try:
        int(parts[3])
    except ValueError:
        return False
    role = parts[5]
    if not role or any(
        character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in role
    ):
        return False
    encoded_session_id = base64.urlsafe_b64encode(session_id.encode("utf-8")).decode("ascii").rstrip("=")
    return parts[1] == encoded_session_id


__all__ = [
    "DuplexFence",
    "DuplexAppendPlan",
    "DuplexOutputAction",
    "DuplexOutputContext",
    "DuplexOutputDecision",
    "DuplexRequestIdentity",
    "DuplexStagePort",
    "DuplexStageRequestContext",
    "DuplexStageSubmission",
    "DuplexStageSubmissionResult",
    "duplex_data_plane_request_info",
    "duplex_resource_request_belongs_to_session",
    "duplex_resource_request_id",
]
