# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Model-neutral value types shared by the duplex engine components.

Immutable DTOs plus the ``DuplexStagePort`` base class that ``DuplexOrchestrator``
implements for the session manager/runner.
"""

from __future__ import annotations

import base64
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType

import regex as re


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
    # True: resume/update an existing stage0 id. False: open a new ephemeral id.
    # Distinct from DuplexCapabilities.supports_core_resumable_request.
    resumable: bool = True

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


def duplex_ephemeral_stage_request_id(fence: DuplexFence, *, stage_id: int) -> str:
    """Turn-scoped Stage request id for non-resumable (ephemeral) duplex models."""
    return duplex_resource_request_id(fence, f"stage{stage_id}-turn{fence.turn_id}")


_EPHEMERAL_TURN_IN_REQUEST_ID = re.compile(r"\.r\.stage\d+-turn(\d+)$")


def duplex_turn_id_from_request_id(request_id: str | None) -> int | None:
    """Parse ephemeral ``…r.stage{N}-turn{T}`` ids."""
    if not isinstance(request_id, str):
        return None
    match = _EPHEMERAL_TURN_IN_REQUEST_ID.search(request_id)
    return int(match.group(1)) if match else None


def duplex_same_turn_request_ids(request_id: str, candidate_ids: Iterable[str]) -> list[str]:
    """Other ephemeral stage ids from the same session, epoch, and turn.

    ``request_id`` itself is not included. Non-ephemeral ids are ignored.
    """
    turn_id = duplex_turn_id_from_request_id(request_id)
    if turn_id is None or ".r." not in request_id:
        return []
    prefix = request_id.rsplit(".r.", 1)[0] + ".r."
    return [
        candidate
        for candidate in candidate_ids
        if candidate != request_id
        and isinstance(candidate, str)
        and candidate.startswith(prefix)
        and duplex_turn_id_from_request_id(candidate) == turn_id
    ]


def _duplex_resource_request_fields(request_id: str | None) -> tuple[str, str, str] | None:
    """Split ``duplex-s.<b64url>.e.<epoch>.r.<role>`` or return None."""
    if not isinstance(request_id, str):
        return None
    parts = request_id.split(".")
    if len(parts) != 6 or parts[0] != "duplex-s" or parts[2] != "e" or parts[4] != "r":
        return None
    return parts[1], parts[3], parts[5]


def duplex_session_id_from_request_id(request_id: str | None) -> str | None:
    """Decode the session id from ``duplex-s.<b64url>.e.<epoch>.r.<role>``."""
    fields = _duplex_resource_request_fields(request_id)
    if fields is None:
        return None
    encoded, _, _ = fields
    pad = "=" * (-len(encoded) % 4)
    try:
        return base64.urlsafe_b64decode(encoded + pad).decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return None


def duplex_resource_request_belongs_to_session(request_id: str, session_id: str) -> bool:
    """Return whether a current-format resource request belongs to a session."""
    fields = _duplex_resource_request_fields(request_id)
    if fields is None:
        return False
    encoded, epoch, role = fields
    try:
        int(epoch)
    except ValueError:
        return False
    if not role or any(
        character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in role
    ):
        return False
    encoded_session_id = base64.urlsafe_b64encode(session_id.encode("utf-8")).decode("ascii").rstrip("=")
    return encoded == encoded_session_id


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
    "duplex_ephemeral_stage_request_id",
    "duplex_turn_id_from_request_id",
    "duplex_resource_request_belongs_to_session",
    "duplex_resource_request_id",
    "duplex_same_turn_request_ids",
    "duplex_session_id_from_request_id",
]
