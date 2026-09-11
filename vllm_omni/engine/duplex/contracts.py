# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Model-neutral contracts for the experimental duplex engine plugin.

This module contains only immutable data transfer objects and narrow protocols.
Duplex control algorithms, session implementations, model policy, and Realtime
serving remain in sibling experimental modules.
"""

from __future__ import annotations

import base64
from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from vllm_omni.engine.duplex.session import DuplexSessionRuntimeManager, DuplexSessionRuntimeState

from vllm_omni.engine.duplex.messages import DuplexFence
from vllm_omni.engine.messages import EngineQueueMessage

DUPLEX_CONTRACT_VERSION = "duplex.capabilities.v1"


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
    scheduler_native_append: bool = False
    prompt_replay: bool = False
    contract_version: str = DUPLEX_CONTRACT_VERSION
    adapter_id: str = ""
    runtime_extension_id: str = ""
    stage_count: int | None = None

    def __post_init__(self) -> None:
        if self.prompt_replay and not self.scheduler_native_append:
            raise ValueError("duplex prompt replay requires scheduler-native append")

    def plugin_descriptor(self) -> DuplexPluginDescriptor | None:
        if self.contract_version != DUPLEX_CONTRACT_VERSION:
            raise ValueError(
                f"unsupported duplex contract version: {self.contract_version!r}; expected {DUPLEX_CONTRACT_VERSION!r}"
            )
        values = (self.adapter_id, self.runtime_extension_id, self.stage_count)
        if not any(value not in ("", None) for value in values):
            return None
        if not self.adapter_id or not self.runtime_extension_id or self.stage_count is None:
            raise ValueError("duplex plugin descriptor must declare adapter_id, runtime_extension_id, and stage_count")
        return DuplexPluginDescriptor(
            contract_version=self.contract_version,
            adapter_id=self.adapter_id,
            runtime_extension_id=self.runtime_extension_id,
            stage_count=self.stage_count,
        )


@dataclass(frozen=True)
class DuplexPluginDescriptor:
    """Versioned binding between serving, engine extension, and stage topology."""

    contract_version: str
    adapter_id: str
    runtime_extension_id: str
    stage_count: int

    def __post_init__(self) -> None:
        if self.contract_version != DUPLEX_CONTRACT_VERSION:
            raise ValueError(
                f"unsupported duplex contract version: {self.contract_version!r}; expected {DUPLEX_CONTRACT_VERSION!r}"
            )
        if not self.adapter_id or not self.runtime_extension_id:
            raise ValueError("duplex plugin descriptor requires adapter_id and runtime_extension_id")
        if self.stage_count <= 0:
            raise ValueError("duplex plugin descriptor stage_count must be positive")


@dataclass(frozen=True, slots=True)
class DuplexTraceEnvelope:
    """Bounded causal IDs propagated across the duplex engine boundary."""

    session_id: str
    fence: DuplexFence
    event: str
    control_id: str | None = None
    operation_id: str | None = None
    request_id: str | None = None
    sequence: int | None = None
    clock_origin: str = "engine_monotonic"

    def __post_init__(self) -> None:
        if not self.session_id or self.session_id != self.fence.session_id:
            raise ValueError("duplex trace session_id must match fence.session_id")
        if not self.event:
            raise ValueError("duplex trace event must be non-empty")
        if self.sequence is not None and self.sequence < 0:
            raise ValueError("duplex trace sequence must be non-negative")


@dataclass(frozen=True)
class DuplexAppendPlan:
    prompt: dict[str, Any]


@dataclass(frozen=True)
class DuplexContextOutput:
    unit_sequence: int
    data: Mapping[str, Any]


@dataclass(frozen=True)
class DuplexContextUnit:
    """A model-selected immutable input unit for context reconstruction."""

    unit_id: str
    prompt: Mapping[str, Any]

    def __post_init__(self):
        object.__setattr__(self, "prompt", MappingProxyType(deepcopy(dict(self.prompt))))


@dataclass(frozen=True)
class DuplexContextPlan:
    """Model policy; engine validates budgets and owns KV replacement."""

    units: tuple[DuplexContextUnit, ...]
    retained_unit_ids: tuple[str, ...]
    dropped_unit_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class DuplexOutputDecision:
    action: DuplexOutputAction
    metadata: Mapping[str, Any] = field(default_factory=dict)
    final_output_type: str = "text"
    # A direct model control action can end a turn without visiting TTS.
    ends_model_turn: bool = False

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
    scheduler_native_append: bool = False
    recovery_replay: bool = False
    session_config: Mapping[str, Any] = field(default_factory=dict)
    runtime_config: Mapping[str, Any] = field(default_factory=dict)
    trace: DuplexTraceEnvelope | None = None

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
    operation_id: str | None = None
    operation_fingerprint: bytes | None = None
    deadline_monotonic: float | None = None
    recovery_replay: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "prompt", MappingProxyType(dict(self.prompt)))


@dataclass(frozen=True)
class DuplexStageSubmissionResult:
    request_id: str
    stage_id: int
    replica_id: int
    metrics: Mapping[str, int | float | bool | str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))


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

    def supports_scheduler_native_append(self, stage_id: int = 0) -> bool: ...

    def ensure_request(self, context: DuplexStageRequestContext) -> None: ...

    async def submit(self, submission: DuplexStageSubmission) -> DuplexStageSubmissionResult: ...

    async def cleanup(self, request_ids: list[str], *, abort: bool = False) -> None: ...


class DuplexControlPlanePort(Protocol):
    @property
    def sessions(self) -> DuplexSessionRuntimeManager: ...

    def accepts(self, message: object) -> bool: ...

    def dispatch(self, message: object) -> None: ...

    async def shutdown(self) -> None: ...

    async def reap_expired(self, now: float | None = None) -> int: ...

    def defer_request_cleanups(self, session_ids: Iterable[str]) -> None: ...

    def prepare_replica_recovery(
        self,
        stage_id: int,
        request_ids: Iterable[str],
        *,
        uncertain_request_ids: Iterable[str] = (),
        allow_recovery: bool = True,
    ) -> tuple[set[str], set[str]]: ...

    def close_sessions_for_request_ids(
        self,
        request_ids: list[str],
        *,
        abort: bool = False,
        cleanup_in_progress: bool = False,
        reason: str = "request_cleanup",
    ) -> dict[str, list[str]]: ...

    def finalize_closed_sessions(self, session_ids: Iterable[str]) -> None: ...

    def session_for_identity(self, identity: DuplexRequestIdentity | None) -> DuplexSessionRuntimeState | None: ...

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


def duplex_resource_request_generation(
    request_id: str,
    fence: DuplexFence,
    role: str,
) -> int | None:
    """Parse one exact logical resource's physical request generation.

    Generation zero uses the canonical ``role`` request id.  Later physical
    requests append ``g<N>`` to that role.  Comparing against the canonical
    prefix first makes the parser reject a valid-looking request belonging to
    another session, incarnation, epoch, or resource role.
    """
    canonical = duplex_resource_request_id(fence, role)
    if request_id == canonical:
        return 0
    generation_prefix = f"{canonical}g"
    if not request_id.startswith(generation_prefix):
        return None
    raw_generation = request_id[len(generation_prefix) :]
    if not raw_generation.isascii() or not raw_generation.isdecimal():
        return None
    generation = int(raw_generation)
    if generation <= 0 or raw_generation != str(generation):
        return None
    return generation


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
    "DUPLEX_CONTRACT_VERSION",
    "DuplexAppendPlan",
    "DuplexControlPlanePort",
    "DuplexInputMode",
    "DuplexOutputAction",
    "DuplexOutputContext",
    "DuplexOutputDecision",
    "DuplexPluginDescriptor",
    "DuplexRequestIdentity",
    "DuplexRuntimeCapabilities",
    "DuplexRuntimeExtension",
    "DuplexStagePort",
    "DuplexStageRequestContext",
    "DuplexStageSubmission",
    "DuplexStageSubmissionResult",
    "DuplexTraceEnvelope",
    "SessionMode",
    "duplex_data_plane_request_info",
    "duplex_resource_request_belongs_to_session",
    "duplex_resource_request_generation",
    "duplex_resource_request_id",
]
