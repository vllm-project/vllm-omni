# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import struct
import time
from collections import OrderedDict
from collections.abc import Callable, Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from hashlib import sha256
from types import MappingProxyType
from typing import Any

from vllm_omni.engine.duplex.contracts import (
    DuplexInputMode,
    DuplexRuntimeCapabilities,
)
from vllm_omni.engine.duplex.lease import (
    DuplexLeaseActivity,
    DuplexLeaseConfig,
    DuplexLeaseState,
    DuplexSessionExpiry,
)
from vllm_omni.engine.duplex.messages import DuplexFence

_APPEND_TOMBSTONE_LIMIT = 65536


def _default_capabilities() -> DuplexRuntimeCapabilities:
    return DuplexRuntimeCapabilities()


class DuplexFenceMismatchError(RuntimeError):
    def __init__(self, expected: DuplexFence, actual: DuplexFence) -> None:
        super().__init__(f"duplex fence mismatch: expected {expected!r}, got {actual!r}")
        self.expected = expected
        self.actual = actual


@dataclass
class DuplexStageBinding:
    request_id: str
    fence: DuplexFence


@dataclass
class DuplexRequestResource:
    stage_id: int
    request_id: str
    fence: DuplexFence
    submitted: bool = False


@dataclass
class DuplexInputAppend:
    seq: int
    turn_seq: int
    turn_id: int


@dataclass(frozen=True)
class DuplexAppendReservation:
    fence: DuplexFence
    mode: DuplexInputMode
    base_fence: DuplexFence
    base_input_seq: int
    base_input_turn_seq: int
    base_append_turn_key: tuple[int, int, int] | None
    update: DuplexInputAppend


@dataclass(frozen=True)
class DuplexCompletedAppend:
    fence: DuplexFence
    mode: DuplexInputMode
    final: bool
    stage_results: tuple[dict[str, object], ...]
    operation_fingerprint: bytes | None = None
    config_generation: int = 0


def _replay_payload_size(value: object, *, _seen: set[int] | None = None) -> int:
    """Conservatively account retained replay payload bytes.

    The journal stores nested prompt metadata containing audio/video strings
    and typed fences.  ``sys.getsizeof`` misses referenced containers, while a
    wire encoder is not guaranteed to support every future dataclass.  This
    bounded recursive accounting covers the accepted fingerprint value domain
    and rejects unknown mutable objects instead of silently under-counting.
    """
    seen = _seen if _seen is not None else set()
    if value is None:
        return 1
    if isinstance(value, bool):
        return 1
    if isinstance(value, int | float):
        return 8
    if isinstance(value, str):
        return len(value.encode("utf-8")) + 8
    if isinstance(value, bytes | bytearray | memoryview):
        return len(value) + 8
    if isinstance(value, Enum):
        return _replay_payload_size(value.value, _seen=seen) + 16

    identity = id(value)
    if identity in seen:
        raise ValueError("duplex recovery replay payload contains a reference cycle")
    seen.add(identity)
    try:
        if isinstance(value, Mapping):
            return 16 + sum(
                _replay_payload_size(key, _seen=seen) + _replay_payload_size(item, _seen=seen)
                for key, item in value.items()
            )
        if isinstance(value, list | tuple | set | frozenset):
            return 16 + sum(_replay_payload_size(item, _seen=seen) for item in value)
        if is_dataclass(value) and not isinstance(value, type):
            return 16 + sum(_replay_payload_size(getattr(value, item.name), _seen=seen) for item in fields(value))
    finally:
        seen.discard(identity)
    raise TypeError(f"unsupported duplex recovery replay value: {type(value).__name__}")


@dataclass(frozen=True)
class DuplexReplayAppend:
    """One fully committed materialized append that can rebuild scheduler KV."""

    operation_id: str
    operation_fingerprint: bytes
    prompt: Mapping[str, Any]
    token_count: int
    byte_count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "prompt", MappingProxyType(deepcopy(dict(self.prompt))))


@dataclass(frozen=True, slots=True)
class DuplexGenerationSnapshot:
    resource_generation: int
    context_tokens: int
    context_limit: int | None
    replay_token_count: int
    replay_byte_count: int
    replay_append_count: int
    compaction_count: int
    dropped_append_count: int
    dropped_token_count: int
    dropped_byte_count: int
    rebuild_count: int
    last_recovery_reason: str | None


@dataclass
class _DuplexGenerationStats:
    context_tokens: int = 0
    context_limit: int | None = None
    compaction_count: int = 0
    dropped_append_count: int = 0
    dropped_token_count: int = 0
    dropped_byte_count: int = 0
    rebuild_count: int = 0
    last_recovery_reason: str | None = None


@dataclass(frozen=True, slots=True)
class DuplexContextLedgerSnapshot:
    """Read-only context lifecycle view; it never becomes a second state owner."""

    session_id: str
    fence: DuplexFence
    resource_generation: int
    input_seq: int
    context_tokens: int
    context_limit: int | None
    replay_token_count: int
    replay_byte_count: int
    replay_append_count: int
    completed_append_count: int
    recovery_required: bool
    recovery_reason: str | None
    generation_history: tuple[DuplexGenerationSnapshot, ...] = ()

    @property
    def context_utilization(self) -> float | None:
        if self.context_limit is None or self.context_limit <= 0:
            return None
        return min(max(self.context_tokens / self.context_limit, 0.0), 1.0)


def _fingerprint_value(hasher: Any, value: object) -> None:
    """Hash control payloads with explicit type and length framing."""

    def framed(tag: bytes, payload: bytes = b"") -> None:
        hasher.update(tag)
        hasher.update(len(payload).to_bytes(8, "big"))
        hasher.update(payload)

    if value is None:
        framed(b"n")
    elif isinstance(value, Enum):
        framed(b"e", f"{type(value).__module__}.{type(value).__qualname__}".encode())
        _fingerprint_value(hasher, value.value)
    elif isinstance(value, bool):
        framed(b"b", b"1" if value else b"0")
    elif isinstance(value, int):
        framed(b"i", str(value).encode())
    elif isinstance(value, float):
        framed(b"f", struct.pack(">d", value))
    elif isinstance(value, str):
        framed(b"s", value.encode())
    elif isinstance(value, bytes | bytearray | memoryview):
        framed(b"y", bytes(value))
    elif isinstance(value, Mapping):
        framed(b"m", str(len(value)).encode())
        entries: list[tuple[bytes, object]] = []
        for key, item in value.items():
            key_hasher = sha256()
            _fingerprint_value(key_hasher, key)
            entries.append((key_hasher.digest(), item))
        for key_digest, item in sorted(entries, key=lambda entry: entry[0]):
            framed(b"k", key_digest)
            _fingerprint_value(hasher, item)
    elif isinstance(value, list | tuple):
        framed(b"l" if isinstance(value, list) else b"t", str(len(value)).encode())
        for item in value:
            _fingerprint_value(hasher, item)
    elif isinstance(value, set | frozenset):
        framed(b"q" if isinstance(value, set) else b"r", str(len(value)).encode())
        digests: list[bytes] = []
        for item in value:
            item_hasher = sha256()
            _fingerprint_value(item_hasher, item)
            digests.append(item_hasher.digest())
        for item_digest in sorted(digests):
            framed(b"v", item_digest)
    elif is_dataclass(value) and not isinstance(value, type):
        framed(b"d", f"{type(value).__module__}.{type(value).__qualname__}".encode())
        for item in fields(value):
            framed(b"a", item.name.encode())
            _fingerprint_value(hasher, getattr(value, item.name))
    elif isinstance(struct_fields := getattr(value, "__struct_fields__", None), tuple):
        framed(b"u", f"{type(value).__module__}.{type(value).__qualname__}".encode())
        for name in struct_fields:
            framed(b"a", str(name).encode())
            _fingerprint_value(hasher, getattr(value, name))
    else:
        raise TypeError(f"unsupported duplex append fingerprint value: {type(value).__name__}")


def duplex_append_fingerprint(
    *,
    mode: DuplexInputMode,
    payload: object,
    final: bool,
    config_generation: int,
    request_metadata: object | None = None,
) -> bytes:
    """Bind an append idempotency key to its complete logical input."""
    hasher = sha256()
    _fingerprint_value(hasher, mode)
    _fingerprint_value(hasher, payload)
    _fingerprint_value(hasher, final)
    _fingerprint_value(hasher, config_generation)
    _fingerprint_value(hasher, request_metadata)
    return hasher.digest()


@dataclass
class DuplexSessionRuntimeState:
    """Engine resource handles associated with a core-owned identity fence."""

    fence: DuplexFence
    lease: DuplexLeaseState
    _clock: Callable[[], float] = field(repr=False)
    capabilities: DuplexRuntimeCapabilities = field(default_factory=_default_capabilities)
    session_config: dict[str, Any] = field(default_factory=dict)
    runtime_config: dict[str, Any] = field(default_factory=dict)
    config_generation: int = 0
    stage_bindings: dict[int, DuplexStageBinding] = field(default_factory=dict)
    request_resources: dict[tuple[int, str], DuplexRequestResource] = field(default_factory=dict)
    input_seq: int = 0
    input_turn_seq: int = 0
    _append_turn_key: tuple[int, int, int] | None = None
    completed_append_limit: int = 256
    completed_appends: OrderedDict[str, DuplexCompletedAppend] = field(default_factory=OrderedDict)
    # Logical epoch ownership: physical KV rollover/recovery must not forget
    # committed operations whose full replies have left the bounded cache.
    retired_append_ids: set[bytes] = field(default_factory=set)
    recovery_max_replay_tokens: int = 8192
    recovery_max_replay_bytes: int = 64 * 1024 * 1024
    rollover_trigger_fraction: float = 0.8
    rollover_retain_tokens: int = 4096
    replay_appends: list[DuplexReplayAppend] = field(default_factory=list)
    pending_context_outputs: dict[int, object] = field(default_factory=dict)
    replay_token_count: int = 0
    replay_byte_count: int = 0
    resource_generation: int = 0
    recovery_required: bool = False
    recovery_reason: str | None = None
    scheduler_context_tokens: int = 0
    scheduler_context_limit: int | None = None
    generation_history_limit: int = 8
    _generation_stats: OrderedDict[int, _DuplexGenerationStats] = field(default_factory=OrderedDict, repr=False)

    def __post_init__(self) -> None:
        if self.completed_append_limit <= 0:
            raise ValueError("completed_append_limit must be positive")
        if self.recovery_max_replay_tokens <= 0 or self.recovery_max_replay_bytes <= 0:
            raise ValueError("duplex recovery replay limits must be positive")
        if not 0 <= self.rollover_trigger_fraction < 1:
            raise ValueError("duplex rollover trigger fraction must be in [0, 1)")
        if self.rollover_retain_tokens <= 0:
            raise ValueError("duplex rollover retain tokens must be positive")
        if self.generation_history_limit <= 0:
            raise ValueError("generation_history_limit must be positive")
        if self.rollover_trigger_fraction > 0 and self.rollover_retain_tokens >= self.recovery_max_replay_tokens:
            raise ValueError(
                "duplex rollover retain tokens must be smaller than the recovery replay token limit "
                "when rollover is enabled"
            )
        self._ensure_generation_stats()

    def _ensure_generation_stats(self) -> _DuplexGenerationStats:
        stats = self._generation_stats.get(self.resource_generation)
        if stats is None:
            stats = _DuplexGenerationStats()
            self._generation_stats[self.resource_generation] = stats
            while len(self._generation_stats) > self.generation_history_limit:
                self._generation_stats.popitem(last=False)
        return stats

    @property
    def session_id(self) -> str:
        return self.fence.session_id

    def context_ledger(self) -> DuplexContextLedgerSnapshot:
        """Expose context facts without duplicating mutation ownership."""
        current = self._ensure_generation_stats()
        current.context_tokens = self.scheduler_context_tokens
        current.context_limit = self.scheduler_context_limit
        history = tuple(
            DuplexGenerationSnapshot(
                resource_generation=generation,
                context_tokens=stats.context_tokens,
                context_limit=stats.context_limit,
                replay_token_count=(self.replay_token_count if generation == self.resource_generation else 0),
                replay_byte_count=(self.replay_byte_count if generation == self.resource_generation else 0),
                replay_append_count=(len(self.replay_appends) if generation == self.resource_generation else 0),
                compaction_count=stats.compaction_count,
                dropped_append_count=stats.dropped_append_count,
                dropped_token_count=stats.dropped_token_count,
                dropped_byte_count=stats.dropped_byte_count,
                rebuild_count=stats.rebuild_count,
                last_recovery_reason=stats.last_recovery_reason,
            )
            for generation, stats in self._generation_stats.items()
        )
        return DuplexContextLedgerSnapshot(
            session_id=self.session_id,
            fence=self.fence,
            resource_generation=self.resource_generation,
            input_seq=self.input_seq,
            context_tokens=self.scheduler_context_tokens,
            context_limit=self.scheduler_context_limit,
            replay_token_count=self.replay_token_count,
            replay_byte_count=self.replay_byte_count,
            replay_append_count=len(self.replay_appends),
            completed_append_count=len(self.completed_appends),
            recovery_required=self.recovery_required,
            recovery_reason=self.recovery_reason,
            generation_history=history,
        )

    @property
    def epoch(self) -> int:
        return self.fence.epoch

    @property
    def turn_id(self) -> int:
        return self.fence.turn_id

    def _validate_fence(self, fence: DuplexFence) -> None:
        if fence.session_id != self.session_id or fence.incarnation != self.fence.incarnation:
            raise DuplexFenceMismatchError(self.fence, fence)
        current = self.fence
        if fence.epoch < current.epoch or (
            fence.epoch == current.epoch
            and (fence.turn_id < current.turn_id or fence.response_seq < current.response_seq)
        ):
            raise DuplexFenceMismatchError(current, fence)

    def accept_fence(self, fence: DuplexFence) -> None:
        self._validate_fence(fence)
        if fence.epoch != self.fence.epoch:
            self.input_seq = 0
            self.input_turn_seq = 0
            self._append_turn_key = None
            self.completed_appends.clear()
            self.retired_append_ids.clear()
            self.clear_replay_journal()
            self.pending_context_outputs.clear()
            self.resource_generation = 0
            self.recovery_required = False
            self.recovery_reason = None
            self.scheduler_context_tokens = 0
            self.scheduler_context_limit = None
            self._generation_stats.clear()
            self.resource_generation = 0
            self._ensure_generation_stats()
        self.fence = fence

    def prepare_replay_append(
        self,
        *,
        operation_id: str | None,
        operation_fingerprint: bytes | None,
        prompt: Mapping[str, Any],
    ) -> DuplexReplayAppend:
        if not operation_id:
            raise ValueError("scheduler-native recovery requires a non-empty operation_id")
        if not isinstance(operation_fingerprint, bytes) or not operation_fingerprint:
            raise ValueError("scheduler-native recovery requires a full operation_fingerprint")
        raw_tokens = prompt.get("prompt_token_ids")
        if not isinstance(raw_tokens, list) or not raw_tokens:
            raise ValueError("scheduler-native recovery requires prompt_token_ids")
        token_count = len(raw_tokens)
        byte_count = _replay_payload_size(prompt) + len(operation_id.encode("utf-8")) + len(operation_fingerprint)
        if token_count > self.recovery_max_replay_tokens or byte_count > self.recovery_max_replay_bytes:
            raise RuntimeError(
                "duplex_recovery_journal_entry_too_large: "
                f"tokens={token_count}/{self.recovery_max_replay_tokens}, "
                f"bytes={byte_count}/{self.recovery_max_replay_bytes}"
            )
        return DuplexReplayAppend(
            operation_id=operation_id,
            operation_fingerprint=operation_fingerprint,
            prompt=prompt,
            token_count=token_count,
            byte_count=byte_count,
        )

    def replay_append_would_overflow(self, append: DuplexReplayAppend) -> bool:
        return (
            self.replay_token_count + append.token_count > self.recovery_max_replay_tokens
            or self.replay_byte_count + append.byte_count > self.recovery_max_replay_bytes
        )

    def replay_window_rollover_needed(self, append: DuplexReplayAppend) -> bool:
        if self.replay_append_would_overflow(append):
            return True
        if self.rollover_trigger_fraction <= 0 or self.scheduler_context_limit is None:
            return False
        trigger = max(1, int(self.scheduler_context_limit * self.rollover_trigger_fraction))
        return self.scheduler_context_tokens + append.token_count >= trigger

    def context_rollover_needed(self, append: DuplexReplayAppend) -> bool:
        return self.recovery_required or self.replay_window_rollover_needed(append)

    def compacted_replay_appends(self) -> list[DuplexReplayAppend]:
        retained: list[DuplexReplayAppend] = []
        retained_tokens = 0
        retained_bytes = 0
        for append in reversed(self.replay_appends):
            if retained and retained_tokens + append.token_count > self.rollover_retain_tokens:
                break
            if retained_bytes + append.byte_count > self.recovery_max_replay_bytes:
                break
            retained.append(append)
            retained_tokens += append.token_count
            retained_bytes += append.byte_count
        retained.reverse()
        return retained

    def replace_replay_journal(self, appends: Iterable[DuplexReplayAppend]) -> None:
        retained = list(appends)
        token_count = sum(item.token_count for item in retained)
        byte_count = sum(item.byte_count for item in retained)
        if token_count > self.recovery_max_replay_tokens or byte_count > self.recovery_max_replay_bytes:
            raise RuntimeError("duplex_recovery_journal_capacity_exhausted")
        self.replay_appends = retained
        self.replay_token_count = token_count
        self.replay_byte_count = byte_count

    def begin_resource_generation(
        self,
        retained_appends: Iterable[DuplexReplayAppend],
        *,
        reason: str,
    ) -> None:
        """Atomically roll physical context and account for whole-unit compaction."""
        previous = tuple(self.replay_appends)
        retained = tuple(retained_appends)
        retained_ids = {item.operation_id for item in retained}
        dropped = [item for item in previous if item.operation_id not in retained_ids]
        self.resource_generation += 1
        self.replace_replay_journal(retained)
        stats = self._ensure_generation_stats()
        stats.compaction_count += 1
        stats.dropped_append_count += len(dropped)
        stats.dropped_token_count += sum(item.token_count for item in dropped)
        stats.dropped_byte_count += sum(item.byte_count for item in dropped)
        stats.rebuild_count += 1
        stats.last_recovery_reason = reason
        self.scheduler_context_tokens = 0
        self.scheduler_context_limit = None

    def record_replay_append(self, append: DuplexReplayAppend) -> None:
        if self.replay_append_would_overflow(append):
            raise RuntimeError("duplex_recovery_journal_capacity_exhausted")
        self.replay_appends.append(append)
        self.replay_token_count += append.token_count
        self.replay_byte_count += append.byte_count
        self._ensure_generation_stats()

    def clear_replay_journal(self) -> None:
        self.replay_appends.clear()
        self.replay_token_count = 0
        self.replay_byte_count = 0

    def mark_recovery_required(self, reason: str) -> None:
        self.recovery_required = True
        self.recovery_reason = reason

    def mark_recovered(self) -> None:
        self.recovery_required = False
        self.recovery_reason = None
        # Physical prompt size is model/adapter-specific and can include a
        # rebuilt prefix. It must be refreshed by scheduler metrics, not
        # inferred from logical replay token count.

    def update_scheduler_context(self, *, tokens: int, limit: int | None) -> None:
        self.scheduler_context_tokens = max(int(tokens), 0)
        if limit is not None and limit > 0:
            self.scheduler_context_limit = int(limit)
        stats = self._ensure_generation_stats()
        stats.context_tokens = self.scheduler_context_tokens
        stats.context_limit = self.scheduler_context_limit

    def touch(self, fence: DuplexFence, activity: DuplexLeaseActivity) -> None:
        self._validate_fence(fence)
        self.lease.touch(self._clock(), activity)

    def detach(self, fence: DuplexFence) -> None:
        self._validate_fence(fence)
        self.lease.detach(self._clock())

    def resume(self, fence: DuplexFence, *, expected_lease_generation: int) -> int:
        self._validate_fence(fence)
        return self.lease.resume(
            self._clock(),
            expected_generation=expected_lease_generation,
        )

    def begin_operation(self, fence: DuplexFence, operation_id: str) -> None:
        self._validate_fence(fence)
        self.lease.begin_operation(self._clock(), operation_id)

    def end_operation(self, fence: DuplexFence, operation_id: str) -> None:
        self._validate_fence(fence)
        self.lease.end_operation(self._clock(), operation_id)

    def replace_session_config(self, session_config: dict[str, Any]) -> None:
        self.session_config = dict(session_config)
        self.config_generation += 1

    def replace_runtime_config(self, runtime_config: dict[str, Any]) -> None:
        self.runtime_config = dict(runtime_config)
        self.config_generation += 1

    def replace_configs(
        self,
        *,
        session_config: dict[str, Any] | None = None,
        runtime_config: dict[str, Any] | None = None,
    ) -> None:
        """Atomically publish one validated configuration generation."""
        if session_config is None and runtime_config is None:
            return
        next_session_config = self.session_config if session_config is None else dict(session_config)
        next_runtime_config = self.runtime_config if runtime_config is None else dict(runtime_config)
        self.session_config = next_session_config
        self.runtime_config = next_runtime_config
        self.config_generation += 1

    def reserve_stage_request(self, stage_id: int, request_id: str, *, fence: DuplexFence) -> None:
        self._validate_fence(fence)
        resource_key = (stage_id, request_id)
        existing = self.request_resources.get(resource_key)
        if existing is not None:
            if (
                existing.fence.session_id != fence.session_id
                or existing.fence.incarnation != fence.incarnation
                or existing.fence.epoch != fence.epoch
            ):
                raise ValueError(f"Duplex request resource already reserved with different identity: {request_id}")
            return
        self.request_resources[resource_key] = DuplexRequestResource(
            stage_id=stage_id,
            request_id=request_id,
            fence=fence,
        )

    def bind_stage_request(self, stage_id: int, request_id: str, *, fence: DuplexFence) -> None:
        self.reserve_stage_request(stage_id, request_id, fence=fence)
        self.accept_fence(fence)
        resource = self.request_resources[(stage_id, request_id)]
        resource.fence = fence
        resource.submitted = True
        self.stage_bindings[stage_id] = DuplexStageBinding(request_id=request_id, fence=fence)

    def stage_request_ids(self, fence: DuplexFence | None = None) -> list[str]:
        return [
            binding.request_id for binding in self.stage_bindings.values() if fence is None or binding.fence == fence
        ]

    def resource_request_ids(
        self,
        fence: DuplexFence | None = None,
        *,
        submitted: bool | None = None,
    ) -> list[str]:
        return list(
            dict.fromkeys(
                resource.request_id
                for resource in self.request_resources.values()
                if (fence is None or resource.fence == fence) and (submitted is None or resource.submitted is submitted)
            )
        )

    def release_request_ids(self, request_ids: list[str]) -> None:
        released = set(request_ids)
        if not released:
            return
        self.stage_bindings = {
            stage_id: binding for stage_id, binding in self.stage_bindings.items() if binding.request_id not in released
        }
        self.request_resources = {
            resource_key: resource
            for resource_key, resource in self.request_resources.items()
            if resource.request_id not in released
        }

    def prepare_append(self, *, mode: DuplexInputMode, fence: DuplexFence) -> DuplexAppendReservation:
        if mode not in self.capabilities.input_modes:
            raise ValueError(f"Duplex input mode {mode.value!r} is not supported by session {self.session_id}")
        self._validate_fence(fence)
        input_seq = 0 if fence.epoch != self.fence.epoch else self.input_seq
        input_turn_seq = 0 if fence.epoch != self.fence.epoch else self.input_turn_seq
        append_turn_key = None if fence.epoch != self.fence.epoch else self._append_turn_key
        turn_key = (fence.epoch, fence.turn_id, fence.response_seq)
        turn_seq = input_turn_seq + 1 if turn_key == append_turn_key else 1
        return DuplexAppendReservation(
            fence=fence,
            mode=mode,
            base_fence=self.fence,
            base_input_seq=self.input_seq,
            base_input_turn_seq=self.input_turn_seq,
            base_append_turn_key=self._append_turn_key,
            update=DuplexInputAppend(
                seq=input_seq + 1,
                turn_seq=turn_seq,
                turn_id=fence.turn_id,
            ),
        )

    def commit_append(self, reservation: DuplexAppendReservation) -> DuplexInputAppend:
        if (
            self.fence != reservation.base_fence
            or self.input_seq != reservation.base_input_seq
            or self.input_turn_seq != reservation.base_input_turn_seq
            or self._append_turn_key != reservation.base_append_turn_key
        ):
            raise RuntimeError("duplex append reservation is stale")
        self.accept_fence(reservation.fence)
        self.input_seq = reservation.update.seq
        self.input_turn_seq = reservation.update.turn_seq
        self._append_turn_key = (
            reservation.fence.epoch,
            reservation.fence.turn_id,
            reservation.fence.response_seq,
        )
        return reservation.update

    def completed_append(
        self,
        operation_id: str,
        *,
        fence: DuplexFence,
        mode: DuplexInputMode,
        final: bool,
        operation_fingerprint: bytes,
        config_generation: int,
    ) -> list[dict[str, object]] | None:
        self._validate_fence(fence)
        completed = self.completed_appends.get(operation_id)
        if completed is None:
            if sha256(operation_id.encode()).digest() in self.retired_append_ids:
                raise RuntimeError(
                    f"streaming_prompt_idempotency_window_expired: operation={operation_id!r}; "
                    "the append was committed in this session epoch"
                )
            if (
                len(self.completed_appends) >= self.completed_append_limit
                and len(self.retired_append_ids) >= _APPEND_TOMBSTONE_LIMIT
            ):
                # Reject BEFORE submitting; never discover exhausted receipt
                # capacity after the scheduler has already committed the unit.
                raise RuntimeError("streaming_prompt_idempotency_capacity_exhausted: cancel/reopen the session")
            return None
        if (
            completed.fence != fence
            or completed.mode != mode
            or completed.final != final
            or completed.operation_fingerprint != operation_fingerprint
            or completed.config_generation != config_generation
        ):
            raise ValueError(f"duplex append operation {operation_id!r} was reused with different metadata")
        return [dict(result) for result in completed.stage_results]

    def record_completed_append(
        self,
        operation_id: str,
        *,
        fence: DuplexFence,
        mode: DuplexInputMode,
        final: bool,
        stage_results: list[dict[str, object]],
        operation_fingerprint: bytes | None = None,
        config_generation: int = 0,
    ) -> None:
        self.completed_appends[operation_id] = DuplexCompletedAppend(
            fence=fence,
            mode=mode,
            final=final,
            stage_results=tuple(dict(result) for result in stage_results),
            operation_fingerprint=operation_fingerprint,
            config_generation=config_generation,
        )
        self.completed_appends.move_to_end(operation_id)
        while len(self.completed_appends) > self.completed_append_limit:
            retired_id, _ = self.completed_appends.popitem(last=False)
            self.retired_append_ids.add(sha256(retired_id.encode()).digest())

    def append_input(self, *, mode: DuplexInputMode, fence: DuplexFence) -> DuplexInputAppend:
        return self.commit_append(self.prepare_append(mode=mode, fence=fence))

    def release_fence(self, fence: DuplexFence) -> list[str]:
        stale = self.resource_request_ids(fence)
        self.stage_bindings = {
            stage_id: binding for stage_id, binding in self.stage_bindings.items() if binding.fence != fence
        }
        self.request_resources = {
            resource_key: resource
            for resource_key, resource in self.request_resources.items()
            if resource.fence != fence
        }
        return stale

    def cancel_fence(self, cancelled_fence: DuplexFence, next_fence: DuplexFence) -> list[str]:
        stale = self.prepare_cancel_fence(cancelled_fence, next_fence)
        self.release_fence(cancelled_fence)
        return stale

    def prepare_cancel_fence(self, cancelled_fence: DuplexFence, next_fence: DuplexFence) -> list[str]:
        """Advance the cancellation fence without dropping cleanup records."""
        if cancelled_fence.session_id != self.session_id or cancelled_fence.incarnation != self.fence.incarnation:
            raise DuplexFenceMismatchError(self.fence, cancelled_fence)
        if (
            next_fence.session_id != self.session_id
            or next_fence.incarnation != self.fence.incarnation
            or next_fence.epoch <= cancelled_fence.epoch
        ):
            raise DuplexFenceMismatchError(cancelled_fence, next_fence)
        current_key = (self.fence.epoch, self.fence.turn_id, self.fence.response_seq)
        cancelled_key = (cancelled_fence.epoch, cancelled_fence.turn_id, cancelled_fence.response_seq)
        next_key = (next_fence.epoch, next_fence.turn_id, next_fence.response_seq)
        if cancelled_key > current_key:
            raise DuplexFenceMismatchError(self.fence, cancelled_fence)
        if next_key > current_key:
            self.accept_fence(next_fence)
        return self.resource_request_ids(cancelled_fence)

    def begin_close(self, fence: DuplexFence, *, reason: str) -> bool:
        """Make close irreversible while retaining resources for cleanup retry."""
        self.accept_fence(fence)
        if self.lease.terminal_reason is not None:
            return True
        return self.lease.mark_terminal(reason)

    def finalize_close(self) -> list[str]:
        return self.close()

    def close(self, fence: DuplexFence | None = None) -> list[str]:
        if fence is not None:
            self.accept_fence(fence)
        stale = self.resource_request_ids()
        self.stage_bindings.clear()
        self.request_resources.clear()
        return stale

    def terminate(self, fence: DuplexFence, *, reason: str) -> bool:
        if not self.begin_close(fence, reason=reason):
            return False
        self.finalize_close()
        return True


class DuplexSessionRuntimeManager:
    def __init__(
        self,
        *,
        clock: Callable[[], float] | None = None,
        max_sessions: int | None = None,
        completed_append_limit: int = 256,
        recovery_max_replay_tokens: int = 8192,
        recovery_max_replay_bytes: int = 64 * 1024 * 1024,
        rollover_trigger_fraction: float = 0.8,
        rollover_retain_tokens: int = 4096,
    ) -> None:
        if max_sessions is not None and max_sessions <= 0:
            raise ValueError("max_sessions must be positive or null")
        if completed_append_limit <= 0:
            raise ValueError("completed_append_limit must be positive")
        self._sessions: dict[str, DuplexSessionRuntimeState] = {}
        self._clock = clock or time.monotonic
        self._max_sessions = max_sessions
        self._completed_append_limit = completed_append_limit
        self._recovery_max_replay_tokens = recovery_max_replay_tokens
        self._recovery_max_replay_bytes = recovery_max_replay_bytes
        self._rollover_trigger_fraction = rollover_trigger_fraction
        self._rollover_retain_tokens = rollover_retain_tokens

    @property
    def session_count(self) -> int:
        return len(self._sessions)

    @property
    def closing_session_count(self) -> int:
        return sum(session.lease.terminal_reason is not None for session in self._sessions.values())

    @property
    def active_session_count(self) -> int:
        return self.session_count - self.closing_session_count

    @property
    def max_sessions(self) -> int | None:
        return self._max_sessions

    @property
    def has_capacity(self) -> bool:
        return self._max_sessions is None or self.session_count < self._max_sessions

    def iter_sessions(self) -> tuple[DuplexSessionRuntimeState, ...]:
        """Return a stable snapshot without exposing the mutable session map."""
        return tuple(self._sessions.values())

    def context_ledgers(self) -> tuple[DuplexContextLedgerSnapshot, ...]:
        """Return read-only context views for diagnostics and bounded metrics."""
        return tuple(session.context_ledger() for session in self._sessions.values())

    def open_session(
        self,
        fence: DuplexFence,
        *,
        capabilities: DuplexRuntimeCapabilities | None = None,
        session_config: dict[str, Any] | None = None,
        runtime_config: dict[str, Any] | None = None,
        lease_config: DuplexLeaseConfig | None = None,
    ) -> DuplexSessionRuntimeState:
        if not isinstance(fence, DuplexFence):
            raise TypeError("open_session requires DuplexFence")
        if fence.session_id in self._sessions:
            raise ValueError(f"Duplex session already exists: {fence.session_id}")
        if self._max_sessions is not None and len(self._sessions) >= self._max_sessions:
            raise RuntimeError(f"duplex_session_capacity_exhausted: limit={self._max_sessions}")
        session = DuplexSessionRuntimeState(
            fence=fence,
            lease=DuplexLeaseState(
                config=lease_config or DuplexLeaseConfig(),
                generation=0,
                last_activity=self._clock(),
            ),
            _clock=self._clock,
            capabilities=capabilities or _default_capabilities(),
            session_config=dict(session_config or {}),
            runtime_config=dict(runtime_config or {}),
            completed_append_limit=self._completed_append_limit,
            recovery_max_replay_tokens=self._recovery_max_replay_tokens,
            recovery_max_replay_bytes=self._recovery_max_replay_bytes,
            rollover_trigger_fraction=self._rollover_trigger_fraction,
            rollover_retain_tokens=self._rollover_retain_tokens,
        )
        self._sessions[fence.session_id] = session
        return session

    def get(self, session_id: str) -> DuplexSessionRuntimeState | None:
        return self._sessions.get(session_id)

    def require(self, session_id: str) -> DuplexSessionRuntimeState:
        session = self.get(session_id)
        if session is None:
            raise KeyError(f"Unknown duplex session: {session_id}")
        return session

    def close_session(
        self,
        fence: DuplexFence,
        *,
        reason: str = "explicit_close",
    ) -> DuplexSessionRuntimeState | None:
        if not isinstance(fence, DuplexFence):
            raise TypeError("close_session requires DuplexFence")
        session = self._sessions.get(fence.session_id)
        if session is not None:
            if session.lease.terminal_reason is not None:
                return None
            if not session.terminate(fence, reason=reason):
                return None
            if self._sessions.get(fence.session_id) is session:
                self._sessions.pop(fence.session_id)
        return session

    def begin_close_session(
        self,
        fence: DuplexFence,
        *,
        reason: str = "explicit_close",
    ) -> DuplexSessionRuntimeState | None:
        if not isinstance(fence, DuplexFence):
            raise TypeError("begin_close_session requires DuplexFence")
        session = self._sessions.get(fence.session_id)
        if session is not None:
            session.begin_close(fence, reason=reason)
        return session

    def finalize_close_session(self, session: DuplexSessionRuntimeState) -> None:
        session.finalize_close()
        if self._sessions.get(session.session_id) is session:
            self._sessions.pop(session.session_id, None)

    def collect_expired(
        self,
        now: float | None = None,
        *,
        excluded_session_ids: set[str] | None = None,
    ) -> list[DuplexSessionExpiry]:
        effective_now = self._clock() if now is None else now
        excluded = excluded_session_ids or set()
        expired: list[DuplexSessionExpiry] = []
        for session_id, session in list(self._sessions.items()):
            if session_id in excluded:
                continue
            if session.lease.disconnect_grace_expired(effective_now):
                reason = "disconnect_grace_expired"
            elif session.lease.idle_expired(effective_now):
                reason = "idle_ttl_expired"
            else:
                continue
            submitted = tuple(session.resource_request_ids(submitted=True))
            reserved = tuple(session.resource_request_ids(submitted=False))
            if not session.begin_close(session.fence, reason=reason):
                continue
            expired.append(
                DuplexSessionExpiry(
                    session_id=session_id,
                    fence=session.fence,
                    lease_generation=session.lease.generation,
                    reason=reason,
                    submitted_request_ids=submitted,
                    reserved_request_ids=reserved,
                )
            )
        return expired

    def close_sessions_for_request_ids(
        self,
        request_ids: list[str],
        *,
        reason: str = "request_cleanup",
    ) -> dict[str, list[str]]:
        request_id_set = set(request_ids)
        closed: dict[str, list[str]] = {}
        for session_id, session in list(self._sessions.items()):
            stale = session.resource_request_ids()
            if request_id_set.isdisjoint(stale):
                continue
            if not session.begin_close(session.fence, reason=reason):
                continue
            closed[session_id] = stale
        return closed

    def finalize_closed_sessions(self, session_ids: Iterable[str]) -> None:
        for session_id in session_ids:
            session = self._sessions.get(session_id)
            if session is None or session.lease.terminal_reason is None:
                continue
            self.finalize_close_session(session)


__all__ = [
    "DuplexAppendReservation",
    "DuplexCompletedAppend",
    "DuplexContextLedgerSnapshot",
    "DuplexFenceMismatchError",
    "DuplexInputAppend",
    "DuplexLeaseActivity",
    "DuplexLeaseConfig",
    "DuplexLeaseState",
    "DuplexRequestResource",
    "DuplexReplayAppend",
    "DuplexSessionExpiry",
    "DuplexSessionRuntimeManager",
    "DuplexSessionRuntimeState",
    "DuplexStageBinding",
    "duplex_append_fingerprint",
]
