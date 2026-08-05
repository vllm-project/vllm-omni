# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Generic engine lease primitives shared by optional session runtimes."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from vllm_omni.experimental.fullduplex.engine.messages import DuplexFence


class DuplexLeaseActivity(str, Enum):
    APPEND = "append"
    SIGNAL = "signal"
    PLAYBACK_ACK = "playback_ack"
    HEARTBEAT = "heartbeat"
    ATTACH = "attach"
    DETACH = "detach"
    RESUME = "resume"
    MODEL_OUTPUT = "model_output"


@dataclass(frozen=True)
class DuplexLeaseConfig:
    idle_ttl_s: float | None = 300.0
    disconnect_grace_s: float = 30.0

    def __post_init__(self) -> None:
        if self.idle_ttl_s is not None and self.idle_ttl_s <= 0:
            raise ValueError("idle_ttl_s must be positive or null")
        if self.disconnect_grace_s <= 0:
            raise ValueError("disconnect_grace_s must be positive")


@dataclass
class DuplexLeaseState:
    config: DuplexLeaseConfig
    generation: int
    last_activity: float
    detached_at: float | None = None
    active_operations: set[str] = field(default_factory=set)
    terminal_reason: str | None = None

    @property
    def expires_at(self) -> float | None:
        if self.config.idle_ttl_s is None:
            return None
        return self.last_activity + self.config.idle_ttl_s

    def _require_open(self) -> None:
        if self.terminal_reason is not None:
            raise RuntimeError(f"duplex lease is terminal: {self.terminal_reason}")

    def touch(self, now: float, activity: DuplexLeaseActivity) -> None:
        self._require_open()
        if not isinstance(activity, DuplexLeaseActivity):
            raise TypeError("lease activity must be DuplexLeaseActivity")
        self.last_activity = now

    def detach(self, now: float) -> None:
        self.touch(now, DuplexLeaseActivity.DETACH)
        self.detached_at = now

    def resume(self, now: float, *, expected_generation: int) -> int:
        self._require_open()
        if expected_generation != self.generation:
            raise ValueError(f"duplex lease generation mismatch: expected {self.generation}, got {expected_generation}")
        self.generation += 1
        self.detached_at = None
        self.last_activity = now
        return self.generation

    def begin_operation(self, now: float, operation_id: str) -> None:
        self._require_open()
        if not operation_id:
            raise ValueError("duplex lease operation_id must not be empty")
        if operation_id in self.active_operations:
            raise ValueError(f"duplex lease operation already active: {operation_id}")
        self.active_operations.add(operation_id)
        self.last_activity = now

    def end_operation(self, now: float, operation_id: str) -> None:
        self._require_open()
        if operation_id not in self.active_operations:
            raise KeyError(f"duplex lease operation is not active: {operation_id}")
        self.active_operations.remove(operation_id)
        self.last_activity = now

    def disconnect_grace_expired(self, now: float) -> bool:
        return (
            self.terminal_reason is None
            and self.detached_at is not None
            and now >= self.detached_at + self.config.disconnect_grace_s
        )

    def idle_expired(self, now: float) -> bool:
        expires_at = self.expires_at
        return (
            self.terminal_reason is None and not self.active_operations and expires_at is not None and now >= expires_at
        )

    def mark_terminal(self, reason: str) -> bool:
        if self.terminal_reason is not None:
            return False
        self.terminal_reason = reason
        self.generation += 1
        self.detached_at = None
        self.active_operations.clear()
        return True


@dataclass(frozen=True)
class DuplexSessionExpiry:
    session_id: str
    fence: DuplexFence
    lease_generation: int
    reason: str
    submitted_request_ids: tuple[str, ...]
    reserved_request_ids: tuple[str, ...]


__all__ = [
    "DuplexLeaseActivity",
    "DuplexLeaseConfig",
    "DuplexLeaseState",
    "DuplexSessionExpiry",
]
