# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TaskKind(str, Enum):
    TEXT_ENCODE = "text_encode"
    DIT_STEP_CHUNK = "dit_step_chunk"
    VAE_DECODE = "vae_decode"


class ArtifactKind(str, Enum):
    REQUEST_STATE = "request_state"
    TENSOR = "tensor"
    OUTPUT = "output"


class ArtifactLayout(str, Enum):
    HOST = "host"
    WORKER_LOCAL = "worker_local"


class TaskStatus(str, Enum):
    PENDING = "pending"
    READY = "ready"
    DISPATCHED = "dispatched"
    RUNNING = "running"
    FINISHED = "finished"
    FAILED = "failed"


class WorkerEventKind(str, Enum):
    TASK_LAUNCH_END = "task_launch_end"
    TASK_EXEC_BEGIN = "task_exec_begin"
    TASK_EXEC_END = "task_exec_end"
    TASK_FAILED = "task_failed"
    REQUEST_FINISHED = "request_finished"
    REQUEST_FAILED = "request_failed"


@dataclass(frozen=True)
class ParallelSpec:
    """Static parallel metadata for scheduling and fixed worker sessions."""

    tp: int = 1
    sp: int = 1
    cfg: int = 1
    cfg_parallel: bool = False

    def __post_init__(self) -> None:
        if self.tp < 1:
            raise ValueError(f"tp must be >= 1, got {self.tp}")
        if self.sp < 1:
            raise ValueError(f"sp must be >= 1, got {self.sp}")
        cfg = int(self.cfg)
        if cfg < 1:
            raise ValueError(f"cfg must be >= 1, got {cfg}")
        if self.cfg_parallel and cfg == 1:
            cfg = 2
        object.__setattr__(self, "cfg", cfg)
        object.__setattr__(self, "cfg_parallel", cfg > 1)


@dataclass(frozen=True)
class ExecutionGroupSpec:
    group_id: str
    ranks: tuple[int, ...]
    parallel_spec: ParallelSpec
    supported_task_kinds: tuple[TaskKind, ...]
    ulysses_degree: int = 1
    ring_degree: int = 1

    def __post_init__(self) -> None:
        if not self.ranks:
            raise ValueError("execution group must contain at least one rank")
        ulysses_degree = int(self.ulysses_degree)
        ring_degree = int(self.ring_degree)
        sp = int(self.parallel_spec.sp)
        if ulysses_degree < 1:
            raise ValueError(f"ulysses_degree must be >= 1, got {self.ulysses_degree}")
        if ring_degree < 1:
            raise ValueError(f"ring_degree must be >= 1, got {self.ring_degree}")
        if sp != ulysses_degree * ring_degree and ulysses_degree == 1 and ring_degree == 1:
            ulysses_degree = sp
            object.__setattr__(self, "ulysses_degree", ulysses_degree)
        if sp != ulysses_degree * ring_degree:
            raise ValueError(
                "execution group sequence parallel mismatch: "
                f"sp={self.parallel_spec.sp} != ulysses_degree({ulysses_degree}) * ring_degree({ring_degree})"
            )


@dataclass(frozen=True)
class ArtifactHandle:
    request_id: str
    artifact_id: str
    kind: ArtifactKind
    layout: ArtifactLayout
    producer_task_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ArtifactValue:
    handle: ArtifactHandle
    value: Any


@dataclass(frozen=True)
class WorkerLocalArtifactRef:
    handle: ArtifactHandle
    group_id: str
    worker_rank: int


@dataclass(frozen=True)
class StepRange:
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError(f"start must be >= 0, got {self.start}")
        if self.end <= self.start:
            raise ValueError(f"end must be greater than start, got start={self.start}, end={self.end}")


@dataclass
class InferenceTask:
    task_id: str
    request_id: str
    kind: TaskKind
    group_id: str | None
    parallel_spec: ParallelSpec
    status: TaskStatus = TaskStatus.PENDING
    priority: int = 0
    dependencies: tuple[str, ...] = ()
    inputs: tuple[ArtifactHandle, ...] = ()
    outputs: tuple[ArtifactHandle, ...] = ()
    step_range: StepRange | None = None
    payload: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class RequestExecutionPlan:
    request_id: str
    tasks: dict[str, InferenceTask]
    terminal_task_ids: tuple[str, ...]
    initial_artifacts: tuple[ArtifactValue, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WorkerEvent:
    event_id: str
    task_id: str
    request_id: str
    group_id: str
    worker_rank: int
    kind: WorkerEventKind
    timestamp_ns: int
    message: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)
