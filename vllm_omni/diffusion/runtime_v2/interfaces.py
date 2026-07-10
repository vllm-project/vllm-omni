# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from typing import Any

from vllm_omni.diffusion.runtime_v2.protocol import (
    ArtifactHandle,
    ArtifactValue,
    InferenceTask,
    RequestExecutionPlan,
    TaskKind,
    WorkerEvent,
)

# Re-export ArtifactValue so callers that `from interfaces import ArtifactValue` work.
__all__ = [
    "ArtifactStore",
    "ArtifactValue",
    "RuntimeV2Adapter",
    "SchedulerPolicy",
    "TaskCompiler",
    "WorkerExecutor",
]


class TaskCompiler(ABC):
    @abstractmethod
    def compile_request(self, request: Any) -> RequestExecutionPlan:
        raise NotImplementedError


class ArtifactStore(ABC):
    @abstractmethod
    def put(self, artifact: ArtifactHandle, value: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    def get(self, artifact: ArtifactHandle) -> Any:
        raise NotImplementedError

    @abstractmethod
    def is_ready(self, artifact: ArtifactHandle) -> bool:
        raise NotImplementedError

    @abstractmethod
    def evict_request(self, request_id: str) -> None:
        raise NotImplementedError


class SchedulerPolicy(ABC):
    @abstractmethod
    def on_request_submitted(self, plan: RequestExecutionPlan) -> Iterable[InferenceTask]:
        raise NotImplementedError

    @abstractmethod
    def on_tasks_runnable(self, tasks: Iterable[InferenceTask]) -> Iterable[InferenceTask]:
        raise NotImplementedError

    @abstractmethod
    def on_worker_event(self, event: WorkerEvent) -> Iterable[InferenceTask]:
        raise NotImplementedError


class WorkerExecutor(ABC):
    @abstractmethod
    def execute(self, task: InferenceTask, resolved_inputs: Mapping[str, Any]) -> tuple[ArtifactValue, ...]:
        raise NotImplementedError


class RuntimeV2Adapter(ABC):
    model_class_name: str

    @property
    @abstractmethod
    def supported_task_kinds(self) -> tuple[TaskKind, ...]:
        raise NotImplementedError

    @abstractmethod
    def normalize_request(self, request: Any, denoise_chunk_size: int) -> Any:
        raise NotImplementedError

    @abstractmethod
    def build_task_compiler(
        self,
        default_denoise_chunk_size: int,
        *,
        od_config: Any = None,
        pipeline: Any = None,
    ) -> TaskCompiler:
        raise NotImplementedError

    @abstractmethod
    def build_executors(self, pipeline: Any) -> dict[TaskKind, WorkerExecutor]:
        raise NotImplementedError

    @abstractmethod
    def validate_pipeline(self, pipeline: Any, od_config: Any) -> None:
        raise NotImplementedError
