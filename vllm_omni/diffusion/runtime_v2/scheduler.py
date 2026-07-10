# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from vllm.logger import init_logger

from vllm_omni.diffusion.runtime_v2.interfaces import ArtifactStore, SchedulerPolicy, TaskCompiler

# Policy re-export for convenience
from vllm_omni.diffusion.runtime_v2.policies import FCFSSchedulerPolicy  # noqa: F401
from vllm_omni.diffusion.runtime_v2.protocol import (
    ArtifactHandle,
    ArtifactValue,
    InferenceTask,
    RequestExecutionPlan,
    TaskStatus,
    WorkerEvent,
    WorkerEventKind,
    WorkerLocalArtifactRef,
)
from vllm_omni.diffusion.runtime_v2.topology import RuntimeTopology

logger = init_logger(__name__)

# Upper bound on the terminal-output fetch (worker GPU->host copy + serialize).
# The async fetch path (start_fetch_artifacts / poll_fetch_artifacts) otherwise
# has NO deadline, so a worker that finishes the terminal task but then wedges
# (e.g. a CUDA-stuck D2H copy while the process stays alive) would keep the
# request "pending" forever. Generous -- a real fetch is sub-second to a few
# seconds even for video.
_OUTPUT_FETCH_TIMEOUT_S = 300.0


class InMemoryArtifactStore(ArtifactStore):
    def __init__(self) -> None:
        self._values: dict[tuple[str, str], Any] = {}

    def put(self, artifact: ArtifactHandle, value: Any) -> None:
        self._values[(artifact.request_id, artifact.artifact_id)] = value

    def get(self, artifact: ArtifactHandle) -> Any:
        return self._values[(artifact.request_id, artifact.artifact_id)]

    def is_ready(self, artifact: ArtifactHandle) -> bool:
        return (artifact.request_id, artifact.artifact_id) in self._values

    def evict_request(self, request_id: str) -> None:
        keys = [key for key in self._values if key[0] == request_id]
        for key in keys:
            del self._values[key]


class GlobalScheduler:
    """Minimal event-driven task-graph scheduler for diffusion runtime_v2."""

    def __init__(
        self,
        topology: RuntimeTopology,
        worker_pool: Any,
        compiler: TaskCompiler,
        artifact_store: ArtifactStore,
        policy: SchedulerPolicy,
    ) -> None:
        self.topology = topology
        self.worker_pool = worker_pool
        self.compiler = compiler
        self.artifact_store = artifact_store
        self.policy = policy

        self.plans: dict[str, RequestExecutionPlan] = {}
        self.task_index: dict[str, InferenceTask] = {}
        self.pending_dependencies: dict[str, int] = {}
        self.dependents: dict[str, list[str]] = {}
        self.failed_requests: dict[str, str] = {}
        self.released_requests: set[str] = set()
        self.cleaned_requests: set[str] = set()
        self.failed_tasks: set[str] = set()
        self.artifact_refcounts: dict[tuple[str, str], int] = {}
        self.request_artifacts: dict[str, set[tuple[str, str]]] = {}
        self.pending_output_fetches: dict[str, _PendingOutputFetch] = {}
        # request_id -> delivered terminal output. Membership (NOT value identity)
        # is the "request finished" signal, so a legitimately None/falsy output
        # still reports finished, and the artifact store + per-request bookkeeping
        # can be evicted at cleanup without breaking a later get_request_status poll.
        self._completed_outputs: dict[str, Any] = {}

    def start(self, timeout_s: float | None = None) -> None:
        # Forward the caller's startup timeout (the runner passes
        # od_config.stage_init_timeout) so a slow checkpoint load honors
        # --stage-init-timeout instead of the worker pool's hardcoded default.
        if timeout_s is not None:
            self.worker_pool.start(timeout_s=timeout_s)
        else:
            self.worker_pool.start()

    def shutdown(self) -> None:
        self.worker_pool.shutdown()

    def submit_request(self, request: Any) -> str:
        plan = self.compiler.compile_request(request)
        return self._submit_compiled_plan(plan)

    def _submit_compiled_plan(self, plan: RequestExecutionPlan) -> str:
        self.plans[plan.request_id] = plan
        logger.debug(
            "runtime_v2 plan compiled: request_id=%s tasks=%s terminal=%s metadata=%s",
            plan.request_id,
            len(plan.tasks),
            len(plan.terminal_task_ids),
            plan.metadata,
        )

        for artifact_value in plan.initial_artifacts:
            self.artifact_store.put(artifact_value.handle, artifact_value.value)

        request_artifacts: set[tuple[str, str]] = set()
        for task in plan.tasks.values():
            if task.group_id is not None:
                group = self.topology.get_group(task.group_id)
                if task.kind not in group.supported_task_kinds:
                    raise ValueError(
                        f"task {task.task_id} is pinned to unsupported group {task.group_id!r} for kind={task.kind}"
                    )
            elif self.topology.select_group_for_task(task.kind) is None:
                raise ValueError(f"no execution group supports task kind {task.kind}")
            self.task_index[task.task_id] = task
            self.pending_dependencies[task.task_id] = len(task.dependencies)
            for artifact in task.inputs:
                key = (artifact.request_id, artifact.artifact_id)
                self.artifact_refcounts[key] = self.artifact_refcounts.get(key, 0) + 1
                request_artifacts.add(key)
            for artifact in task.outputs:
                request_artifacts.add((artifact.request_id, artifact.artifact_id))
            for dependency in task.dependencies:
                self.dependents.setdefault(dependency, []).append(task.task_id)
        for artifact_value in plan.initial_artifacts:
            request_artifacts.add((artifact_value.handle.request_id, artifact_value.handle.artifact_id))
        self.request_artifacts[plan.request_id] = request_artifacts

        self._dispatch_tasks(self.policy.on_request_submitted(plan))
        return plan.request_id

    def poll_once(self, timeout_s: float = 0.0) -> None:
        events = self.worker_pool.poll(timeout_s=timeout_s)
        for event in events:
            self._handle_worker_event(event)

    def get_request_status(self, request_id: str) -> tuple[str, Any | None]:
        """
        Returns:
            ("pending", None) if request is not done,
            ("finished", output) if request completed successfully,
            ("failed", error_message) if request failed.
        """
        if request_id in self.failed_requests:
            return "failed", self.failed_requests[request_id]
        if request_id in self._completed_outputs:
            return "finished", self._completed_outputs[request_id]
        if request_id not in self.plans and request_id not in self.pending_output_fetches:
            return "pending", None
        self._try_collect_request_output(request_id)
        if request_id in self._completed_outputs:
            return "finished", self._completed_outputs[request_id]
        if request_id in self.failed_requests:
            return "failed", self.failed_requests[request_id]
        return "pending", None

    def _dispatch_tasks(self, tasks: Iterable[InferenceTask]) -> None:
        for task in tasks:
            if task.group_id is None:
                raise ValueError(f"task {task.task_id} has no assigned group_id before dispatch")
            group = self.topology.get_group(task.group_id)
            task.parallel_spec = group.parallel_spec
            self._finalize_task_dispatch(task)

    def _finalize_task_dispatch(self, task: InferenceTask) -> None:
        """Assemble inline inputs, apply refcount/release accounting, and dispatch.
        Called once all of the task's inputs are ready on the task's own group."""
        inline_inputs: list[ArtifactValue] = []
        release_after_exec_artifact_ids: list[str] = []
        for artifact in task.inputs:
            stored = self.artifact_store.get(artifact)
            if not isinstance(stored, WorkerLocalArtifactRef):
                inline_inputs.append(ArtifactValue(handle=artifact, value=stored))
            key = (artifact.request_id, artifact.artifact_id)
            remaining = self.artifact_refcounts.get(key)
            if remaining is not None:
                remaining -= 1
                if remaining <= 0:
                    self.artifact_refcounts.pop(key, None)
                    release_after_exec_artifact_ids.append(artifact.artifact_id)
                else:
                    self.artifact_refcounts[key] = remaining
        if task.step_range is not None:
            logger.debug(
                "runtime_v2 dispatch: request_id=%s task_id=%s kind=%s group=%s step_range=[%s,%s)",
                task.request_id,
                task.task_id,
                task.kind,
                task.group_id,
                task.step_range.start,
                task.step_range.end,
            )
        else:
            logger.debug(
                "runtime_v2 dispatch: request_id=%s task_id=%s kind=%s group=%s",
                task.request_id,
                task.task_id,
                task.kind,
                task.group_id,
            )
        self.worker_pool.dispatch(
            task=task,
            inline_inputs=tuple(inline_inputs),
            release_after_exec_artifact_ids=tuple(release_after_exec_artifact_ids),
        )

    def _handle_worker_event(self, event: WorkerEvent) -> None:
        if event.kind == WorkerEventKind.REQUEST_FINISHED:
            if not self._mark_request_event_delivered(event.request_id, event.kind):
                return
            logger.debug("runtime_v2 request finished: request_id=%s group=%s", event.request_id, event.group_id)
            self._dispatch_tasks(self.policy.on_worker_event(event))
            return
        if event.kind == WorkerEventKind.REQUEST_FAILED:
            if not self._mark_request_event_delivered(event.request_id, event.kind):
                return
            self.failed_requests.setdefault(event.request_id, event.message or f"request {event.request_id} failed")
            logger.error(
                "runtime_v2 request failed: request_id=%s group=%s message=%s",
                event.request_id,
                event.group_id,
                event.message,
            )
            self._cleanup_request_worker_state(event.request_id)
            self._dispatch_tasks(self.policy.on_worker_event(event))
            return

        task = self.task_index.get(event.task_id)
        if task is None:
            return

        if event.kind == WorkerEventKind.TASK_EXEC_BEGIN:
            task.status = TaskStatus.RUNNING
            if task.step_range is not None:
                logger.debug(
                    "runtime_v2 task begin: request_id=%s task_id=%s kind=%s step_range=[%s,%s)",
                    task.request_id,
                    task.task_id,
                    task.kind,
                    task.step_range.start,
                    task.step_range.end,
                )
            else:
                logger.debug(
                    "runtime_v2 task begin: request_id=%s task_id=%s kind=%s",
                    task.request_id,
                    task.task_id,
                    task.kind,
                )
            return

        if event.kind == WorkerEventKind.TASK_LAUNCH_END:
            self._dispatch_tasks(self.policy.on_worker_event(event))
            published_outputs = event.metadata.get("published_outputs", ())
            for artifact_ref in published_outputs:
                self.artifact_store.put(artifact_ref.handle, artifact_ref)

            # Skip dependent scheduling if the request already failed/was retired:
            # decrementing or dispatching downstream tasks of a dead request would
            # offer them to the policy (and the bookkeeping may already be gone).
            newly_runnable: list[InferenceTask] = []
            if task.request_id not in self.failed_requests:
                for dependent_id in self.dependents.get(task.task_id, []):
                    if dependent_id not in self.pending_dependencies:
                        continue
                    self.pending_dependencies[dependent_id] -= 1
                    if self.pending_dependencies[dependent_id] == 0:
                        newly_runnable.append(self.task_index[dependent_id])
            self._dispatch_tasks(self.policy.on_tasks_runnable(newly_runnable))
            return

        if event.kind == WorkerEventKind.TASK_EXEC_END:
            task.status = TaskStatus.FINISHED
            logger.debug(
                "runtime_v2 task end: request_id=%s task_id=%s kind=%s",
                task.request_id,
                task.task_id,
                task.kind,
            )
            self._dispatch_tasks(self.policy.on_worker_event(event))
            if task.task_id in self.plans[task.request_id].terminal_task_ids:
                # Keep the FCFS slot until the terminal output has been copied
                # and the worker-local request state has been evicted. The fetch
                # runs on a worker background thread, so merely enqueueing it
                # before the next dispatch would not provide that ordering.
                self._try_start_request_output_fetch(task.request_id)
            return

        if event.kind == WorkerEventKind.TASK_FAILED:
            if task.task_id in self.failed_tasks:
                return
            self.failed_tasks.add(task.task_id)
            task.status = TaskStatus.FAILED
            self.failed_requests.setdefault(task.request_id, event.message or f"task {task.task_id} failed")
            logger.error(
                "runtime_v2 task failed: request_id=%s task_id=%s kind=%s message=%s",
                task.request_id,
                task.task_id,
                task.kind,
                event.message,
            )
            self._dispatch_tasks(self.policy.on_worker_event(event))
            self._cleanup_request_worker_state(task.request_id)
            self._emit_request_event(
                request_id=task.request_id,
                group_id=event.group_id,
                kind=WorkerEventKind.REQUEST_FAILED,
                message=event.message,
            )

    def _try_collect_request_output(self, request_id: str) -> Any | None:
        if request_id in self._completed_outputs:
            return self._completed_outputs[request_id]
        # Use _completed_outputs membership (not "output is not None") as the
        # finished signal: a fetch can legitimately complete with a None value, and
        # the finish path pops the plan, so we must NOT fall through to
        # _try_start_request_output_fetch afterwards (it would KeyError on the plan).
        self._try_finish_request_output_fetch(request_id)
        if request_id in self._completed_outputs:
            return self._completed_outputs[request_id]
        if request_id not in self.plans:
            return None
        self._try_start_request_output_fetch(request_id)
        if request_id in self._completed_outputs:
            return self._completed_outputs[request_id]
        return None

    def _fail_request_from_fetch_error(self, request_id: str, message: str) -> None:
        # Record an output-fetch failure as a terminal "failed" status (mirroring
        # a REQUEST_FAILED worker event) and free the request's controller state.
        logger.error("runtime_v2 output fetch failed: request_id=%s error=%s", request_id, message)
        group_id = self._resolve_request_group_id(request_id) or ""
        self.failed_requests.setdefault(request_id, message)
        self._cleanup_request_worker_state(request_id)
        self._emit_request_event(
            request_id=request_id,
            group_id=group_id,
            kind=WorkerEventKind.REQUEST_FAILED,
            message=message,
        )

    def _try_start_request_output_fetch(self, request_id: str) -> Any | None:
        plan = self.plans[request_id]
        unfinished_terminal = [
            task_id for task_id in plan.terminal_task_ids if self.task_index[task_id].status != TaskStatus.FINISHED
        ]
        if unfinished_terminal:
            return None

        if request_id in self.pending_output_fetches:
            return None

        terminal_task = self.task_index[plan.terminal_task_ids[0]]
        if not terminal_task.outputs:
            return None
        output_handle = terminal_task.outputs[0]
        stored = self.artifact_store.get(output_handle)

        if isinstance(stored, WorkerLocalArtifactRef):
            if not hasattr(self.worker_pool, "start_fetch_artifacts"):
                fetch = self.worker_pool.fetch_artifacts(
                    request_id=request_id,
                    group_id=stored.group_id,
                    artifact_ids=(output_handle.artifact_id,),
                )
                if fetch.error:
                    self._fail_request_from_fetch_error(request_id, str(fetch.error))
                    return None
                if not fetch.artifacts:
                    self._fail_request_from_fetch_error(
                        request_id,
                        f"runtime_v2 fetch returned empty artifacts: request_id={request_id}",
                    )
                    return None
                artifact_value = fetch.artifacts[0]
                self.artifact_store.put(artifact_value.handle, artifact_value.value)
                self._completed_outputs[request_id] = artifact_value.value
                group_id = stored.group_id
                self._cleanup_request_worker_state(request_id)
                self._emit_request_event(
                    request_id=request_id,
                    group_id=group_id,
                    kind=WorkerEventKind.REQUEST_FINISHED,
                )
                return artifact_value.value

            fetch_start_ns = time.monotonic_ns()
            fetch_id = self.worker_pool.start_fetch_artifacts(
                request_id=request_id,
                group_id=stored.group_id,
                artifact_ids=(output_handle.artifact_id,),
            )
            self.pending_output_fetches[request_id] = _PendingOutputFetch(
                fetch_id=fetch_id,
                output_handle=output_handle,
                group_id=stored.group_id,
                started_ns=fetch_start_ns,
            )
            logger.debug(
                "runtime_v2 output fetch started: request_id=%s group=%s artifact_id=%s fetch_id=%s",
                request_id,
                stored.group_id,
                output_handle.artifact_id,
                fetch_id,
            )
            return None

        group_id = terminal_task.group_id or ""
        self._completed_outputs[request_id] = stored
        self._cleanup_request_worker_state(request_id)
        self._emit_request_event(
            request_id=request_id,
            group_id=group_id,
            kind=WorkerEventKind.REQUEST_FINISHED,
        )
        return stored

    def _try_finish_request_output_fetch(self, request_id: str) -> Any | None:
        pending = self.pending_output_fetches.get(request_id)
        if pending is None:
            return None
        if not hasattr(self.worker_pool, "poll_fetch_artifacts"):
            return None

        fetch = self.worker_pool.poll_fetch_artifacts(pending.fetch_id)
        if fetch is None:
            if pending.started_ns and (time.monotonic_ns() - pending.started_ns) > int(_OUTPUT_FETCH_TIMEOUT_S * 1e9):
                # Terminal task finished but the fetch never came back within the
                # deadline (worker fetch wedged). Fail the request instead of
                # polling forever -- _fail_request_from_fetch_error ->
                # _cleanup_request_worker_state pops pending_output_fetches and
                # discards the fetch.
                self._fail_request_from_fetch_error(
                    request_id,
                    f"runtime_v2 output fetch timed out after {_OUTPUT_FETCH_TIMEOUT_S:.0f}s: "
                    f"request_id={request_id} fetch_id={pending.fetch_id}",
                )
                return None
            return None

        self.pending_output_fetches.pop(request_id, None)
        if fetch.error:
            self._fail_request_from_fetch_error(request_id, str(fetch.error))
            return None
        if not fetch.artifacts:
            self._fail_request_from_fetch_error(
                request_id,
                f"runtime_v2 fetch returned empty artifacts: request_id={request_id} fetch_id={pending.fetch_id}",
            )
            return None
        artifact_value = fetch.artifacts[0]
        self.artifact_store.put(artifact_value.handle, artifact_value.value)
        self._completed_outputs[request_id] = artifact_value.value
        self._cleanup_request_worker_state(request_id)
        self._emit_request_event(
            request_id=request_id,
            group_id=pending.group_id,
            kind=WorkerEventKind.REQUEST_FINISHED,
        )
        logger.debug(
            "runtime_v2 output fetch completed: request_id=%s group=%s artifact_id=%s fetch_id=%s",
            request_id,
            pending.group_id,
            pending.output_handle.artifact_id,
            pending.fetch_id,
        )
        return artifact_value.value

    def _mark_request_event_delivered(self, request_id: str, kind: WorkerEventKind) -> bool:
        key = f"{request_id}:{kind.value}"
        if key in self.released_requests:
            return False
        self.released_requests.add(key)
        return True

    def _emit_request_event(
        self,
        *,
        request_id: str,
        group_id: str,
        kind: WorkerEventKind,
        message: str = "",
    ) -> None:
        if not self._mark_request_event_delivered(request_id, kind):
            return
        self._dispatch_tasks(
            self.policy.on_worker_event(
                WorkerEvent(
                    event_id=f"request:{request_id}:{kind.value}",
                    task_id="",
                    request_id=request_id,
                    group_id=group_id,
                    worker_rank=-1,
                    kind=kind,
                    timestamp_ns=time.monotonic_ns(),
                    message=message,
                )
            )
        )

    def _cleanup_request_worker_state(self, request_id: str) -> None:
        if request_id in self.cleaned_requests:
            return
        self.cleaned_requests.add(request_id)
        pending = self.pending_output_fetches.pop(request_id, None)
        if pending is not None and hasattr(self.worker_pool, "discard_fetch"):
            self.worker_pool.discard_fetch(pending.fetch_id)
        for key in self.request_artifacts.pop(request_id, set()):
            self.artifact_refcounts.pop(key, None)
        self.worker_pool.evict_request(request_id)
        # Free controller-side per-request state. The terminal output (if any) has
        # already been copied into self._completed_outputs by the caller, so evicting
        # the artifact store here cannot strand a not-yet-delivered result; a later
        # get_request_status poll is served from _completed_outputs. Without this the
        # store, plan, and task bookkeeping grew unboundedly for the process lifetime.
        self.artifact_store.evict_request(request_id)
        plan = self.plans.pop(request_id, None)
        if plan is not None:
            for task_id in plan.tasks:
                self.failed_tasks.discard(task_id)
                self.task_index.pop(task_id, None)
                self.pending_dependencies.pop(task_id, None)
                self.dependents.pop(task_id, None)

    def _clear_request_tombstones(self, request_id: str) -> None:
        """Drop small dedupe markers once no late task can reference a request."""
        self.cleaned_requests.discard(request_id)
        for kind in (WorkerEventKind.REQUEST_FINISHED, WorkerEventKind.REQUEST_FAILED):
            self.released_requests.discard(f"{request_id}:{kind.value}")

    def abort_request(self, request_id: str) -> None:
        """Abort an in-flight request, freeing its policy slot AND controller state.

        A bare :meth:`release_request` only frees controller-side bookkeeping; it
        does NOT tell the policy the request is gone. For an FCFS group that
        leaves ``active_request_by_group`` pinned to the aborted id, so the next
        request for that group is parked in ``pending_requests_by_group`` forever
        (a single-group deployment ⇒ whole-group deadlock).

        The normal terminal path frees the policy slot by feeding a REQUEST_FAILED
        ``WorkerEvent`` to ``policy.on_worker_event`` (which pops the active slot
        and promotes the next pending request, returning its now-dispatchable
        tasks). Reuse EXACTLY that machinery here: emit a synthetic REQUEST_FAILED
        event for this request's bound group (dispatching any promoted tasks),
        THEN free controller state (what ``release_request`` does).

        Guarded to a no-op for an unknown / already-finished / already-aborted id:
        if the request has no plan and no live bookkeeping it is either never seen
        or already retired, so there is no slot to free.
        """
        group_id = self._resolve_request_group_id(request_id)
        # No live state for this id: unknown or already retired -> no-op. We
        # still fall through to release_request()'s cleanup below only when
        # there is something to clean, so guard here.
        if (
            request_id not in self.plans
            and request_id not in self.failed_requests
            and request_id not in self._completed_outputs
            and request_id not in self.pending_output_fetches
            and group_id is None
        ):
            return
        # Retire controller state + EVICT the aborted request's worker-local
        # artifacts FIRST, BEFORE the REQUEST_FAILED event below promotes and
        # dispatches the next queued request. Worker command pipes are FIFO,
        # so queuing the EvictRequestCommand ahead of the promoted request's
        # dispatch makes the worker free the aborted latents/output before it
        # runs the next request -- otherwise they stay resident through it and
        # can cause an avoidable OOM. (Ordering only matters vs. the promoted
        # dispatch; the two operate on different request ids, so it is safe.)
        self._cleanup_request_worker_state(request_id)
        self._completed_outputs.pop(request_id, None)
        self.failed_requests.pop(request_id, None)
        # Free the policy's active slot + promote any queued request for the
        # group, dispatching its newly-runnable tasks -- the SAME path the
        # normal REQUEST_FAILED terminal uses. group_id may be "" ; the policy
        # falls back to its bound request_group mapping in that case.
        self._emit_request_event(
            request_id=request_id,
            group_id=group_id or "",
            kind=WorkerEventKind.REQUEST_FAILED,
            message=f"request {request_id} aborted.",
        )
        self._clear_request_tombstones(request_id)

    def _resolve_request_group_id(self, request_id: str) -> str | None:
        """Best-effort lookup of a request's bound execution group id.

        Prefers the policy's ``request_group`` mapping (authoritative once the
        request was admitted); falls back to any task's ``group_id`` on the plan.
        Returns ``None`` when nothing is known about the request.
        """
        policy_map = getattr(self.policy, "request_group", None)
        if isinstance(policy_map, dict):
            group_id = policy_map.get(request_id)
            if group_id is not None:
                return group_id
        plan = self.plans.get(request_id)
        if plan is not None:
            for task in plan.tasks.values():
                if task.group_id is not None:
                    return task.group_id
        return None

    def release_request(self, request_id: str) -> None:
        """Retire a request once its terminal status has been delivered to the
        caller. Frees the large per-request state (plan, artifact store, task
        bookkeeping, cached output, and temporary event-deduplication markers.
        Idempotent and safe to call more than once."""
        self._cleanup_request_worker_state(request_id)
        self._completed_outputs.pop(request_id, None)
        self.failed_requests.pop(request_id, None)
        self._clear_request_tombstones(request_id)


@dataclass(frozen=True)
class _PendingOutputFetch:
    fetch_id: str
    output_handle: ArtifactHandle
    group_id: str
    started_ns: int = 0
