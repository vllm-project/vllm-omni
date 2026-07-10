# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections import deque
from collections.abc import Iterable

from vllm.logger import init_logger

from vllm_omni.diffusion.runtime_v2.interfaces import SchedulerPolicy
from vllm_omni.diffusion.runtime_v2.protocol import (
    InferenceTask,
    RequestExecutionPlan,
    TaskStatus,
    WorkerEvent,
    WorkerEventKind,
)
from vllm_omni.diffusion.runtime_v2.topology import RuntimeTopology

logger = init_logger(__name__)


class FCFSSchedulerPolicy(SchedulerPolicy):
    """First-come-first-served admission with one active request per execution group.

    PR1 is single-group: a request is bound to the one group that supports every
    task kind in its plan, and only one request per group runs at a time; the rest
    queue in arrival order. (Cost-model / multi-group scheduling is deferred.)
    """

    def __init__(self, topology: RuntimeTopology) -> None:
        self.topology = topology
        self.ready_queue: deque[InferenceTask] = deque()
        self.active_request_by_group: dict[str, str] = {}
        self.pending_requests_by_group: dict[str, deque[str]] = {}
        self.blocked_tasks_by_request: dict[str, list[InferenceTask]] = {}
        self.request_group: dict[str, str] = {}

    def on_request_submitted(self, plan: RequestExecutionPlan) -> Iterable[InferenceTask]:
        # Bind once at ingress so all downstream tasks inherit the same group_id
        # and execution never crosses groups.
        self._bind_request_group(plan)
        root_tasks = [task for task in plan.tasks.values() if not task.dependencies]
        return self.on_tasks_runnable(root_tasks)

    def on_tasks_runnable(self, tasks: Iterable[InferenceTask]) -> Iterable[InferenceTask]:
        for task in tasks:
            if task.group_id is None:
                task.group_id = self.request_group.get(task.request_id)
            if task.group_id is None:
                raise ValueError(f"request {task.request_id} has runnable task without bound group_id: {task.task_id}")
            group_id = task.group_id
            self.request_group.setdefault(task.request_id, group_id)

            active_request = self.active_request_by_group.get(group_id)
            if active_request is None:
                self.active_request_by_group[group_id] = task.request_id
                active_request = task.request_id

            if active_request == task.request_id:
                task.status = TaskStatus.READY
                self.ready_queue.append(task)
                continue

            # Group is busy with another request -> park this one in arrival order.
            queue_for_group = self.pending_requests_by_group.setdefault(group_id, deque())
            if task.request_id not in queue_for_group:
                queue_for_group.append(task.request_id)
            self.blocked_tasks_by_request.setdefault(task.request_id, []).append(task)

        return self._take_dispatchable_tasks()

    def on_worker_event(self, event: WorkerEvent) -> Iterable[InferenceTask]:
        if event.kind in (WorkerEventKind.REQUEST_FINISHED, WorkerEventKind.REQUEST_FAILED):
            request_id = event.request_id
            # Fall back to the bound group if the event omits group_id, so the
            # active slot is always released and queued requests are promoted.
            group_id = event.group_id or self.request_group.get(request_id)
            was_active = self.active_request_by_group.get(group_id) == request_id
            if was_active:
                del self.active_request_by_group[group_id]
            else:
                # A queued (non-active) request terminated -- e.g. it was aborted
                # while parked behind the running request. Remove it from the
                # pending queue WITHOUT promoting: promoting here would overwrite
                # the active slot with this now-dead request while the real active
                # request is still running.
                queue_for_group = self.pending_requests_by_group.get(group_id)
                if queue_for_group is not None:
                    try:
                        queue_for_group.remove(request_id)
                    except ValueError:
                        pass
                    if not queue_for_group:
                        self.pending_requests_by_group.pop(group_id, None)

            # Promote the next queued request ONLY when the group's active slot is
            # now free (never while another request is still active).
            if group_id is not None and self.active_request_by_group.get(group_id) is None:
                queue_for_group = self.pending_requests_by_group.get(group_id)
                if queue_for_group:
                    while queue_for_group:
                        next_request_id = queue_for_group.popleft()
                        blocked_tasks = self.blocked_tasks_by_request.pop(next_request_id, [])
                        if not blocked_tasks:
                            continue
                        self.active_request_by_group[group_id] = next_request_id
                        for task in blocked_tasks:
                            task.status = TaskStatus.READY
                            self.ready_queue.append(task)
                        break
                    if not queue_for_group:
                        self.pending_requests_by_group.pop(group_id, None)
            self._cleanup_request(request_id)

        return self._take_dispatchable_tasks()

    def _bind_request_group(self, plan: RequestExecutionPlan) -> str:
        # Select the (single, in PR1) group that supports every task kind in the plan.
        candidate_group_ids = [
            group.group_id
            for group in self.topology.groups
            if all(task.kind in group.supported_task_kinds for task in plan.tasks.values())
        ]
        if not candidate_group_ids:
            task_kinds = sorted({task.kind.value for task in plan.tasks.values()})
            raise ValueError(f"no execution group supports all request task kinds: {task_kinds!r}")
        selected_group_id = candidate_group_ids[0]
        self.request_group[plan.request_id] = selected_group_id
        for task in plan.tasks.values():
            if task.group_id is None:
                task.group_id = selected_group_id
        return selected_group_id

    def _cleanup_request(self, request_id: str) -> None:
        self.request_group.pop(request_id, None)
        self.blocked_tasks_by_request.pop(request_id, None)

    def _take_dispatchable_tasks(self) -> list[InferenceTask]:
        dispatchable: list[InferenceTask] = []
        while self.ready_queue:
            task = self.ready_queue.popleft()
            task.status = TaskStatus.DISPATCHED
            dispatchable.append(task)
        return dispatchable
