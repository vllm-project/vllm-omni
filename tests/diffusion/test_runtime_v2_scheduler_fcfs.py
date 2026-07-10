# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm_omni.diffusion.runtime_v2.policies.fcfs import FCFSSchedulerPolicy
from vllm_omni.diffusion.runtime_v2.protocol import (
    ArtifactHandle,
    ArtifactKind,
    ArtifactLayout,
    ArtifactValue,
    InferenceTask,
    ParallelSpec,
    RequestExecutionPlan,
    TaskKind,
    WorkerEvent,
    WorkerEventKind,
    WorkerLocalArtifactRef,
)
from vllm_omni.diffusion.runtime_v2.scheduler import GlobalScheduler, InMemoryArtifactStore
from vllm_omni.diffusion.runtime_v2.topology import RuntimeTopology

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


class _TwoTaskCompiler:
    def compile_request(self, request):
        state = ArtifactHandle(
            request_id="r",
            artifact_id="state",
            kind=ArtifactKind.REQUEST_STATE,
            layout=ArtifactLayout.WORKER_LOCAL,
        )
        output = ArtifactHandle(
            request_id="r",
            artifact_id="output",
            kind=ArtifactKind.OUTPUT,
            layout=ArtifactLayout.WORKER_LOCAL,
        )
        prepare = InferenceTask(
            task_id="r:prepare",
            request_id="r",
            kind=TaskKind.TEXT_ENCODE,
            group_id="g0",
            parallel_spec=ParallelSpec(),
            outputs=(state,),
        )
        finish = InferenceTask(
            task_id="r:finish",
            request_id="r",
            kind=TaskKind.VAE_DECODE,
            group_id="g0",
            parallel_spec=ParallelSpec(),
            dependencies=(prepare.task_id,),
            inputs=(state,),
            outputs=(output,),
        )
        return RequestExecutionPlan(
            request_id="r",
            tasks={prepare.task_id: prepare, finish.task_id: finish},
            terminal_task_ids=(finish.task_id,),
        )


class _EchoPool:
    def __init__(self):
        self.events = []
        self.values = {}

    def dispatch(self, task, inline_inputs, release_after_exec_artifact_ids=()):
        references = tuple(
            WorkerLocalArtifactRef(handle=handle, group_id="g0", worker_rank=0) for handle in task.outputs
        )
        for handle in task.outputs:
            self.values[(task.request_id, handle.artifact_id)] = f"value:{handle.artifact_id}"
        for kind, metadata in (
            (WorkerEventKind.TASK_LAUNCH_END, {"published_outputs": references}),
            (WorkerEventKind.TASK_EXEC_END, {}),
        ):
            self.events.append(
                WorkerEvent(
                    event_id=f"{task.task_id}:{kind.value}",
                    task_id=task.task_id,
                    request_id=task.request_id,
                    group_id="g0",
                    worker_rank=0,
                    kind=kind,
                    timestamp_ns=0,
                    metadata=metadata,
                )
            )

    def poll(self, timeout_s=0.0):
        events, self.events = self.events, []
        return events

    def fetch_artifacts(self, request_id, group_id, artifact_ids):
        artifacts = tuple(
            ArtifactValue(
                handle=ArtifactHandle(
                    request_id=request_id,
                    artifact_id=artifact_id,
                    kind=ArtifactKind.OUTPUT,
                    layout=ArtifactLayout.WORKER_LOCAL,
                ),
                value=self.values[(request_id, artifact_id)],
            )
            for artifact_id in artifact_ids
        )
        return SimpleNamespace(artifacts=artifacts, error=None)

    def evict_request(self, request_id):
        pass


def _scheduler(pool, compiler):
    topology = RuntimeTopology.single_group(num_gpus=1, parallel_spec=ParallelSpec())
    return GlobalScheduler(
        topology=topology,
        worker_pool=pool,
        compiler=compiler,
        artifact_store=InMemoryArtifactStore(),
        policy=FCFSSchedulerPolicy(topology),
    )


def test_fcfs_executes_plan_and_fetches_terminal_output():
    scheduler = _scheduler(_EchoPool(), _TwoTaskCompiler())
    request_id = scheduler.submit_request(object())

    for _ in range(4):
        scheduler.poll_once()
        status, output = scheduler.get_request_status(request_id)
        if status == "finished":
            break

    assert (status, output) == ("finished", "value:output")

    scheduler.release_request(request_id)
    assert request_id not in scheduler.cleaned_requests
    assert not any(key.startswith(f"{request_id}:") for key in scheduler.released_requests)


class _SingleTaskCompiler:
    def compile_request(self, request):
        output = ArtifactHandle(
            request_id=request.request_id,
            artifact_id="output",
            kind=ArtifactKind.OUTPUT,
            layout=ArtifactLayout.WORKER_LOCAL,
        )
        task = InferenceTask(
            task_id=f"{request.request_id}:task",
            request_id=request.request_id,
            kind=TaskKind.TEXT_ENCODE,
            group_id="g0",
            parallel_spec=ParallelSpec(),
            outputs=(output,),
        )
        return RequestExecutionPlan(
            request_id=request.request_id,
            tasks={task.task_id: task},
            terminal_task_ids=(task.task_id,),
        )


class _RecordingPool:
    def __init__(self):
        self.calls = []

    def dispatch(self, task, inline_inputs, release_after_exec_artifact_ids=()):
        self.calls.append(("dispatch", task.request_id))

    def poll(self, timeout_s=0.0):
        return []

    def evict_request(self, request_id):
        self.calls.append(("evict", request_id))


class _AsyncFetchPool(_RecordingPool):
    def __init__(self):
        super().__init__()
        self.events = []
        self.fetch_ready = False
        self.output_handle = None

    def poll(self, timeout_s=0.0):
        events, self.events = self.events, []
        return events

    def finish(self, request_id):
        task_id = f"{request_id}:task"
        self.output_handle = ArtifactHandle(
            request_id=request_id,
            artifact_id="output",
            kind=ArtifactKind.OUTPUT,
            layout=ArtifactLayout.WORKER_LOCAL,
        )
        reference = WorkerLocalArtifactRef(handle=self.output_handle, group_id="g0", worker_rank=0)
        self.events.extend(
            [
                WorkerEvent(
                    event_id=f"{task_id}:launch",
                    task_id=task_id,
                    request_id=request_id,
                    group_id="g0",
                    worker_rank=0,
                    kind=WorkerEventKind.TASK_LAUNCH_END,
                    timestamp_ns=0,
                    metadata={"published_outputs": (reference,)},
                ),
                WorkerEvent(
                    event_id=f"{task_id}:end",
                    task_id=task_id,
                    request_id=request_id,
                    group_id="g0",
                    worker_rank=0,
                    kind=WorkerEventKind.TASK_EXEC_END,
                    timestamp_ns=1,
                ),
            ]
        )

    def start_fetch_artifacts(self, request_id, group_id, artifact_ids):
        self.calls.append(("start_fetch", request_id))
        return f"fetch:{request_id}"

    def poll_fetch_artifacts(self, fetch_id):
        if not self.fetch_ready:
            return None
        return SimpleNamespace(
            artifacts=(ArtifactValue(handle=self.output_handle, value="done"),),
            error=None,
        )

    def discard_fetch(self, fetch_id):
        pass


def test_abort_evicts_active_before_promoting_next_request():
    pool = _RecordingPool()
    scheduler = _scheduler(pool, _SingleTaskCompiler())
    scheduler.submit_request(SimpleNamespace(request_id="A"))
    scheduler.submit_request(SimpleNamespace(request_id="B"))
    pool.calls.clear()

    scheduler.abort_request("A")

    assert pool.calls == [("evict", "A"), ("dispatch", "B")]
    assert scheduler.policy.active_request_by_group == {"g0": "B"}


def test_abort_queued_request_keeps_active_request():
    pool = _RecordingPool()
    scheduler = _scheduler(pool, _SingleTaskCompiler())
    scheduler.submit_request(SimpleNamespace(request_id="A"))
    scheduler.submit_request(SimpleNamespace(request_id="B"))

    scheduler.abort_request("B")

    assert scheduler.policy.active_request_by_group == {"g0": "A"}
    assert not scheduler.policy.pending_requests_by_group.get("g0")
    assert pool.calls == [("dispatch", "A"), ("evict", "B")]


def test_terminal_fetch_finishes_before_evict_and_next_dispatch():
    pool = _AsyncFetchPool()
    scheduler = _scheduler(pool, _SingleTaskCompiler())
    scheduler.submit_request(SimpleNamespace(request_id="A"))
    scheduler.submit_request(SimpleNamespace(request_id="B"))
    pool.finish("A")

    scheduler.poll_once()
    assert scheduler.get_request_status("A") == ("pending", None)
    assert ("dispatch", "B") not in pool.calls

    pool.fetch_ready = True
    assert scheduler.get_request_status("A") == ("finished", "done")
    assert pool.calls.index(("evict", "A")) < pool.calls.index(("dispatch", "B"))


def test_start_forwards_worker_timeout():
    class _Pool:
        def start(self, timeout_s=600.0):
            self.timeout_s = timeout_s

    pool = _Pool()
    scheduler = _scheduler(pool, SimpleNamespace())
    scheduler.start(timeout_s=123.0)
    assert pool.timeout_s == 123.0
