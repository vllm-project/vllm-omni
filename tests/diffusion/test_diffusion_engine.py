# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import queue
import threading
import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest


@dataclass
class DiffusionSchedulerOutput:
    step_id: int
    scheduled_new_reqs: list = field(default_factory=list)
    scheduled_cached_reqs: Any = None
    finished_req_ids: set = field(default_factory=set)
    num_running_reqs: int = 0
    num_waiting_reqs: int = 0

    @property
    def scheduled_req_ids(self):
        ids = [req.request_id for req in self.scheduled_new_reqs]
        if self.scheduled_cached_reqs:
            ids.extend(self.scheduled_cached_reqs.sched_req_ids)
        return ids

    @property
    def is_empty(self):
        return len(self.scheduled_req_ids) == 0


class MockScheduler:
    def __init__(self):
        self._waiting_queue = []
        self._step_id = 0

    def add_request(self, request):
        self._waiting_queue.append(request)
        return request.request_id

    def has_requests(self):
        return len(self._waiting_queue) > 0

    def schedule(self) -> DiffusionSchedulerOutput:
        if not self._waiting_queue:
            return DiffusionSchedulerOutput(step_id=self._step_id)

        batch = []
        while self._waiting_queue:
            req = self._waiting_queue.pop(0)
            batch.append(SimpleNamespace(request_id=req.request_id))

        output = DiffusionSchedulerOutput(step_id=self._step_id, scheduled_new_reqs=batch)
        self._step_id += 1
        return output

    def update_from_output(self, sched_output, runner_output):
        # assume all new req finished
        return [req.request_id for req in sched_output.scheduled_new_reqs]


@pytest.mark.asyncio
async def test_async_add_req_and_wait_for_response():
    from vllm_omni.diffusion.diffusion_engine import DiffusionEngine

    engine = object.__new__(DiffusionEngine)
    engine.scheduler = MockScheduler()
    engine._out_queue = {}
    engine.abort_queue: queue.Queue[str] = queue.Queue()
    engine._rpc_lock = threading.Lock()

    engine._finalize_finished_request = lambda rid, out, err: out.result

    def mock_execute_batch(sched_output):
        req_ids = sched_output.scheduled_req_ids

        time.sleep(1)

        class MockRunnerOutput:
            def __init__(self, ids):
                self.req_id = ids
                self.step_index = 0
                self.finished = True
                self._results = {rid: SimpleNamespace(result_data=f"data_{rid}") for rid in ids}

            def get_req_output(self, rid):
                return SimpleNamespace(result=self._results[rid], step_index=0, finished=True)

        return MockRunnerOutput(req_ids)

    engine.execute_fn = mock_execute_batch

    engine.stop_event = threading.Event()
    engine.start_background_loop()

    async def run_task(rid):
        req = SimpleNamespace(request_id=rid)
        start = time.time()
        res = await engine.async_add_req_and_wait_for_response(req)
        return rid, res, time.time() - start

    task_ids = [f"req_{i}" for i in range(5)]
    tasks = [run_task(rid) for rid in task_ids]
    results = await asyncio.gather(*tasks)

    engine.stop_event.set()
    engine.worker_thread.join(timeout=5)
    assert len(results) == 5
    for rid, res, elapsed in results:
        assert rid in res.result_data

    eps = 0.5
    latencies = [r[2] for r in results]
    assert max(latencies) - min(latencies) < eps
