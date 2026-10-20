# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import threading
from types import SimpleNamespace

import pytest

from tests.diffusion.test_diffusion_engine_rpc_routing import _make_engine_with_loop, _stop_engine
from tests.diffusion.test_diffusion_scheduler import _make_step_request
from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.diffusion_engine import DiffusionExecutionMode
from vllm_omni.diffusion.sched import DiffusionRequestStatus, StepScheduler
from vllm_omni.diffusion.worker.utils import BatchRunnerOutput, RunnerOutput, StepWait

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _scheduler():
    scheduler = StepScheduler()
    scheduler.initialize(SimpleNamespace(max_num_seqs=1))
    scheduler.add_request(_make_step_request("held"))
    return scheduler


def _chunk(wait=StepWait()):
    return RunnerOutput(request_id="held", step_index=3, result=DiffusionOutput(), wait_for_interaction=wait)


def test_indefinite_wait_retains_admission_and_wakes_without_a_fake_step():
    scheduler = _scheduler()
    scheduler.update_from_output(scheduler.schedule(), _chunk())
    scheduler.add_request(_make_step_request("queued"))
    assert scheduler.get_execution_wait_timeout(1e12) is None
    assert scheduler.schedule().scheduled_request_ids == []
    assert scheduler._running == ["held"]
    scheduler.notify_interaction("held")
    resumed = scheduler.schedule()
    assert resumed.scheduled_request_ids == ["held"]
    assert resumed.step_resumes == {"held": "interaction"}
    assert scheduler.get_execution_wait_timeout(1e12) == 0


def test_interaction_before_output_update_is_not_lost():
    scheduler = _scheduler()
    scheduled = scheduler.schedule()
    scheduler.notify_interaction("held")
    scheduler.update_from_output(scheduled, _chunk())
    assert scheduler.schedule().step_resumes == {"held": "interaction"}
    # A previously consumed interaction must not wake the following window.
    scheduler.update_from_output(scheduler.schedule(), _chunk())
    assert scheduler.get_execution_wait_timeout(1e12) is None


@pytest.mark.parametrize("accepted_prompt", [False, True])
def test_host_deadline_and_accepted_prompt_priority(monkeypatch, accepted_prompt):
    now = [100.0]
    monkeypatch.setattr("vllm_omni.diffusion.sched.step_scheduler.time.monotonic", lambda: now[0])
    scheduler = _scheduler()
    scheduler.update_from_output(scheduler.schedule(), _chunk(StepWait(2)))
    assert scheduler.get_execution_wait_timeout(100.5) == 1.5
    now[0] = 103
    if accepted_prompt:
        scheduler.notify_interaction("held")
    resumed = scheduler.schedule()
    assert resumed.step_resumes == {"held": "interaction" if accepted_prompt else "timeout"}
    # Once sent to the Worker, a late prompt cannot change the chosen reason.
    scheduler.notify_interaction("held")
    assert resumed.step_resumes == {"held": "interaction" if accepted_prompt else "timeout"}


def test_abort_discards_wait_and_frees_admission():
    scheduler = _scheduler()
    scheduler.update_from_output(scheduler.schedule(), _chunk())
    scheduler.finish_requests("held", DiffusionRequestStatus.FINISHED_ABORTED)
    scheduler.notify_interaction("held")
    scheduler.pop_request_state("held")
    scheduler.add_request(_make_step_request("queued"))
    assert scheduler.schedule().scheduled_request_ids == ["queued"]
    assert "held" not in scheduler._request_progress
    assert scheduler.get_execution_wait_timeout(1e12) == 0


def test_finite_deadline_wakes_only_its_request_among_indefinite_waits(monkeypatch):
    now = [100.0]
    monkeypatch.setattr("vllm_omni.diffusion.sched.step_scheduler.time.monotonic", lambda: now[0])
    scheduler = _scheduler()
    scheduler.max_num_running_reqs = 2
    scheduler.add_request(_make_step_request("finite"))
    scheduled = scheduler.schedule()
    scheduler.update_from_output(
        scheduled,
        BatchRunnerOutput.from_list([_chunk(), RunnerOutput("finite", step_index=3, wait_for_interaction=StepWait(2))]),
    )
    assert scheduler.get_execution_wait_timeout(now[0]) == 2
    now[0] += 2
    resumed = scheduler.schedule()
    assert resumed.scheduled_request_ids == ["finite"] and resumed.step_resumes == {"finite": "timeout"}
    scheduler.finish_requests("finite", DiffusionRequestStatus.FINISHED_COMPLETED)
    scheduler.schedule()  # Deliver terminal bookkeeping before waiting again.
    assert scheduler.get_execution_wait_timeout(now[0]) is None
    scheduler.notify_interaction("held")
    assert scheduler.schedule().step_resumes == {"held": "interaction"}


@pytest.mark.parametrize("wake", ["interaction", "timeout", "abort", "shutdown"])
def test_engine_sleeps_without_executing_held_request(monkeypatch, wake):
    scheduler = _scheduler()
    engine = _make_engine_with_loop(None, scheduler=scheduler, start_loop=False)
    engine.execution_mode = DiffusionExecutionMode.STEP_BATCH
    engine._request_cancellations = None
    sleeping = threading.Event()
    resumed = threading.Event()
    executions = []
    condition_wait = engine._cv.wait

    def observe_wait(timeout=None):
        progress = scheduler._request_progress.get("held")
        if progress is not None and progress.wait_deadline is not None:
            sleeping.set()
        return condition_wait(timeout)

    monkeypatch.setattr(engine._cv, "wait", observe_wait)

    def execute(output):
        executions.append(output)
        if len(executions) == 1:
            return BatchRunnerOutput.from_list([_chunk(StepWait(0.1) if wake == "timeout" else StepWait())])
        resumed.set()
        return BatchRunnerOutput.from_list([RunnerOutput("held", step_index=3, finished=True)])

    engine.execute_fn = execute
    engine._loop_started = True
    engine.worker_thread = threading.Thread(target=engine._busy_loop, daemon=True)
    engine.worker_thread.start()
    try:
        assert sleeping.wait(5), "Engine did not enter the event wait"
        assert len(executions) == 1
        if wake == "interaction":
            engine.collective_rpc("submit_interaction", args=("held", {"event": {"prompt": "new"}}), timeout=5)
        elif wake == "abort":
            engine.abort("held")
            # A following RPC is a barrier: aborts are drained first.
            engine.collective_rpc("barrier", timeout=5)
            assert any(call["method"] == "release_step_requests" for call in engine.executor.calls)
        if wake in {"interaction", "timeout"}:
            assert resumed.wait(5)
            assert len(executions) == 2
            assert executions[-1].step_resumes == {"held": wake}
        else:
            assert len(executions) == 1
    finally:
        _stop_engine(engine)


def test_failed_interaction_rpc_does_not_wake(monkeypatch):
    scheduler = _scheduler()
    scheduler.update_from_output(scheduler.schedule(), _chunk())
    engine = _make_engine_with_loop(None, scheduler=scheduler, start_loop=False)

    def reject(**kwargs):
        raise ValueError("invalid prompt")

    monkeypatch.setattr(engine.executor, "collective_rpc", reject)
    task = engine._submit_rpc("submit_interaction", None, ("held", {}), None, None)
    engine._process_rpc_queue()
    with pytest.raises(ValueError, match="invalid prompt"):
        task.future.result()
    assert scheduler.get_execution_wait_timeout(1e12) is None
    assert scheduler._request_progress["held"].interaction_generation == 0
