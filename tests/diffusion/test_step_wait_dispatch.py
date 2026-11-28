# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Main-worker wait/resume dispatch for stepwise diffusion (RFC #11195 task 2).

The scheduler already owns the deadline and the wake reason; these tests cover
what the main worker owes around it: resume before batching and inside the
caller's binding, let only a successful non-terminal chunk wait again, and clean
up when a resume fails without letting a late wake revive a cancelled request.

They deliberately do not import the AR-Diffusion runner: wait/resume is a main
worker capability that any step pipeline may declare.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm_omni.diffusion.data import DiffusionKVCacheMode, DiffusionOutput
from vllm_omni.diffusion.models.interface import (
    step_resume_hook,
    step_wait_hook,
    supports_step_wait,
)
from vllm_omni.diffusion.sched.interface import CachedRequestData, DiffusionSchedulerOutput
from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner
from vllm_omni.diffusion.worker.utils import BatchRunnerOutput, RunnerOutput, StepWait

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

REQUEST_ID = "held"


class _WaitPipeline:
    """A step pipeline that declares the optional wait/resume capability."""

    def __init__(self, *, wait: StepWait | None, resume_error: Exception | None = None) -> None:
        self._wait = wait
        self._resume_error = resume_error
        self.resumed: list[tuple[object, str]] = []
        self.waited: list[object] = []
        self.bound_state: object | None = None

    def step_wait(self, state):
        self.waited.append(state)
        return self._wait

    def resume_step(self, state, reason):
        self.resumed.append((state, reason))
        if self._resume_error is not None:
            raise self._resume_error
        state.ready = True


def _runner(pipeline: object) -> DiffusionModelRunner:
    """A runner with only what the stepwise dispatch path touches."""
    runner = object.__new__(DiffusionModelRunner)
    runner.pipeline = pipeline
    runner.state_cache = {}
    runner.input_batch = None
    runner.od_config = SimpleNamespace(
        cache_backend=None,
        # The wave-level cleanup checks the KV mode even when nothing finished.
        diffusion_kv_mode=DiffusionKVCacheMode.DENSE_LEGACY,
        streaming_output=False,
        step_execution=True,
    )
    runner.release_calls: list[list[str]] = []
    runner.install_diffusion_kv_metadata = lambda metadata: None
    runner.remove_diffusion_kv_requests = lambda ids: None

    def release_step_requests(request_ids):
        runner.release_calls.append(list(request_ids))
        for request_id in request_ids:
            runner.state_cache.pop(request_id, None)

    runner.release_step_requests = release_step_requests
    return runner


def _scheduled(*request_ids: str, resumes: dict[str, str] | None = None) -> DiffusionSchedulerOutput:
    output = DiffusionSchedulerOutput(
        step_id=0,
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData(request_ids=list(request_ids)),
        finished_req_ids=set(),
        num_running_reqs=len(request_ids),
        num_waiting_reqs=0,
    )
    output.step_resumes.update(resumes or {})
    return output


@pytest.fixture()
def dispatch(monkeypatch):
    """Drive ``_execute_stepwise`` with the real dispatch and a stubbed core."""
    calls: list[dict] = []

    def run(runner, scheduled, *, core_result=None, core_raises=None):
        def core(self, scheduler_output, **kwargs):
            calls.append({"state": dict(self.state_cache)})
            if core_raises is not None:
                raise core_raises
            return BatchRunnerOutput.from_list(core_result or [])

        monkeypatch.setattr(DiffusionModelRunner, "_execute_stepwise_core", core)
        monkeypatch.setattr(DiffusionModelRunner, "_supports_step_mode", lambda self: True)
        return runner._execute_stepwise(
            scheduled,
            validate_kv_metadata=False,
            record_output_peak_memory=False,
        )

    return SimpleNamespace(run=run, calls=calls)


# ----------------------------------------------------------------------------
# capability detection


def test_hooks_are_found_under_both_the_neutral_and_legacy_names() -> None:
    neutral = _WaitPipeline(wait=StepWait())
    assert supports_step_wait(neutral)
    assert step_resume_hook(neutral) is not None
    assert step_wait_hook(neutral) is not None

    legacy = SimpleNamespace(
        ar_diffusion_step_wait=lambda state: None,
        ar_diffusion_resume_step=lambda state, reason: None,
    )
    assert supports_step_wait(legacy)
    assert step_wait_hook(legacy) is legacy.ar_diffusion_step_wait
    assert step_resume_hook(legacy) is legacy.ar_diffusion_resume_step

    # Half a contract is not a contract: a pipeline must declare both hooks.
    assert not supports_step_wait(SimpleNamespace(step_wait=lambda state: None))
    assert not supports_step_wait(SimpleNamespace())


def test_a_pipeline_without_the_capability_is_left_alone(dispatch) -> None:
    """A plain step pipeline must not gain a wait or a resume it never asked for."""
    runner = _runner(SimpleNamespace())
    runner.state_cache[REQUEST_ID] = SimpleNamespace(ready=False)
    plain_output = RunnerOutput(request_id=REQUEST_ID, result=DiffusionOutput())

    output = dispatch.run(runner, _scheduled(REQUEST_ID), core_result=[plain_output])

    assert output.get_request_output(REQUEST_ID).wait_for_interaction is None


# ----------------------------------------------------------------------------
# resume dispatch


@pytest.mark.parametrize("reason", ["interaction", "timeout"])
def test_resume_runs_before_the_core_builds_its_batch(dispatch, reason) -> None:
    pipeline = _WaitPipeline(wait=None)
    runner = _runner(pipeline)
    state = SimpleNamespace(ready=False)
    runner.state_cache[REQUEST_ID] = state
    result = RunnerOutput(request_id=REQUEST_ID, result=DiffusionOutput())

    dispatch.run(runner, _scheduled(REQUEST_ID, resumes={REQUEST_ID: reason}), core_result=[result])

    # The pipeline saw the reason, and the core saw the input resume installed.
    assert pipeline.resumed == [(state, reason)]
    assert dispatch.calls[0]["state"][REQUEST_ID].ready is True


def test_resume_is_dispatched_once_even_when_the_core_repeats_the_wake(dispatch) -> None:
    """A grouped-step runner re-enters the core with the same ``step_resumes``.

    Dispatching from inside the core would replay the wake, and a pipeline that has
    already cleared its held flag rejects the replay. Counting every dispatch call
    proves the placement rather than just the resulting count: one dispatch from
    the entry point, none from the core re-entry a continuation performs.
    """
    pipeline = _WaitPipeline(wait=None)
    runner = _runner(pipeline)
    state = SimpleNamespace(ready=False)
    runner.state_cache[REQUEST_ID] = state
    scheduled = _scheduled(REQUEST_ID, resumes={REQUEST_ID: "interaction"})

    dispatch_calls: list[str] = []
    real_dispatch = DiffusionModelRunner._dispatch_step_resumes

    def counting_dispatch(self, scheduler_output):
        dispatch_calls.append(scheduler_output.step_resumes.get(REQUEST_ID))
        return real_dispatch(self, scheduler_output)

    DiffusionModelRunner._dispatch_step_resumes = counting_dispatch
    try:
        dispatch.run(
            runner,
            scheduled,
            core_result=[RunnerOutput(REQUEST_ID, result=DiffusionOutput())],
        )
        after_entry = list(pipeline.resumed)

        # What a grouped runner does: re-enter the core with a continuation that
        # still carries ``step_resumes``.
        DiffusionModelRunner._execute_stepwise_core(runner, scheduled, record_output_peak_memory=False)
    finally:
        DiffusionModelRunner._dispatch_step_resumes = real_dispatch

    assert after_entry == [(state, "interaction")]
    # The core re-entry did not resume a second time.
    assert pipeline.resumed == [(state, "interaction")]
    # Exactly one dispatch happened, from the entry point.
    assert dispatch_calls == ["interaction"]


def test_a_late_wake_does_not_recreate_a_cancelled_request(dispatch) -> None:
    """The scheduler may report a wake after the request was released."""
    pipeline = _WaitPipeline(wait=None)
    runner = _runner(pipeline)
    # No cached state: the request was cancelled or released before this wave.
    result = RunnerOutput(request_id=REQUEST_ID, result=DiffusionOutput())

    dispatch.run(runner, _scheduled(REQUEST_ID, resumes={REQUEST_ID: "interaction"}), core_result=[result])

    assert pipeline.resumed == []
    # Nothing re-admitted it.
    assert REQUEST_ID not in runner.state_cache


def test_resume_without_a_pipeline_hook_is_an_error_not_a_silent_continue(dispatch) -> None:
    pipeline = SimpleNamespace(step_wait=lambda state: None)  # waits, but cannot resume
    runner = _runner(pipeline)
    runner.state_cache[REQUEST_ID] = SimpleNamespace(ready=False)

    with pytest.raises(RuntimeError, match="no step resume hook"):
        dispatch.run(runner, _scheduled(REQUEST_ID, resumes={REQUEST_ID: "timeout"}))
    # The start-of-wave cleanup releases the (empty) finished set first; what
    # matters is that this wave's request was released after the failure.
    assert [REQUEST_ID] in runner.release_calls


# ----------------------------------------------------------------------------
# failure cleanup


def test_resume_failure_releases_the_wave_through_the_runner_boundary(dispatch) -> None:
    pipeline = _WaitPipeline(wait=None, resume_error=ValueError("fallback encoding failed"))
    runner = _runner(pipeline)
    runner.state_cache[REQUEST_ID] = SimpleNamespace(ready=False)

    with pytest.raises(ValueError, match="fallback encoding failed"):
        dispatch.run(runner, _scheduled(REQUEST_ID, resumes={REQUEST_ID: "timeout"}))

    # The main runner's existing except path releases the wave. It needs no session
    # concept here: the AR runner overrides release_step_requests.
    # The start-of-wave cleanup releases the (empty) finished set first; what
    # matters is that this wave's request was released after the failure.
    assert [REQUEST_ID] in runner.release_calls
    assert REQUEST_ID not in runner.state_cache
    assert dispatch.calls == [], "a failed resume must not reach the core"


def test_core_failure_after_a_successful_resume_still_releases(dispatch) -> None:
    pipeline = _WaitPipeline(wait=None)
    runner = _runner(pipeline)
    runner.state_cache[REQUEST_ID] = SimpleNamespace(ready=False)

    with pytest.raises(RuntimeError, match="denoise exploded"):
        dispatch.run(
            runner,
            _scheduled(REQUEST_ID, resumes={REQUEST_ID: "interaction"}),
            core_raises=RuntimeError("denoise exploded"),
        )

    # The resume ran, then the core failed; the wave is still released.
    assert [reason for _, reason in pipeline.resumed] == ["interaction"]
    # The start-of-wave cleanup releases the (empty) finished set first; what
    # matters is that this wave's request was released after the failure.
    assert [REQUEST_ID] in runner.release_calls
    assert REQUEST_ID not in runner.state_cache


# ----------------------------------------------------------------------------
# letting a chunk wait again


@pytest.mark.parametrize(
    ("finished", "result", "expect_wait"),
    [
        (False, DiffusionOutput(), True),  # a clean, non-terminal chunk
        (True, DiffusionOutput(), False),  # nothing left to resume
        (False, None, False),  # no chunk boundary reached
        (False, DiffusionOutput(error="boom"), False),  # already failing
    ],
)
def test_only_a_successful_nonterminal_chunk_may_wait_again(dispatch, finished, result, expect_wait) -> None:
    pipeline = _WaitPipeline(wait=StepWait(2))
    runner = _runner(pipeline)
    state = SimpleNamespace(ready=False)
    runner.state_cache[REQUEST_ID] = state
    member = RunnerOutput(request_id=REQUEST_ID, finished=finished, result=result)

    output = dispatch.run(runner, _scheduled(REQUEST_ID), core_result=[member])

    applied = output.get_request_output(REQUEST_ID).wait_for_interaction
    if expect_wait:
        assert applied == StepWait(2)
        assert pipeline.waited == [state]
    else:
        assert applied is None
        assert pipeline.waited == []


def test_a_wait_is_never_applied_to_a_request_without_state(dispatch) -> None:
    pipeline = _WaitPipeline(wait=StepWait())
    runner = _runner(pipeline)
    member = RunnerOutput(request_id=REQUEST_ID, result=DiffusionOutput())

    output = dispatch.run(runner, _scheduled(REQUEST_ID), core_result=[member])

    assert output.get_request_output(REQUEST_ID).wait_for_interaction is None
    assert pipeline.waited == []


def test_a_chunk_that_does_not_wait_is_not_annotated(dispatch) -> None:
    pipeline = _WaitPipeline(wait=None)
    runner = _runner(pipeline)
    runner.state_cache[REQUEST_ID] = SimpleNamespace(ready=False)
    member = RunnerOutput(request_id=REQUEST_ID, result=DiffusionOutput())

    output = dispatch.run(runner, _scheduled(REQUEST_ID), core_result=[member])

    assert output.get_request_output(REQUEST_ID).wait_for_interaction is None
    # The hook was still consulted: ``None`` is the pipeline's decision, not a skip.
    assert len(pipeline.waited) == 1
