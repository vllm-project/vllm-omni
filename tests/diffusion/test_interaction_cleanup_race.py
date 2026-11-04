# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Controls queued during the final step must not reach retired worker state."""

import asyncio
import threading

import pytest
from pytest_mock import MockerFixture

from tests.diffusion.test_diffusion_engine_rpc_routing import _make_engine_with_loop, _stop_engine
from tests.diffusion.test_diffusion_scheduler import _make_step_request
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.diffusion_engine import DiffusionExecutionMode, _StreamingPlayback
from vllm_omni.diffusion.sched import DiffusionRequestStatus, StepScheduler
from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner
from vllm_omni.diffusion.worker.utils import RunnerOutput, StepRequestState

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.mark.parametrize("terminal", ["completion", "error", "abort"])
@pytest.mark.parametrize("keyword_args", [False, True])
def test_control_queued_during_final_step_preserves_cleanup(monkeypatch, mocker: MockerFixture, terminal, keyword_args):
    scheduler = StepScheduler()
    scheduler.initialize(OmniDiffusionConfig(max_num_seqs=1))
    engine = _make_engine_with_loop(None, scheduler=scheduler, start_loop=False)
    engine.execution_mode = DiffusionExecutionMode.STEP_BATCH
    engine._request_cancellations = None
    runner = DiffusionModelRunner.__new__(DiffusionModelRunner)
    runner.pipeline = mocker.MagicMock()
    runner.od_config = OmniDiffusionConfig(streaming_output=True)
    runner._interaction_coordinator = mocker.MagicMock()
    request = _make_step_request("video")
    runner.state_cache = {
        "video": StepRequestState(request_id="video", sampling=request.sampling_params, prompt=request.prompt)
    }
    monkeypatch.setattr(runner, "_supports_step_mode", lambda: True)
    entered = threading.Event()
    release = threading.Event()
    emitted = threading.Event()
    terminal_outputs: asyncio.Queue[DiffusionOutput] = asyncio.Queue()
    engine._out_streams["video"] = terminal_outputs
    original_emit = engine._emit_outputs
    original_rpc = engine.executor.collective_rpc
    executions = []

    def emit(*args, **kwargs):
        original_emit(*args, **kwargs)
        emitted.set()

    def rpc(method, timeout=None, args=(), kwargs=None, unique_reply_rank=None, **extra):
        if method == "submit_interaction":
            return runner.submit_interaction(*args, **(kwargs or {}))
        if method == "release_step_requests":
            return runner.release_step_requests(*args)
        return original_rpc(method, timeout, args, kwargs, unique_reply_rank, **extra)

    def execute(output):
        executions.append(output)
        for new_request in output.scheduled_new_reqs:
            runner.state_cache[new_request.request_id] = StepRequestState(
                request_id=new_request.request_id,
                sampling=new_request.req.sampling_params,
                prompt=new_request.req.prompt,
            )
        entered.set()
        assert release.wait(5), "Final step was not released"
        # The real runner releases request state before returning final output.
        finished = terminal != "abort" or len(executions) > 1
        if finished:
            runner.release_step_requests(output.scheduled_request_ids)
        return RunnerOutput(
            "video",
            step_index=1,
            finished=finished,
            result=DiffusionOutput(error="generation failed" if terminal == "error" and len(executions) == 1 else None),
        )

    monkeypatch.setattr(engine.executor, "collective_rpc", rpc)
    monkeypatch.setattr(engine, "_emit_outputs", emit)
    engine.execute_fn = execute
    scheduler.add_request(request)
    engine._loop_started = True
    engine.worker_thread = threading.Thread(target=engine._busy_loop, daemon=True)
    engine.worker_thread.start()
    try:
        assert entered.wait(5)
        interaction = {"event_id": "late", "event": {"prompt": "updated"}}
        args = () if keyword_args else ("video", interaction)
        kwargs = {"request_id": "video", "interaction": interaction} if keyword_args else None
        task = engine._submit_rpc("submit_interaction", 5, args, kwargs, None)
        if terminal == "abort":
            engine.abort("video")
        release.set()
        assert task.future.result(timeout=5) is None
        assert emitted.wait(5)
        assert scheduler.get_request_state("video") is None
        assert not runner.state_cache
        final_output = terminal_outputs.get_nowait()
        assert final_output.finished
        assert final_output.aborted == (terminal == "abort")
        assert final_output.error == ("generation failed" if terminal == "error" else None)
        runner.pipeline.release_step_request.assert_called_once()
        runner._interaction_coordinator.enqueue_parts.assert_not_called()
        assert engine.worker_thread.is_alive()
        # Completion/cancellation must also release admission for a fresh request.
        entered.clear()
        emitted.clear()
        with engine._cv:
            scheduler.add_request(_make_step_request("video"))
            engine._cv.notify_all()
        assert entered.wait(5)
        assert emitted.wait(5)
        assert scheduler.get_request_state("video") is None
        assert not runner.state_cache
        assert len(executions) == 2
        assert runner.pipeline.release_step_request.call_count == 2
    finally:
        release.set()
        _stop_engine(engine)


@pytest.mark.parametrize("terminal", [False, True])
def test_queued_control_cannot_cross_request_id_reuse(terminal):
    scheduler = StepScheduler()
    scheduler.initialize(OmniDiffusionConfig(max_num_seqs=1))
    scheduler.add_request(_make_step_request("video"))
    engine = _make_engine_with_loop(None, scheduler=scheduler, start_loop=False)
    task = engine._submit_rpc(
        "submit_interaction", 5, ("video", {"event_id": "old", "event": {"prompt": "old"}}), None, None
    )
    if terminal:
        scheduler.finish_requests("video", DiffusionRequestStatus.FINISHED_COMPLETED)
    scheduler.pop_request_state("video")
    scheduler.add_request(_make_step_request("video"))
    # A stale control must not clear an identically named successor reservation.
    playback = _StreamingPlayback(buffer_seconds=1.0, pending_actions={"old": 1.0})
    engine._streaming_playback = {"video": playback}
    engine._process_rpc_queue()
    assert playback.pending_actions == {"old": 1.0}
    assert task.future.result(timeout=5) is None
    assert not engine.executor.calls
    assert scheduler._request_progress["video"].interaction_generation == 0


def test_unknown_control_is_rejected_before_a_later_admission():
    scheduler = StepScheduler()
    scheduler.initialize(OmniDiffusionConfig(max_num_seqs=1))
    engine = _make_engine_with_loop(None, scheduler=scheduler, start_loop=False)
    task = engine._submit_rpc(
        "submit_interaction", 5, ("missing", {"event_id": "bad", "event": {"prompt": "bad"}}), None, None
    )
    scheduler.add_request(_make_step_request("missing"))
    engine._process_rpc_queue()
    with pytest.raises(ValueError, match="No active request state"):
        task.future.result(timeout=5)
    assert not engine.executor.calls


@pytest.mark.parametrize("keyword_args", [False, True])
def test_active_control_releases_its_playback_reservation(keyword_args):
    scheduler = StepScheduler()
    scheduler.initialize(OmniDiffusionConfig(max_num_seqs=1))
    scheduler.add_request(_make_step_request("video"))
    engine = _make_engine_with_loop(None, scheduler=scheduler, start_loop=False)
    playback = _StreamingPlayback(buffer_seconds=1.0, pending_actions={"active": 1.0})
    engine._streaming_playback = {"video": playback}
    interaction = {"event_id": "active", "event": {"prompt": "updated"}}
    args = () if keyword_args else ("video", interaction)
    kwargs = {"request_id": "video", "interaction": interaction} if keyword_args else None
    task = engine._submit_rpc("submit_interaction", 5, args, kwargs, None)
    engine._process_rpc_queue()
    assert task.future.result(timeout=5) is not None
    assert playback.pending_actions == {}
    assert scheduler._request_progress["video"].interaction_generation == 1
