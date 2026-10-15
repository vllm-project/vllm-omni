# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Held terminal outputs of single-stage audio stages."""

from collections import defaultdict, deque
from types import SimpleNamespace

import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.request_queue import SchedulingPolicy, create_request_queue
from vllm.v1.engine import EngineCoreOutputs, FinishReason
from vllm.v1.request import Request, RequestStatus

from vllm_omni.core.sched import omni_ar_scheduler as scheduler_module
from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler
from vllm_omni.engine import OmniEngineCoreOutput
from vllm_omni.model_executor.models.moss_tts.fused_scheduler import MossFusedScheduler
from vllm_omni.outputs import OmniModelRunnerOutput

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _scheduler():
    held = {
        "a": {"client_index": 1, "finish_reason": FinishReason.STOP, "stop_reason": 7},
        "b": {"client_index": 0, "finish_reason": FinishReason.LENGTH, "stop_reason": None},
    }
    return SimpleNamespace(omni_deferred_finishes=held, omni_tail_outputs=deque())


def test_audio_of_a_step_in_flight_at_the_stop_is_forwarded():
    scheduler = _scheduler()
    outputs: dict[int, list] = defaultdict(list)
    payload = {"model_outputs": torch.ones(2, 3)}
    runner_output = SimpleNamespace(req_id_to_index={"a": 0, "z": 1})
    MossFusedScheduler._forward_deferred_audio(scheduler, outputs, "a", [payload, None], runner_output)
    MossFusedScheduler._forward_deferred_audio(scheduler, outputs, "z", [payload, payload], runner_output)
    (out,) = outputs[1]
    assert out.request_id == "a" and out.finish_reason is None and out.multimodal_output is payload
    assert list(outputs) == [1]


def test_tail_outputs_follow_the_queue_and_release_the_hold():
    scheduler = _scheduler()
    audio = {"model_outputs": torch.ones(2, 3)}
    scheduler.omni_tail_outputs.extend(
        [("a", audio, False, None), ("b", None, True, None), ("a", None, False, None), ("a", audio, True, None)]
    )
    outputs: dict[int, list] = defaultdict(list)
    MossFusedScheduler._emit_tail_outputs(scheduler, outputs)
    a = outputs[1]
    assert [o.finish_reason for o in a] == [None, FinishReason.STOP] and a[1].stop_reason == 7
    (b,) = outputs[0]
    assert b.finish_reason == FinishReason.LENGTH and b.multimodal_output is None
    assert not scheduler.omni_deferred_finishes and not scheduler.omni_tail_outputs
    # Late items of a request whose terminal output went out are dropped.
    scheduler.omni_tail_outputs.append(("a", audio, False, None))
    MossFusedScheduler._emit_tail_outputs(scheduler, outputs)
    assert len(outputs[1]) == 2


def test_tail_copy_failure_emits_error_and_releases_worker_state():
    scheduler = _scheduler()
    scheduler.finished_req_ids = set()
    scheduler.omni_tail_outputs.extend([("a", None, True, "copy failed"), ("a", {}, True, None)])
    outputs: dict[int, list] = defaultdict(list)
    MossFusedScheduler._emit_tail_outputs(scheduler, outputs)
    (output,) = outputs[1]
    assert output.finish_reason == FinishReason.ERROR and output.stop_reason == "copy failed"
    assert scheduler.finished_req_ids == {"a"} and "a" not in scheduler.omni_deferred_finishes


def test_schedule_reserves_detached_codec_capacity(monkeypatch):
    scheduler = MossFusedScheduler.__new__(MossFusedScheduler)
    scheduler.max_num_active_reqs = 128
    scheduler.omni_deferred_finishes = {"old": {}}

    def schedule(self, throttle_prefills=False):
        assert self.max_num_active_reqs == 127
        return object()

    monkeypatch.setattr(scheduler_module.OmniARAsyncScheduler, "schedule", schedule)
    scheduler.schedule()
    assert scheduler.max_num_active_reqs == 128


def test_fused_preemption_fails_request_instead_of_resuming(mocker):
    scheduler = MossFusedScheduler.__new__(MossFusedScheduler)
    scheduler.waiting = create_request_queue(SchedulingPolicy.FCFS)
    scheduler.omni_deferred_finishes = {}
    scheduler.omni_tail_outputs = deque()
    scheduler.finished_req_ids = set()
    scheduler.reset_preempted_req_ids = set()
    scheduler.aux_output_connector = None
    scheduler.encoder_cache_manager = mocker.Mock()
    scheduler._inflight_prefills = set()
    scheduler.log_stats = False
    scheduler._free_request_blocks = mocker.Mock()
    request = Request("r", [1], SamplingParams(max_tokens=32), None)
    request.status = RequestStatus.RUNNING
    scheduler._free_request = mocker.Mock()
    scheduler._preempt_request(request, 0.0)
    assert not list(scheduler.waiting) and request.status == RequestStatus.FINISHED_ERROR
    outputs: dict[int, list] = defaultdict(list)
    scheduler._emit_tail_outputs(outputs)
    assert outputs[0][0].finish_reason == FinishReason.ERROR
    assert "preemption" in outputs[0][0].stop_reason
    scheduler._free_request.assert_called_once_with(request)


def test_abort_detached_request_releases_hold_and_notifies_worker(mocker):
    scheduler = MossFusedScheduler.__new__(MossFusedScheduler)
    scheduler.omni_deferred_finishes = _scheduler().omni_deferred_finishes
    scheduler.finished_req_ids = set()
    scheduler._kv_wait_start_ts = {}
    finish = mocker.patch.object(scheduler_module.OmniSchedulerMixin, "finish_requests", return_value=[])
    scheduler.finish_requests("a", RequestStatus.FINISHED_ABORTED)
    assert "a" not in scheduler.omni_deferred_finishes and "b" in scheduler.omni_deferred_finishes
    assert scheduler.finished_req_ids == {"a"}
    finish.assert_called_once_with(("a",), RequestStatus.FINISHED_ABORTED)


@pytest.mark.parametrize("finish_reason", [FinishReason.STOP, FinishReason.LENGTH])
def test_terminal_is_held_until_in_flight_audio_and_tail_are_emitted(monkeypatch, finish_reason):
    scheduler = MossFusedScheduler.__new__(MossFusedScheduler)
    scheduler.omni_deferred_finishes = {}
    scheduler.omni_tail_outputs = deque()
    terminal = OmniEngineCoreOutput(request_id="r", new_token_ids=[7], finish_reason=finish_reason, stop_reason=7)
    monkeypatch.setattr(OmniARScheduler, "update_from_output", lambda *args: {2: EngineCoreOutputs(outputs=[terminal])})
    runner = OmniModelRunnerOutput(req_ids=["r"], sampled_token_ids=[[]], req_id_to_index={"r": 0})
    first = scheduler.update_from_output(None, runner)
    assert first[2].outputs[0].new_token_ids == [7]
    assert first[2].outputs[0].finish_reason is None
    assert scheduler.has_requests()

    monkeypatch.setattr(OmniARScheduler, "update_from_output", lambda *args: {})
    in_flight, tail = {"model_outputs": torch.ones(3)}, {"model_outputs": torch.ones(5)}
    runner.multimodal_outputs = [in_flight]
    scheduler.omni_tail_outputs.append(("r", tail, True, None))
    second = scheduler.update_from_output(None, runner)
    assert [o.multimodal_output for o in second[2].outputs] == [in_flight, tail]
    assert [o.finish_reason for o in second[2].outputs] == [None, finish_reason]
    assert second[2].outputs[-1].stop_reason == 7
    assert scheduler.omni_deferred_finishes == {}


def test_only_fused_pipeline_selects_tail_scheduler():
    from vllm_omni.model_executor.models.moss_tts.pipeline import (
        MOSS_TTS_LOCAL_FUSED_PIPELINE,
        MOSS_TTS_LOCAL_PIPELINE,
        MOSS_TTS_PIPELINE,
    )

    assert MOSS_TTS_LOCAL_FUSED_PIPELINE.stages[0].scheduler_cls.endswith(".MossFusedScheduler")
    assert all(stage.scheduler_cls is None for p in (MOSS_TTS_PIPELINE, MOSS_TTS_LOCAL_PIPELINE) for stage in p.stages)
