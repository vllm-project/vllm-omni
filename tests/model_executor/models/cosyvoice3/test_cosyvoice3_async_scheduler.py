# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Scheduler routing tests for deferred completion, cancellation and stale audio.

Use real engine output types; isolate only the base AR scheduler.
"""

import queue
from types import SimpleNamespace

import pytest
import torch
from vllm.v1.engine import FinishReason
from vllm.v1.request import RequestStatus

from vllm_omni.core.sched.omni_ar_scheduler import OmniARAsyncScheduler
from vllm_omni.engine import OmniEngineCoreOutput, OmniEngineCoreOutputs
from vllm_omni.model_executor.models.cosyvoice3 import async_codec, async_scheduler

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def modules(monkeypatch):
    def initialize(s):
        s.running, s.requests = [], {}
        s.max_num_active_reqs = 2

    def schedule(s, throttle_prefills=False):
        s.observed_limit = s.max_num_active_reqs
        return SimpleNamespace(scheduled_new_reqs=[], preempted_req_ids=set())

    monkeypatch.setattr(OmniARAsyncScheduler, "__init__", initialize)
    monkeypatch.setattr(OmniARAsyncScheduler, "update_from_output", lambda s, batch, result: result)
    monkeypatch.setattr(OmniARAsyncScheduler, "schedule", schedule)
    monkeypatch.setattr(
        OmniARAsyncScheduler, "add_request", lambda s, request: s.requests.update({request.request_id: request})
    )
    monkeypatch.setattr(OmniARAsyncScheduler, "has_requests", lambda s: False)
    monkeypatch.setattr(OmniARAsyncScheduler, "has_unfinished_requests", lambda s: False)
    monkeypatch.setattr(OmniARAsyncScheduler, "finish_requests", lambda s, ids, status: [])
    return SimpleNamespace(
        codec=async_codec,
        scheduler=async_scheduler,
        Reason=FinishReason,
        Status=RequestStatus,
        Output=OmniEngineCoreOutput,
        Outputs=OmniEngineCoreOutputs,
    )


def _scheduler(m):
    s = m.scheduler.CosyVoice3AsyncScheduler()

    class Codec:
        def __init__(self):
            self.outputs = queue.SimpleQueue()
            self.released, self.cancelled = [], []

        def check_error(self):
            pass

        def release(self, req_id):
            self.released.append(req_id)

        def cancel(self, req_id):
            self.cancelled.append(req_id)

    s.codec = Codec()
    s.audio_sample_rate = 24000
    s.audio_routes["r"] = 7
    return s


def _terminal(m, s):
    out = m.Output(request_id="r", new_token_ids=[6561], finish_reason=m.Reason.STOP)
    batch = m.Outputs(outputs=[out])
    batch.finished_requests = {"r"}
    result = s.update_from_output(None, {7: batch})
    assert not result[7].outputs
    assert not result[7].finished_requests
    return out


def test_eos_frees_ar_but_defers_client_finish_until_host_audio(modules):
    m = modules
    s = _scheduler(m)
    terminal = _terminal(m, s)
    assert s.has_requests()  # Engine must continue without an AR batch.
    s.codec.outputs.put(m.codec.ReadyAudio("r", torch.tensor([1.0, 2.0]), True))
    out = s.update_from_output(None, {})[7]
    assert out.outputs == [terminal]
    assert terminal.multimodal_output["audio"].tolist() == [1.0, 2.0]
    assert out.finished_requests == {"r"}
    assert s.codec.released == ["r"]
    assert not s.has_requests()


def test_audio_can_finish_before_engine_consumes_eos(modules):
    m = modules
    s = _scheduler(m)
    s.codec.outputs.put(m.codec.ReadyAudio("r", torch.ones(2), True))
    first = s.update_from_output(None, {})[7].outputs[0]
    assert first.finish_reason is None
    assert not s.codec.released
    terminal = m.Output(request_id="r", new_token_ids=[6561], finish_reason=m.Reason.STOP)
    result = s.update_from_output(None, {7: m.Outputs(outputs=[terminal])})
    assert result[7].outputs[0].finish_reason == m.Reason.STOP
    assert s.codec.released == ["r"]


def test_multiple_ready_chunks_coalesce_without_duplicate_request_outputs(modules):
    m = modules
    s = _scheduler(m)
    for n in (1.0, 2.0):
        s.codec.outputs.put(m.codec.ReadyAudio("r", torch.tensor([n]), False))
    tokens = m.Output(request_id="r", new_token_ids=[12])
    batch = s.update_from_output(None, {7: m.Outputs(outputs=[tokens])})[7]
    assert len(batch.outputs) == 1
    assert batch.outputs[0].new_token_ids == [12]
    assert batch.outputs[0].multimodal_output["audio"].tolist() == [1.0, 2.0]


def test_audio_for_retired_request_is_discarded(modules):
    m = modules
    s = _scheduler(m)
    s.retire_request("r")
    s.codec.outputs.put(m.codec.ReadyAudio("r", torch.ones(1), True))
    assert s.update_from_output(None, {}) == {}
    assert not s.audio_done


def test_cancel_detached_tail_does_not_hang_or_emit_pcm(modules):
    m = modules
    s = _scheduler(m)
    _terminal(m, s)
    s.finish_requests(["r"], m.Status.FINISHED_ABORTED)
    s.codec.outputs.put(m.codec.ReadyAudio("r", torch.ones(1), True))
    output = s.update_from_output(None, {})[7].outputs[0]
    assert output.finish_reason == m.Reason.ABORT
    assert output.multimodal_output is None
    assert not s.has_requests()


def test_detached_tail_reserves_admission_capacity(modules):
    s = _scheduler(modules)
    _terminal(modules, s)
    s.schedule()
    assert s.observed_limit == 1
    assert s.max_num_active_reqs == 2


def test_queued_requests_have_routes_without_reserving_tail_capacity(modules):
    s = _scheduler(modules)
    for i in range(4):
        s.add_request(SimpleNamespace(request_id=f"queued-{i}", client_index=i))
    assert all(s.audio_routes[f"queued-{i}"] == i for i in range(4))
    s.schedule()
    assert s.observed_limit == s.max_num_active_reqs == 2
    assert not s.audio_terminals


def test_cancel_before_first_schedule_releases_route_on_terminal_output(modules):
    m = modules
    s = _scheduler(m)
    s.add_request(SimpleNamespace(request_id="queued", client_index=3))
    s.finish_requests(["queued"], m.Status.FINISHED_ABORTED)
    terminal = m.Output(request_id="queued", new_token_ids=[], finish_reason=m.Reason.ABORT)
    result = s.update_from_output(None, {3: m.Outputs(outputs=[terminal], finished_requests={"queued"})})
    assert result[3].outputs == [terminal]
    assert result[3].finished_requests == {"queued"}
    assert "queued" not in s.audio_routes
    assert "queued" in s.codec.released
