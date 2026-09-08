# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
from vllm import SamplingParams
from vllm.v1.core.sched.request_queue import SchedulingPolicy, create_request_queue
from vllm.v1.request import Request, RequestStatus

from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler
from vllm_omni.core.sched.omni_cfg_ar_scheduler import OmniCFGARScheduler

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _request(name: str, prompt_length: int) -> Request:
    return Request(name, [0] * prompt_length, SamplingParams(max_tokens=100), None)


def _scheduler(monkeypatch, waiting, running=()):
    scheduler = OmniCFGARScheduler.__new__(OmniCFGARScheduler)
    scheduler.policy = SchedulingPolicy.FCFS
    scheduler.waiting = create_request_queue(scheduler.policy)
    for request in waiting:
        scheduler.waiting.add_request(request)
    scheduler.skipped_waiting = create_request_queue(scheduler.policy)
    scheduler.running = list(running)
    scheduler.requests = {request.request_id: request for request in [*waiting, *running]}
    scheduler.max_model_len = 128
    scheduler.max_num_running_reqs = 2
    scheduler.max_num_scheduled_tokens = 256
    scheduler.block_size = 16
    scheduler._capacity = 16
    scheduler._pairs = {"pair": {"cond": "cond", "uncond": "uncond"}}
    scheduler._request_pairs = {"cond": "pair", "uncond": "pair"}
    scheduler._prefill_min_output_tokens = 0

    def advance(self, throttle_prefills=False):
        self.running.extend(self.waiting)
        self.waiting = create_request_queue(self.policy)
        scheduled = {}
        for request in self.running:
            count = request.num_tokens - request.num_computed_tokens
            request.num_computed_tokens += count
            request.append_output_token_ids(0)
            scheduled[request.request_id] = count
        return SimpleNamespace(num_scheduled_tokens=scheduled)

    monkeypatch.setattr(OmniARScheduler, "schedule", advance)
    return scheduler


def test_unequal_prompts_keep_real_positions_and_share_context_limit(monkeypatch):
    cond, uncond = _request("cond", 80), _request("uncond", 30)
    scheduler = _scheduler(monkeypatch, [cond, uncond])
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {"cond": 80, "uncond": 30}
    assert (cond.max_tokens, uncond.max_tokens) == (48, 48)
    for _ in range(4):
        assert scheduler.schedule().num_scheduled_tokens == {"cond": 1, "uncond": 1}
    assert cond.num_computed_tokens - uncond.num_computed_tokens == 50


def test_waiting_pair_cannot_be_split_or_starved_by_later_single(monkeypatch):
    cond, uncond = _request("cond", 40), _request("uncond", 20)
    running, later = _request("running", 10), _request("later", 10)
    scheduler = _scheduler(monkeypatch, [cond, uncond, later], [running])
    assert scheduler.schedule().num_scheduled_tokens == {"running": 10}
    assert list(scheduler.waiting) == [cond, uncond, later]
    scheduler.running.clear()
    del scheduler.requests["running"]
    assert scheduler.schedule().num_scheduled_tokens == {"cond": 40, "uncond": 20}
    assert list(scheduler.waiting) == [later]


def test_cancel_either_branch_finishes_both_and_releases_pair(monkeypatch):
    cond, uncond = _request("cond", 40), _request("uncond", 20)
    scheduler = _scheduler(monkeypatch, [cond, uncond])

    def finish(self, ids, status):
        assert set(ids) == {"cond", "uncond"}
        assert status == RequestStatus.FINISHED_ABORTED
        finished = [self.requests.pop(rid) for rid in ids]
        self.waiting.remove_requests(finished)
        return finished

    monkeypatch.setattr(OmniARScheduler, "finish_requests", finish)
    assert len(scheduler.finish_requests("uncond", RequestStatus.FINISHED_ABORTED)) == 2
    assert not scheduler._pairs and not scheduler._request_pairs and not scheduler.waiting


def test_new_prefill_waits_for_initial_streaming_audio(monkeypatch):
    request = _request("plain", 20)
    scheduler = _scheduler(monkeypatch, [], [request])
    scheduler._prefill_min_output_tokens = 8
    request.append_output_token_ids([0] * 7)
    assert scheduler._should_defer_waiting_admission()
    request.append_output_token_ids(0)
    assert not scheduler._should_defer_waiting_admission()
