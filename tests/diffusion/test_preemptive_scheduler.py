# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from threading import Lock

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.scheduler import Scheduler
from vllm_omni.inputs.data import OmniDiffusionSamplingParams


class _DummyBroadcastMQ:
    def __init__(self):
        self.messages = []

    def enqueue(self, msg):
        self.messages.append(msg)


class _DummyResultMQ:
    def __init__(self, messages):
        self.messages = list(messages)

    def dequeue(self):
        if not self.messages:
            raise RuntimeError("no message")
        return self.messages.pop(0)


def _make_req(req_id: str) -> OmniDiffusionRequest:
    return OmniDiffusionRequest(
        prompts=["test"],
        sampling_params=OmniDiffusionSamplingParams(num_inference_steps=2),
        request_ids=[req_id],
    )


def test_request_priority_is_reproducible():
    req1 = _make_req("same-id")
    req2 = _make_req("same-id")
    assert req1.get_or_assign_priority() == req2.get_or_assign_priority()


def test_scheduler_matches_results_by_request_key():
    req1 = _make_req("req-1")
    req2 = _make_req("req-2")
    out1 = DiffusionOutput(output=None, finished=True, request_key=req1.request_key)
    out2 = DiffusionOutput(output=None, finished=True, request_key=req2.request_key)

    scheduler = Scheduler()
    scheduler.mq = _DummyBroadcastMQ()
    scheduler.result_mq = _DummyResultMQ(
        [
            {"type": "generation_result", "request_key": req2.request_key, "output": out2},
            {"type": "generation_result", "request_key": req1.request_key, "output": out1},
        ]
    )
    scheduler._result_lock = Lock()
    scheduler._pending_results = {}

    first = scheduler.add_req(req1)
    second = scheduler.add_req(req2)

    assert first is out1
    assert second is out2
