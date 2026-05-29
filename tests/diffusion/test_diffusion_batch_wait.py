# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import queue
import threading
import time
from types import SimpleNamespace

from vllm_omni.diffusion.diffusion_engine import DiffusionEngine
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.sched import RequestScheduler
from vllm_omni.inputs.data import OmniDiffusionSamplingParams


def _make_request(req_id: str) -> OmniDiffusionRequest:
    return OmniDiffusionRequest(
        prompts=[f"prompt_{req_id}"],
        sampling_params=OmniDiffusionSamplingParams(num_inference_steps=1),
        request_ids=[req_id],
    )


def test_request_batch_idle_cutoff_dispatches_partial_batch() -> None:
    engine = DiffusionEngine.__new__(DiffusionEngine)
    engine.scheduler = RequestScheduler()
    engine.scheduler.initialize(SimpleNamespace(max_num_seqs=4))
    engine.scheduler.add_request(_make_request("req-idle"))
    engine.request_batching = True
    engine._request_batch_wait_s = 1.0
    engine._request_batch_idle_s = 0.01
    engine._request_batch_idle_min_size = 1
    engine._rpc_queue = queue.Queue()
    engine.abort_queue = queue.Queue()
    engine.stop_event = threading.Event()
    engine._cv = threading.Condition(threading.RLock())

    start = time.monotonic()
    with engine._cv:
        engine._wait_for_request_batch_if_needed()

    assert time.monotonic() - start < 0.2
