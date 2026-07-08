# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import time
from dataclasses import asdict, dataclass

import pytest

from vllm_omni.diffusion.stage_diffusion_proc import StageDiffusionProc
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@dataclass
class MockOmniRequestOutput:
    request_id: str = ""
    status: str = "success"


BASE_HEIGHT = 512
BASE_WIDTH = 512
BASE_INFER_STEPS = 10
DELAY_BASE = 0.01


class MockDiffusionEngine:
    async def step(self, request):
        def simulate_step_delay(height, width, num_inference_steps) -> float:
            return (height / BASE_HEIGHT) * (width / BASE_WIDTH) * (num_inference_steps / BASE_INFER_STEPS)

        DELAY_BASE = 0.01
        delay_scale = simulate_step_delay(
            request.sampling_params.height, request.sampling_params.width, request.sampling_params.num_inference_steps
        )
        delay = DELAY_BASE + delay_scale * DELAY_BASE
        await asyncio.sleep(delay)
        return [MockOmniRequestOutput(request_id=request.request_id)]


@pytest.mark.asyncio
async def test_proc_streaming_request_yields_each_engine_chunk():
    """Ensure that the streaming output chunks from DiffusionEngine reaches StageDiffusionProc"""
    captured = {}
    chunks = [
        OmniRequestOutput.from_diffusion(request_id="", images=[], finished=False),
        OmniRequestOutput.from_diffusion(request_id="", images=[], finished=True),
    ]

    class _StreamingEngine:
        async def step_streaming(self, request):
            captured["request"] = request
            for chunk in chunks:
                yield [chunk]

    stage_proc = object.__new__(StageDiffusionProc)
    stage_proc._engine = _StreamingEngine()

    outputs = [
        output
        async for output in stage_proc._process_streaming_request(
            request_id="req-stream",
            prompt="prompt",
            sampling_params_dict=asdict(OmniDiffusionSamplingParams()),
            kv_sender_info={0: {"host": "127.0.0.1"}},
        )
    ]

    assert outputs == chunks
    assert [output.request_id for output in outputs] == ["req-stream", "req-stream"]
    assert [output.finished for output in outputs] == [False, True]
    assert captured["request"].kv_sender_info == {0: {"host": "127.0.0.1"}}


@pytest.mark.asyncio
async def test_proc_process_request_with_batching_async_output():
    stage_proc = object.__new__(StageDiffusionProc)
    stage_proc._engine = MockDiffusionEngine()

    test_requests = [
        {
            "request_id": "req_1",
            "prompt": "prompt1",
            "params": {"height": BASE_HEIGHT * 2, "width": BASE_WIDTH * 2, "num_inference_steps": BASE_INFER_STEPS * 1},
        },
        {
            "request_id": "req_2",
            "prompt": "prompt2",
            "params": {"height": BASE_HEIGHT * 2, "width": BASE_WIDTH * 2, "num_inference_steps": BASE_INFER_STEPS * 2},
        },
        {
            "request_id": "req_3",
            "prompt": "prompt3",
            "params": {"height": BASE_HEIGHT * 2, "width": BASE_WIDTH * 2, "num_inference_steps": BASE_INFER_STEPS * 3},
        },
    ]

    async def run_task(req_data):
        start_time = time.time()
        result = await stage_proc._process_request(
            request_id=req_data["request_id"], prompt=req_data["prompt"], sampling_params_dict=req_data["params"]
        )
        end_time = time.time()
        return result, end_time - start_time

    coros = [run_task(req) for req in test_requests]
    results = await asyncio.gather(*coros)

    assert len(results) == len(test_requests)
    base_time = DELAY_BASE
    time_gap_std = DELAY_BASE * 2 * 2 * 1  # height/width/steps infer time scale
    eps = 0.1
    for i, (res, elapsed_time) in enumerate(results):
        assert res.request_id == test_requests[i]["request_id"]
        assert isinstance(res, MockOmniRequestOutput)
        time_gap = elapsed_time - base_time
        assert time_gap > time_gap_std - eps and time_gap < time_gap_std + eps
        base_time = elapsed_time
