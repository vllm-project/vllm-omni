# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import queue
from types import SimpleNamespace
from typing import Any

import janus
import pytest
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.sampling_params import SamplingParams

from vllm_omni.engine.orchestrator import Orchestrator, OrchestratorRequestState
from vllm_omni.engine.stage_pool import StagePool

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class FakeStageClient:
    def __init__(
        self,
        *,
        next_inputs: list[dict[str, Any]] | None = None,
        final_output: bool = False,
    ) -> None:
        self.stage_id = 0
        self.replica_id = 0
        self.stage_type = "llm"
        self.final_output = final_output
        self.final_output_type = "text"
        self.default_sampling_params = SamplingParams(max_tokens=1)
        self.requires_multimodal_data = False
        self.engine_input_source = [0]
        self.is_comprehension = False
        self.model_stage = None
        self.custom_process_input_func = None
        self.next_inputs = list(next_inputs or [])
        self.add_request_calls: list[tuple[Any, ...]] = []
        self._engine_core_outputs = queue.Queue()

    async def add_request_async(self, *args, **_kwargs) -> None:
        self.add_request_calls.append(args)

    async def get_output_async(self):
        try:
            return self._engine_core_outputs.get_nowait()
        except queue.Empty:
            return SimpleNamespace(outputs=[])

    def process_engine_inputs(self, _source_outputs, prompt=None, streaming_context=None):
        return list(self.next_inputs)

    async def abort_requests_async(self, _request_ids: list[str]) -> None:
        return None

    def set_engine_outputs(self, _outputs) -> None:
        return None

    def check_health(self) -> None:
        return None

    def shutdown(self) -> None:
        return None


class FakeOutputProcessor:
    def add_request(self, *args, **kwargs) -> None:
        return None


class FakeInputProcessor:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def process_inputs(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            request_id=kwargs["request_id"],
            prompt_token_ids=[101, 102],
            prompt_embeds=None,
            external_req_id=None,
        )


def _request_output(request_id: str) -> RequestOutput:
    completion = CompletionOutput(
        index=0,
        text="transcript",
        token_ids=[11, 12],
        cumulative_logprob=None,
        logprobs=None,
        finish_reason="stop",
        stop_reason=None,
    )
    return RequestOutput(
        request_id=request_id,
        prompt="prompt",
        prompt_token_ids=[1, 2],
        prompt_logprobs=None,
        outputs=[completion],
        finished=True,
        metrics=None,
        lora_request=None,
    )


@pytest.mark.asyncio
async def test_forward_text_prompt_uses_target_stage_input_processor() -> None:
    stage0 = FakeStageClient(final_output=True)
    stage1 = FakeStageClient(
        final_output=True,
        next_inputs=[{"prompt": "hello", "multi_modal_data": {"video": ["frame"]}}],
    )
    stage_pools = [
        StagePool(
            0,
            [stage0],
            output_processor=FakeOutputProcessor(),
            stage_vllm_config=SimpleNamespace(model_config=SimpleNamespace(max_model_len=64)),
        ),
        StagePool(
            1,
            [stage1],
            output_processor=FakeOutputProcessor(),
            stage_vllm_config=SimpleNamespace(model_config=SimpleNamespace(max_model_len=64)),
        ),
    ]
    request_q = janus.Queue()
    output_q = janus.Queue()
    rpc_q = janus.Queue()
    orchestrator = Orchestrator(
        request_async_queue=request_q.async_q,
        output_async_queue=output_q.async_q,
        rpc_async_queue=rpc_q.async_q,
        stage_pools=stage_pools,
        async_chunk=False,
    )
    input_processor = FakeInputProcessor()
    orchestrator._stage_input_processors[1] = input_processor
    req_state = OrchestratorRequestState(
        request_id="req-text",
        prompt={"prompt": "original"},
        sampling_params_list=[SamplingParams(max_tokens=1), SamplingParams(max_tokens=1)],
        final_stage_id=1,
    )

    await orchestrator._forward_to_next_stage("req-text", 0, _request_output("req-text"), req_state)

    assert input_processor.calls
    assert input_processor.calls[0]["prompt"] == {"prompt": "hello", "multi_modal_data": {"video": ["frame"]}}
    assert stage1.add_request_calls
    submitted_request = stage1.add_request_calls[0][0]
    assert submitted_request.prompt_token_ids == [101, 102]
    assert submitted_request.external_req_id == "req-text"
