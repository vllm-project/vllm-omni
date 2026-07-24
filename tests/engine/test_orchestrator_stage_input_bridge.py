# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import queue
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

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
        self.decoded_source_tokens: str | None = None
        self._engine_core_outputs = queue.Queue()

    async def add_request_async(self, *args, **_kwargs) -> None:
        self.add_request_calls.append(args)

    async def get_output_async(self):
        try:
            return self._engine_core_outputs.get_nowait()
        except queue.Empty:
            return SimpleNamespace(outputs=[])

    def process_engine_inputs(self, _source_outputs, prompt=None, streaming_context=None):
        decoder = getattr(streaming_context, "source_token_decoder", None)
        if callable(decoder):
            self.decoded_source_tokens = decoder([11, 12], skip_special_tokens=True)
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
    def __init__(self, tokenizer=None) -> None:
        self.tokenizer = tokenizer

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


class FakePrewarmPool:
    stage_type = "llm"

    def __init__(self, role: str) -> None:
        self.stage_vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(
                max_model_len=64,
                stage_connector_config={"extra": {"role": role}},
            )
        )
        self.submitted: list[Any] = []

    async def submit_initial(self, _request_id, _req_state, request, prompt_text=None):
        self.submitted.append(request)

    def get_bound_replica_id(self, _request_id):
        return 0


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
    class SourceTokenizer:
        def decode(self, token_ids, *, skip_special_tokens):
            assert skip_special_tokens is True
            return ":".join(str(token_id) for token_id in token_ids)

    stage0 = FakeStageClient(final_output=True)
    stage1 = FakeStageClient(
        final_output=True,
        next_inputs=[{"prompt": "hello", "multi_modal_data": {"video": ["frame"]}}],
    )
    stage_pools = [
        StagePool(
            0,
            [stage0],
            output_processor=FakeOutputProcessor(tokenizer=SourceTokenizer()),
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
    assert stage1.decoded_source_tokens == "11:12"
    assert req_state.streaming.source_token_decoder is None
    assert stage1.add_request_calls
    submitted_request = stage1.add_request_calls[0][0]
    assert submitted_request.prompt_token_ids == [101, 102]
    assert submitted_request.external_req_id == "req-text"


@pytest.mark.asyncio
async def test_async_prewarm_skips_outgoing_only_stage() -> None:
    orchestrator = object.__new__(Orchestrator)
    stage0 = FakePrewarmPool("sender")
    stage1 = FakePrewarmPool("sender")
    stage2 = FakePrewarmPool("receiver")
    orchestrator.stage_pools = [stage0, stage1, stage2]
    orchestrator._emit_tx_edge = lambda **_kwargs: None
    req_state = OrchestratorRequestState(
        request_id="req-prewarm",
        prompt={"prompt_token_ids": [1, 2]},
        sampling_params_list=[SamplingParams(max_tokens=1) for _ in range(3)],
        final_stage_id=2,
    )

    await orchestrator._prewarm_async_chunk_stages(
        "req-prewarm",
        SimpleNamespace(prompt_token_ids=[1, 2], resumable=True),
        req_state,
    )

    assert stage1.submitted == []
    assert len(stage2.submitted) == 1
    assert 1 not in req_state.stage_submit_ts
    assert 2 in req_state.stage_submit_ts


@pytest.mark.asyncio
async def test_async_route_forwards_to_outgoing_only_stage() -> None:
    orchestrator = object.__new__(Orchestrator)
    orchestrator.async_chunk = True
    orchestrator._pd_pair = None
    orchestrator._cfg_tracker = SimpleNamespace(
        is_companion=lambda _request_id: False,
        has_companions=lambda _request_id: False,
    )
    stage0 = SimpleNamespace(final_output=False)
    stage1 = FakePrewarmPool("sender")
    orchestrator.stage_pools = [stage0, stage1]
    orchestrator._forward_to_next_stage = AsyncMock()
    req_state = OrchestratorRequestState(
        request_id="req-route",
        sampling_params_list=[SamplingParams(max_tokens=1) for _ in range(2)],
        final_stage_id=1,
    )
    req_state.stage_submit_ts[0] = 1.0
    output = SimpleNamespace(request_id="req-route", finished=True)

    await orchestrator._route_output(0, 0, output, req_state, None)

    orchestrator._forward_to_next_stage.assert_awaited_once()
