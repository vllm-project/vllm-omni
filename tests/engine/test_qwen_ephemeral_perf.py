# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Per-response Qwen perf identity must survive native downstream prewarm."""

from types import SimpleNamespace

import pytest

from tests.engine.duplex.test_qwen_context_memory_runtime import _Processor
from tests.engine.test_duplex_orchestrator import SESSION_ID, _append_audio, _build, _open, _submit
from vllm_omni.config.stage_config import DuplexSessionRuntimeConfig
from vllm_omni.core.sched.duplex_stage_work import DuplexStageWorkRecorder, request_perf_chunk
from vllm_omni.engine.duplex import commands
from vllm_omni.engine.orchestrator import build_engine_core_request_from_tokens
from vllm_omni.inputs.preprocess import OmniRenderer
from vllm_omni.model_executor.models.qwen3_omni.duplex.plugin import Qwen3OmniDuplexPlugin
from vllm_omni.request import OmniRequest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.asyncio
@pytest.mark.parametrize("policy", ["legacy", "budget"])
@pytest.mark.parametrize("collect_stats", [False, True])
async def test_qwen_ephemeral_perf_identity_reaches_all_prewarmed_stages(monkeypatch, policy, collect_stats):
    plugin = Qwen3OmniDuplexPlugin(lambda *args: "")
    plugin.processor = _Processor()
    plugin.max_model_len = 4096
    runtime = DuplexSessionRuntimeConfig(collect_stats=collect_stats, qwen_context_policy=policy)
    orchestrator, clients, rpc_q, _ = _build(stages=3, runtime_config=runtime, plugin=plugin)
    orchestrator.async_chunk = True
    orchestrator._stage_receives_async_chunks = lambda stage: stage > 0
    prompts = []

    def process_inputs(**kwargs):
        # Replace only tokenization: production extra-field preservation and
        # native EngineCore request construction run unchanged.
        prompt = kwargs["prompt"]
        prompts.append(prompt)
        inputs = OmniRenderer._with_omni_extras({"prompt_token_ids": [1, 2]}, prompt)
        return build_engine_core_request_from_tokens(
            request_id=kwargs["request_id"], prompt=inputs, params=kwargs["params"], resumable=False
        )

    monkeypatch.setattr(
        orchestrator, "_get_stage_input_processor", lambda stage: SimpleNamespace(process_inputs=process_inputs)
    )
    try:
        assert (await _open(orchestrator, rpc_q, extra_body={"auto_response": False})).ok
        await _submit(orchestrator, _append_audio())
        await _submit(orchestrator, commands.Commit(final=True, create_response=True))
        requests = [client.add_request_calls[0][0] for client in clients]
        assert len(prompts) == 1
        assert all(request.resumable is False for request in requests)
        assert len({request.request_id for request in requests}) == 1
        session = orchestrator.session_manager.get(SESSION_ID)
        if collect_stats:
            turn = session.perf.snapshot()["turns"][0]
            chunk = turn["chunks"][0]
            expected = (turn["epoch"], chunk["seq"])
            assert [request_perf_chunk(request) for request in requests] == [expected] * 3
            for wire_request in requests:
                request = OmniRequest.from_engine_core_request(wire_request, block_hasher=None)
                step = SimpleNamespace(
                    scheduled_new_reqs=[SimpleNamespace(req_id=request.request_id)],
                    num_scheduled_tokens={request.request_id: 1},
                )
                output = SimpleNamespace(request_id=request.request_id, stage_work=None)
                ticks = iter((100, 200))
                recorder = DuplexStageWorkRecorder(clock=lambda: next(ticks))
                recorder.step_begin(step, {request.request_id: request})
                recorder.step_end(step, {0: SimpleNamespace(outputs=[output])})
                assert output.stage_work == [[*expected, 100, 100]]
            # Prewarm copies containers: a downstream modification cannot
            # corrupt another stage's identity or the original prompt.
            requests[1].model_intermediate_buffer["duplex"]["perf_chunk"][0] += 1
            assert request_perf_chunk(requests[0]) == expected
            assert request_perf_chunk(requests[2]) == expected
        else:
            assert session.perf is None
            assert "model_intermediate_buffer" not in prompts[0]
            assert [request_perf_chunk(request) for request in requests] == [None] * 3
    finally:
        await orchestrator.session_manager.shutdown()
