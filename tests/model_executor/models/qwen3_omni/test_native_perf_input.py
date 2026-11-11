# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The real input processor must preserve per-request perf identity at upgrade."""

import copy

import pytest
from vllm import SamplingParams

from tests.engine.test_duplex_orchestrator import _build
from tests.model_executor.models.qwen3_omni.test_memory_card_stop_integration import (
    native_qwen_input_processor as _native_fixture,
)
from tests.model_executor.models.qwen3_omni.test_memory_card_stop_integration import (
    qwen_processor as _qwen_fixture,
)
from vllm_omni.core.sched.duplex_stage_work import request_perf_chunk
from vllm_omni.model_executor.models.qwen3_omni.duplex.plugin import Qwen3OmniDuplexPlugin

native_qwen_input_processor = _native_fixture
qwen_processor = _qwen_fixture
pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.asyncio
@pytest.mark.parametrize("collect_stats", [False, True])
@pytest.mark.parametrize("builder", ["entry", "next"])
async def test_real_qwen_processor_preserves_perf_identity(
    qwen_processor, native_qwen_input_processor, collect_stats, builder
):
    text = qwen_processor.apply_chat_template(
        [{"role": "user", "content": "Say hello."}], tokenize=False, add_generation_prompt=True
    )
    prompt = {"prompt": text}
    if collect_stats:
        prompt["model_intermediate_buffer"] = {"duplex": {"perf_chunk": [0, 7]}}
    before = copy.deepcopy(prompt)
    params = SamplingParams(max_tokens=64, temperature=0)
    reference = native_qwen_input_processor.process_inputs(
        request_id="reference", prompt={"prompt": text}, params=params, supported_tasks=("generate",)
    )
    plugin = Qwen3OmniDuplexPlugin(lambda *args: "")
    plugin.processor = qwen_processor
    orchestrator, _, _, _ = _build(stages=1, plugin=plugin)
    orchestrator._stage_input_processors[0] = native_qwen_input_processor
    try:
        if builder == "next":
            wire = await orchestrator._build_next_stage_request("native-perf", 0, prompt, params, resumable=False)
        else:
            from types import SimpleNamespace

            state = SimpleNamespace(sampling_params_list=[params], request_timestamp=0.0, final_stage_id=0)
            wire = await orchestrator._build_entry_stage_request("native-perf", 0, prompt, state)
        assert wire.prompt_token_ids == reference.prompt_token_ids
        assert wire.sampling_params.eos_token_id == reference.sampling_params.eos_token_id
        assert wire.sampling_params.all_stop_token_ids == reference.sampling_params.all_stop_token_ids
        assert wire.mm_features == reference.mm_features
        assert prompt == before
        assert request_perf_chunk(wire) == ((0, 7) if collect_stats else None)
    finally:
        await orchestrator.session_manager.shutdown()
