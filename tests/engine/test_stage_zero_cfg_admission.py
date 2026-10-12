# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import janus
import pytest
from vllm import SamplingParams
from vllm.v1.engine import EngineCoreRequest

from vllm_omni.engine.async_omni_engine import AsyncOmniEngine, StageRuntimeInfo
from vllm_omni.engine.serialization import deserialize_additional_information
from vllm_omni.model_executor.stage_input_processors.bagel import ExpandedPrompt

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("stage_zero_cfg", [False, True])
def test_cfg_expansion_respects_stage_zero_guidance(stage_zero_cfg):
    """AR guidance needs its companion even without a downstream consumer."""
    engine = object.__new__(AsyncOmniEngine)
    engine.default_sampling_params_list = [SamplingParams(max_tokens=8)]
    engine.stage_metadata = [StageRuntimeInfo(final_output=False, final_output_type=None, stage_type="llm")]
    engine.supported_tasks = ("generate",)
    engine.prompt_transform_func = None
    engine.request_queue = janus.Queue()

    def process_inputs(request_id, **kwargs):
        return EngineCoreRequest(
            request_id=request_id,
            prompt_token_ids=[1, 2],
            mm_features=None,
            sampling_params=SamplingParams(max_tokens=8),
            pooling_params=None,
            arrival_time=0.0,
            lora_request=None,
            cache_salt=None,
            data_parallel_rank=None,
        )

    engine.input_processor = SimpleNamespace(process_inputs=process_inputs)
    engine.prompt_expand_func = lambda *_: [
        ExpandedPrompt(prompt={"prompt_token_ids": [3, 4]}, role="uncond", request_id_suffix="__uncond")
    ]
    prompt: dict = {"prompt_token_ids": [1, 2]}
    if stage_zero_cfg:
        prompt["additional_information"] = {"cfg_group": {"role": "cond"}}

    engine.add_request("parent", prompt, final_stage_id=0)

    parent = engine.request_queue.sync_q.get_nowait()
    assert parent.request_id == "parent"
    expect_companion = stage_zero_cfg
    assert engine.request_queue.sync_q.qsize() == int(expect_companion)
    if expect_companion:
        companion = engine.request_queue.sync_q.get_nowait()
        assert companion.parent_id == "parent"
        assert companion.companion_id == "parent__uncond"
        metadata = deserialize_additional_information(companion.prompt.additional_information)
        assert metadata["omni_final_stage_id"] == 0
        assert metadata["omni_force_kv_transfer"] is True
    engine.request_queue.close()
