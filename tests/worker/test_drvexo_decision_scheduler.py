# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Actual runner metadata must stay aligned when continuous batches reorder."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pytest
import torch

from vllm_omni.model_executor.models.drvexo_speech_8b.general_prefill import GeneralPrefill
from vllm_omni.worker.gpu_model_runner import OmniGPUModelRunner

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@dataclass
class _RunnerModel:
    requires_request_computed_tokens: bool


@dataclass
class _RunnerConfig:
    has_sampling_extra_args: bool


@dataclass
class _InputBatch:
    req_ids: list[str]
    num_computed_tokens_cpu: np.ndarray


@dataclass
class _Offsets:
    cpu: np.ndarray


@dataclass
class _ScoringHead:
    score: Callable


@dataclass
class _RequestState:
    output_token_ids: list[int]


def test_scheduler_spans_offsets_and_buffers_follow_reordered_request_owners():
    captured = []

    def score(hidden, task):
        captured.append(hidden.clone())
        return hidden.clone()

    decision = GeneralPrefill(_ScoringHead(score=score))
    runner = object.__new__(OmniGPUModelRunner)
    runner.model = _RunnerModel(requires_request_computed_tokens=True)
    runner.model_config = _RunnerConfig(has_sampling_extra_args=False)
    runner._omni_query_start_loc_model_kwarg = False
    runner._local_stage_payload_cache = {}

    def info(start, length):
        return {"drvexo_general_decision": {"task": "choice", "prompt_length": length}}

    a, b = info(2, 5), info(1, 3)
    runner.model_intermediate_buffer = {"a": a, "b": b}
    runner.requests = {key: _RequestState(output_token_ids=[]) for key in ("a", "b")}
    runner.input_batch = _InputBatch(req_ids=["a", "b"], num_computed_tokens_cpu=np.array([0, 0]))
    runner.query_start_loc = _Offsets(cpu=np.array([0, 3, 6]))
    runner._omni_num_scheduled_tokens_np = np.array([3, 3])
    args = runner._build_model_kwargs_extra()
    decision.process(
        torch.arange(6.0).reshape(6, 1),
        args["model_intermediate_buffer"],
        args["request_token_spans"],
        args["request_computed_tokens"],
    )
    runner.input_batch.req_ids = ["b", "a"]
    runner.input_batch.num_computed_tokens_cpu = np.array([3, 3])
    runner.query_start_loc.cpu = np.array([0, 1, 3])
    runner._omni_num_scheduled_tokens_np = np.array([1, 2])
    args = runner._build_model_kwargs_extra()
    result = decision.process(
        torch.tensor([[99.0], [3.0], [4.0]]),
        args["model_intermediate_buffer"],
        args["request_token_spans"],
        args["request_computed_tokens"],
    )
    assert [x.item() for x in result["drvexo_general_status"]] == [0, 1]
    torch.testing.assert_close(captured[0], torch.tensor([5.0]))
    torch.testing.assert_close(captured[1], torch.tensor([4.0]))


@pytest.mark.parametrize("async_scheduling", [False, True])
def test_cached_scheduler_offsets_reach_forward_before_current_chunk_is_counted(mocker, async_scheduling):
    from unittest.mock import Mock

    from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
    from vllm.v1.worker.gpu_input_batch import CachedRequestState
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    from vllm_omni.worker import gpu_model_runner as module

    runner = object.__new__(OmniGPUModelRunner)
    runner.model = _RunnerModel(requires_request_computed_tokens=True)
    runner.model_config = _RunnerConfig(has_sampling_extra_args=False)
    runner._omni_query_start_loc_model_kwarg = False
    runner._local_stage_payload_cache = {}
    runner.model_intermediate_buffer = {key: {} for key in ("a", "b")}
    runner.speculative_config = None
    runner.use_async_spec_decode = False
    runner.use_async_scheduling = async_scheduling
    runner.late_interaction_runner = Mock()
    runner.requests = {key: CachedRequestState(key, [1] * 12, [], None, None, ([],), 0, []) for key in ("a", "b")}
    batch = Mock()
    batch.req_ids = ["a", "b"]
    batch.req_id_to_index = {"a": 0, "b": 1}
    batch.num_computed_tokens_cpu = np.array([99, 99])  # Must be replaced by scheduler values.
    runner.input_batch = batch
    mocker.patch.object(module, "get_pp_group", return_value=Mock(is_last_rank=True))

    def reorder(output):
        batch.req_ids.reverse()
        batch.req_id_to_index = {key: i for i, key in enumerate(batch.req_ids)}
        batch.num_computed_tokens_cpu = batch.num_computed_tokens_cpu[::-1].copy()

    runner._may_reorder_batch = reorder
    captured = []

    def forward(**kwargs):
        captured.append(kwargs["request_computed_tokens"])
        return torch.zeros(5, 1)

    mocker.patch.object(GPUModelRunner, "_model_forward", side_effect=forward)
    for offsets, counts in [([0, 0], {"a": 3, "b": 2}), ([3, 2], {"a": 2, "b": 3})]:
        output = SchedulerOutput(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=CachedRequestData(["a", "b"], set(), [], {}, [None, None], offsets, [0, 0]),
            num_scheduled_tokens=counts,
            total_num_scheduled_tokens=sum(counts.values()),
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            finished_req_ids=set(),
            free_encoder_mm_hashes=[],
        )
        runner._update_states(output)
        scheduled = [counts[key] for key in batch.req_ids]
        runner._omni_num_scheduled_tokens_np = np.array(scheduled)
        runner.query_start_loc = _Offsets(cpu=np.cumsum([0, *scheduled]))
        runner._model_forward(None, None, None, None)
        expected = [offsets[["a", "b"].index(key)] for key in batch.req_ids]
        assert captured[-1] == expected
        assert batch.num_computed_tokens_cpu.tolist() == expected
        assert [runner.requests[key].num_computed_tokens for key in batch.req_ids] == expected
