# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm_omni.core.sched.output import OmniCachedRequestData
from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.worker_v2.omni_generation_model_runner import OmniGenerationModelRunner


class _Array:
    def __init__(self):
        self.np = [0]
        self.writes = []
        self.applied = False

    def stage_write_elem(self, idx, value):
        self.writes.append((idx, value))

    def apply_write(self):
        self.applied = True


class _TokenIds:
    def __init__(self):
        self.writes = []

    def stage_write(self, idx, start, values):
        self.writes.append((idx, start, list(values)))


class _ReqStates:
    def __init__(self, has_request=True):
        self.req_id_to_index = {"r1": 0} if has_request else {}
        self.prompt_len = SimpleNamespace(np=[0])
        self.prefill_len = SimpleNamespace(np=[0])
        self.total_len = _Array()
        self.all_token_ids = _TokenIds()
        self.num_computed_tokens = _Array()
        self.num_computed_prefill_tokens = [7]
        self.applied = False
        self.removed = []

    def apply_staged_writes(self):
        self.applied = True

    def remove_request(self, req_id):
        self.removed.append(req_id)
        self.req_id_to_index.pop(req_id, None)


class _IntermediateBuffer:
    def __init__(self):
        self.buffers = [{"stale": "value", "req_id": "r1"}]

    def remove_request(self, idx):
        self.buffers[idx] = {}


def test_async_chunk_update_clears_stale_intermediate_buffer():
    runner = object.__new__(OmniGenerationModelRunner)
    runner.req_states = _ReqStates()
    runner.model_state = SimpleNamespace(intermediate_buffer=_IntermediateBuffer())
    cached = OmniCachedRequestData(
        req_ids=["r1"],
        resumed_req_ids=set(),
        new_token_ids=[],
        all_token_ids=[],
        new_block_ids=[],
        num_computed_tokens=[],
        num_output_tokens=[],
        prompt_token_ids={"r1": [1, 2, 3]},
        additional_information={"r1": {"fresh": "value"}},
    )

    runner._handle_async_chunk_updates(SimpleNamespace(scheduled_cached_reqs=cached))

    assert runner.model_state.intermediate_buffer.buffers[0] == {}
    assert runner.req_states.prompt_len.np[0] == 3
    assert runner.req_states.prefill_len.np[0] == 3
    assert runner.req_states.all_token_ids.writes == [(0, 0, [1, 2, 3])]
    assert runner.req_states.num_computed_prefill_tokens[0] == 0
    assert runner.req_states.applied is True


def test_async_chunk_update_readds_released_cached_request():
    runner = object.__new__(OmniGenerationModelRunner)
    runner.req_states = _ReqStates(has_request=False)
    runner.model_state = SimpleNamespace(intermediate_buffer=_IntermediateBuffer())
    added = []
    runner.add_requests = lambda scheduler_output: added.extend(scheduler_output.scheduled_new_reqs)
    cached = OmniCachedRequestData(
        req_ids=["r1"],
        resumed_req_ids=set(),
        new_token_ids=[],
        all_token_ids={"r1": [1, 2, 3]},
        new_block_ids=[([7],)],
        num_computed_tokens=[0],
        num_output_tokens=[0],
        prompt_token_ids={"r1": [1, 2, 3]},
        additional_information={"r1": {"fresh": "value"}},
    )

    runner._handle_async_chunk_updates(SimpleNamespace(scheduled_cached_reqs=cached))

    assert len(added) == 1
    assert added[0].req_id == "r1"
    assert added[0].prompt_token_ids == [1, 2, 3]
    assert added[0].prefill_token_ids == [1, 2, 3]
    assert added[0].block_ids == ([7],)
    assert added[0].additional_information == {"fresh": "value"}


def _make_runner_for_sample(async_chunk):
    runner = object.__new__(OmniGenerationModelRunner)
    runner._gen_model_output = OmniOutput(
        text_hidden_states=torch.zeros(1, 1),
        multimodal_outputs={"audio": [torch.zeros(1)]},
    )
    runner._gen_input_batch = SimpleNamespace(
        num_reqs=1,
        idx_mapping_np=[0],
        req_ids=["r1"],
    )
    runner._gen_kv_connector_output = None
    runner.execute_model_state = object()
    runner.model_config = SimpleNamespace(async_chunk=async_chunk)
    runner.req_states = _ReqStates()
    runner.model_state = SimpleNamespace(
        intermediate_buffer=_IntermediateBuffer(),
        remove_request=lambda idx: runner.model_state.intermediate_buffer.remove_request(idx),
    )
    removed = []
    runner._remove_request = lambda req_id: removed.append(req_id) or True
    return runner, removed


def test_generation_sample_keeps_runner_slot_without_async_chunk():
    runner, removed = _make_runner_for_sample(async_chunk=False)

    output = runner.sample_tokens()

    assert output.req_ids == ["r1"]
    assert removed == []
    assert runner.model_state.intermediate_buffer.buffers[0] == {"stale": "value", "req_id": "r1"}


def test_generation_sample_releases_runner_slot_after_chunk_output():
    runner, removed = _make_runner_for_sample(async_chunk=True)

    output = runner.sample_tokens()

    assert output.req_ids == ["r1"]
    assert removed == ["r1"]
    assert runner.model_state.intermediate_buffer.buffers[0] == {}


if __name__ == "__main__":
    test_async_chunk_update_clears_stale_intermediate_buffer()
    test_async_chunk_update_readds_released_cached_request()
    test_generation_sample_keeps_runner_slot_without_async_chunk()
    test_generation_sample_releases_runner_slot_after_chunk_output()
