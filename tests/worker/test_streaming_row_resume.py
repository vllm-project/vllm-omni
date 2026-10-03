# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""A streaming session whose prompt only grew is resumed in its input-batch row.

The in-place resume must leave the batch and the cached request state exactly
as the upstream remove and re-add does, while writing only the new prompt tail.
"""

from types import SimpleNamespace

import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm.v1.worker import gpu_input_batch
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from vllm_omni.worker.gpu_model_runner import OmniGPUModelRunner

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_BLOCK = 4
_SAMPLED = 99
_PROMPT_LENS = {"a": 5, "b": 8, "c": 6}


def _params(**kwargs) -> SamplingParams:
    return SamplingParams(temperature=0.0, max_tokens=100, **kwargs)


def _blocks(num_tokens: int) -> tuple[list[int], ...]:
    return (list(range(10, 10 + -(-num_tokens // _BLOCK))),)


def _batch_after_one_step(monkeypatch) -> tuple[InputBatch, dict[str, CachedRequestState]]:
    """Three sessions in the batch, each after the step that sampled its prompt's next token."""
    monkeypatch.setattr(gpu_input_batch, "PIN_MEMORY", False)
    batch = InputBatch(
        max_num_reqs=4,
        max_model_len=64,
        max_num_batched_tokens=64,
        device=torch.device("cpu"),
        vocab_size=1024,
        block_sizes=[_BLOCK],
        kernel_block_sizes=[_BLOCK],
        max_num_blocks_per_req=[16],
    )
    requests = {}
    for req_id, prompt_len in _PROMPT_LENS.items():
        state = CachedRequestState(
            req_id=req_id,
            prompt_token_ids=list(range(1, prompt_len + 1)),
            mm_features=[],
            sampling_params=_params(),
            generator=None,
            block_ids=_blocks(prompt_len),
            num_computed_tokens=prompt_len - 1,
            output_token_ids=[],
        )
        batch.add_request(state)
        requests[req_id] = state
    batch.refresh_metadata()
    # What the step's bookkeeping leaves: the prompt computed and one sampled token.
    for req_id, state in requests.items():
        index = batch.req_id_to_index[req_id]
        state.num_computed_tokens = len(state.prompt_token_ids)
        batch.num_computed_tokens_cpu[index] = state.num_computed_tokens
        batch.token_ids_cpu[index, state.num_tokens] = _SAMPLED
        batch.num_tokens_no_spec[index] = state.num_tokens + 1
        state.output_token_ids.append(_SAMPLED)
    return batch, requests


def _runner(batch: InputBatch, requests: dict[str, CachedRequestState], *, opt_in: bool = True):
    runner = object.__new__(OmniGPUModelRunner)
    runner.model = SimpleNamespace(resume_streaming_rows_in_place=opt_in)
    runner.input_batch = batch
    runner.requests = requests
    runner.uses_mrope = False
    runner.speculative_config = None
    runner.late_interaction_runner = SimpleNamespace(register_request=lambda *_: None)
    return runner


def _extend(state: CachedRequestState, new_tokens: list[int]) -> SimpleNamespace:
    """The scheduler's next streaming update: the same prompt list, extended."""
    prompt = state.prompt_token_ids
    assert prompt is not None
    prompt.extend(new_tokens)
    data = dict(
        req_id=state.req_id,
        prompt_token_ids=prompt,
        mm_features=[],
        sampling_params=_params(),
        pooling_params=None,
        prompt_embeds=None,
        prompt_is_token_ids=None,
        block_ids=_blocks(len(prompt)),
        num_computed_tokens=state.num_computed_tokens,
        lora_request=None,
    )
    return SimpleNamespace(**data)


_SCHEDULED = SimpleNamespace(scheduled_spec_decode_tokens={})


def _row(batch: InputBatch, req_id: str) -> dict:
    index = batch.req_id_to_index[req_id]
    num_tokens = int(batch.num_tokens_no_spec[index])
    table = batch.block_table[0]
    num_blocks = int(table.num_blocks_per_row[index])
    return {
        "index": index,
        "tokens": batch.token_ids_cpu[index, :num_tokens].tolist(),
        "is_token_ids": batch.is_token_ids[index, :num_tokens].tolist(),
        "num_prompt_tokens": int(batch.num_prompt_tokens[index]),
        "num_computed_tokens": int(batch.num_computed_tokens_cpu[index]),
        "blocks": table.block_table.np[index, :num_blocks].tolist(),
        "output_token_ids": list(batch.req_output_token_ids[index]),
        "temperature": float(batch.temperature_cpu[index]),
        "top_k": int(batch.top_k_cpu[index]),
        "greedy": req_id in batch.greedy_reqs,
        "accepted": int(batch.num_accepted_tokens_cpu[index]),
    }


def test_an_in_place_resume_leaves_the_row_the_re_add_leaves(monkeypatch) -> None:
    resumed_batch, resumed = _batch_after_one_step(monkeypatch)
    reference_batch, reference = _batch_after_one_step(monkeypatch)
    # One frame slot per session; "b" spills into a new block.
    updates = {req_id: [0] for req_id in _PROMPT_LENS}

    runner = _runner(resumed_batch, resumed)
    for req_id, tokens in updates.items():
        assert OmniGPUModelRunner._resume_streaming_row_in_place(
            runner, req_id, _extend(resumed[req_id], tokens), _SCHEDULED
        )
    reference_runner = _runner(reference_batch, reference)
    # GPUModelRunner._update_states' order: every streaming update removes its
    # row first, then the updated requests are re-added.
    reqs_to_add = [
        GPUModelRunner._update_streaming_request(reference_runner, req_id, _extend(reference[req_id], tokens))
        for req_id, tokens in updates.items()
    ]
    for state in reqs_to_add:
        reference_batch.add_request(state)
        reference_batch.update_req_spec_token_ids(state, {})
    reference_batch.condense()

    for req_id in _PROMPT_LENS:
        assert _row(resumed_batch, req_id) == _row(reference_batch, req_id), req_id
        ours, theirs = resumed[req_id], reference[req_id]
        assert (ours.num_prompt_tokens, ours.num_computed_tokens, ours.output_token_ids, ours.block_ids) == (
            theirs.num_prompt_tokens,
            theirs.num_computed_tokens,
            theirs.output_token_ids,
            theirs.block_ids,
        )


@pytest.mark.parametrize("case", ["not_opted_in", "new_prompt_list", "penalties"])
def test_anything_but_a_plain_extension_takes_the_re_add_path(monkeypatch, case: str) -> None:
    batch, requests = _batch_after_one_step(monkeypatch)
    state = requests["b"]
    update = _extend(state, [0])
    if case == "new_prompt_list":
        update.prompt_token_ids = list(update.prompt_token_ids)
    elif case == "other_sampling_params":
        update.sampling_params = _params(seed=7)
    elif case == "min_tokens":
        update.sampling_params = state.sampling_params = _params(min_tokens=2)
    elif case == "not_computed":
        update.num_computed_tokens = 0
    elif case == "penalties":
        batch.repetition_penalties_reqs.add("a")
    before = _row(batch, "b")

    resumed = OmniGPUModelRunner._resume_streaming_row_in_place(
        _runner(batch, requests, opt_in=case != "not_opted_in"), "b", update, _SCHEDULED
    )

    assert resumed is False
    assert _row(batch, "b") == before
    assert state.output_token_ids == [_SAMPLED]


def test_a_request_outside_the_batch_takes_the_re_add_path(monkeypatch) -> None:
    batch, requests = _batch_after_one_step(monkeypatch)
    batch.remove_request("b")
    batch.condense()

    assert not OmniGPUModelRunner._resume_streaming_row_in_place(
        _runner(batch, requests), "b", _extend(requests["b"], [0]), _SCHEDULED
    )
    assert "b" not in batch.req_id_to_index
    assert requests["b"].output_token_ids == [_SAMPLED]
