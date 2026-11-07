# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""How the MRv2 state feeds the MiniMax Music 3 talker its V1 hook inputs.

The talker keeps one implementation of row binding, frame feedback and the
guided draw for both runners; these tests pin what the MRv2 state hands it.
The base-class hooks and the Triton counting kernel are stubbed, so no GPU or
engine is needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from unittest.mock import patch

import numpy as np
import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu.sample.output import SamplerOutput

from vllm_omni.model_executor.models.minimax_music3 import model_state as music3_state
from vllm_omni.model_executor.models.minimax_music3.model_state import MiniMaxMusic3TalkerModelState
from vllm_omni.worker_v2.model_states.omni_model_state import OmniModelState

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@dataclass
class _Drawn:
    sampled_token_ids: torch.Tensor


class _Talker:
    publishes_windows = True

    def __init__(self) -> None:
        self.bound: dict | None = None
        self.logits_input: torch.Tensor | None = None
        self.tokens: torch.Tensor | None = None

    def prepare_runner_inputs(self, *, req_ids, num_computed_tokens, num_scheduled_tokens, input_ids, positions):
        self.bound = {
            "req_ids": req_ids,
            "num_computed_tokens": num_computed_tokens,
            "num_scheduled_tokens": num_scheduled_tokens,
        }
        return input_ids, positions

    def compute_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        self.logits_input = hidden
        return hidden.sum(dim=-1, keepdim=True)

    def sample(self, logits: torch.Tensor, sampling_metadata: None) -> _Drawn | None:
        assert sampling_metadata is None
        return None if self.tokens is None else _Drawn(self.tokens)


@dataclass
class _PrepareBatch:
    """The ``InputBatch`` fields ``prepare_inputs`` reads."""

    num_reqs: int
    req_ids: list[str]
    idx_mapping_np: np.ndarray
    num_computed_tokens_np: np.ndarray
    num_scheduled_tokens: np.ndarray
    prefill_len_np: np.ndarray
    input_ids: torch.Tensor
    positions: torch.Tensor


@dataclass
class _SampleBatch:
    """The ``InputBatch`` fields ``sample_step`` reads."""

    req_ids: list[str]
    logits_indices: torch.Tensor
    num_draft_tokens: int = 0
    num_reqs: int = field(init=False)
    seq_lens: torch.Tensor = field(init=False)
    cu_num_logits: torch.Tensor = field(init=False)
    idx_mapping: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        self.num_reqs = len(self.req_ids)
        self.seq_lens = torch.full((self.num_reqs,), 104)
        self.cu_num_logits = torch.arange(self.num_reqs + 1)
        self.idx_mapping = torch.arange(self.num_reqs)


@dataclass
class _Staged:
    gpu: torch.Tensor


@dataclass
class _ReqStates:
    prefill_len: _Staged


_REQ_STATES = _ReqStates(_Staged(torch.tensor([100, 100])))


def _state() -> MiniMaxMusic3TalkerModelState:
    state = object.__new__(MiniMaxMusic3TalkerModelState)
    state.model = _Talker()
    return state


def _batch(slots, computed, scheduled, prefill_len) -> _PrepareBatch:
    num_tokens = int(sum(scheduled))
    return _PrepareBatch(
        num_reqs=len(slots),
        req_ids=[f"r{slot}" for slot in slots],
        idx_mapping_np=np.array(slots),
        num_computed_tokens_np=np.array(computed),
        num_scheduled_tokens=np.array(scheduled),
        prefill_len_np=np.array(prefill_len),
        input_ids=torch.zeros(num_tokens, dtype=torch.int32),
        positions=torch.arange(num_tokens),
    )


def _prepare(state, batch, buffers):
    with patch.object(OmniModelState, "prepare_inputs", return_value={"model_intermediate_buffer": buffers}):
        return state.prepare_inputs(batch, req_states=None)


def test_rows_are_bound_with_this_steps_counts_and_their_requests_extra_args() -> None:
    state = _state()
    # The base state gathers the buffers in batch order, which is not slot
    # order: slot 4 finishes its prompt this step, slot 1 is mid chunked
    # prefill, slot 2 is decoding, slot 0 has no sampling params.
    buffers = [
        {"sampling_params": SamplingParams(extra_args={"cfg_role": "cond", "cfg_scale": 1.5, "tts_local_seed": 1})},
        {"sampling_params": SamplingParams(extra_args={"cfg_role": "uncond", "cfg_scale": 1.5, "tts_local_seed": 1})},
        {"sampling_params": SamplingParams()},
        {},
    ]
    batch = _batch(
        slots=[4, 1, 2, 0], computed=[0, 10, 107, 107], scheduled=[105, 50, 1, 1], prefill_len=[105, 105, 107, 107]
    )

    inputs = _prepare(state, batch, buffers)

    bound = state.model.bound
    assert bound["req_ids"] == ["r4", "r1", "r2", "r0"]
    assert list(bound["num_computed_tokens"]) == [0, 10, 107, 107]
    assert list(bound["num_scheduled_tokens"]) == [105, 50, 1, 1]
    assert [args.get("cfg_role") for args in inputs["sampling_extra_args"]] == ["cond", "uncond", None, None]
    assert inputs["request_sample_eligible"] == [True, False, True, True]
    assert inputs["model_intermediate_buffer"] is buffers


class _StockSample:
    """Stands in for the runner's standard sampling path."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, hidden_states, input_batch, grammar_output):
        self.calls += 1
        counts = torch.ones(input_batch.num_reqs, dtype=torch.int32)
        tokens = torch.zeros(input_batch.num_reqs, 1, dtype=torch.int64)
        return SamplerOutput(tokens, None, None, counts, counts * 0), counts, counts * 0


def _sampler() -> music3_state.MiniMaxMusic3Sampler:
    state = _state()
    sampler, rejection_sampler = state.custom_sampler(object())
    assert rejection_sampler is None
    assert sampler.talker is state.model
    return sampler


@pytest.mark.parametrize("async_chunk", [True, False])
def test_sampling_reads_this_steps_hidden_states_at_the_logit_rows(async_chunk: bool) -> None:
    sampler = _sampler()
    sampler.talker.publishes_windows = async_chunk
    hidden = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    stock = _StockSample()
    sampler.talker.tokens = torch.tensor([[7], [7]])

    with patch.object(
        music3_state,
        "get_num_sampled_and_rejected",
        side_effect=lambda num_sampled, *_: (num_sampled, torch.zeros_like(num_sampled)),
    ):
        output = sampler.sample_step(hidden, _SampleBatch(["r0", "r1"], torch.tensor([3, 5])), _REQ_STATES, None, stock)

    torch.testing.assert_close(sampler.talker.logits_input, hidden[[3, 5]])
    assert output.sampler_output.sampled_token_ids.tolist() == [[7], [7]]
    assert output.sampler_output.logprobs_tensors is None
    assert output.num_sampled.tolist() == [1, 1]
    # The forward's window payload is kept, and the backbone hidden state is
    # not published next to it (V1 honors omni_pooler_payload_include_hidden).
    assert output.multimodal_outputs is None
    assert output.include_hidden_states is False
    # Without async_chunk and without hidden states, the V2 runner builds no
    # per-request payload unless a finalizer is set.
    if async_chunk:
        assert output.finalize_multimodal is None
    else:
        payload = {"latent": torch.ones(2, 3)}
        assert output.finalize_multimodal(payload, [1, 1]) is payload
    assert stock.calls == 0


def test_a_step_the_talker_declines_falls_back_to_the_stock_sampler() -> None:
    sampler = _sampler()
    stock = _StockSample()

    sampler.sample_step(torch.zeros(2, 2), _SampleBatch(["r0"], torch.tensor([1])), _REQ_STATES, None, stock)

    assert stock.calls == 1


@pytest.mark.parametrize("req_ids", [["_warmup_0_", "_warmup_1_"], ["_v2_mixed_warmup_decode_0"]])
def test_vllm_warmup_requests_go_to_the_stock_sampler_without_the_talker(req_ids) -> None:
    sampler = _sampler()
    sampler.talker.tokens = torch.tensor([[7], [7]])
    stock = _StockSample()
    batch = _SampleBatch(req_ids, torch.zeros(len(req_ids), dtype=torch.long))

    sampler.sample_step(torch.zeros(2, 2), batch, _REQ_STATES, None, stock)

    assert stock.calls == 1
    assert sampler.talker.logits_input is None


def test_speculative_decoding_is_refused_rather_than_sampled_by_the_stock_sampler() -> None:
    sampler = _sampler()
    stock = _StockSample()
    batch = _SampleBatch(["r0"], torch.tensor([1]), num_draft_tokens=1)

    with pytest.raises(ValueError, match="speculative"):
        sampler.sample_step(torch.zeros(2, 2), batch, _REQ_STATES, None, stock)

    assert stock.calls == 0
