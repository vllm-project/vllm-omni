# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Breeze stage 0 through the real Model Runner V2 OmniModelState hooks."""

from dataclasses import dataclass

import numpy as np
import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.request import Request

from tests.model_executor.models.breeze_tts_2_helpers import (
    TalkerConfig,
    _decode_info,
    _embedding_talker,
    _small_talker,
)
from vllm_omni.model_executor.models.breeze_tts_2.first_code_sampler import BreezeFirstCodeSampler
from vllm_omni.model_executor.models.breeze_tts_2.prompt import CFG_UNCOND_SUFFIX
from vllm_omni.model_executor.models.breeze_tts_2.text_encoder_graph import BreezeTextEncoderGraph
from vllm_omni.worker_v2.model_states.intermediate_buffer import OmniIntermediateBuffer
from vllm_omni.worker_v2.model_states.omni_model_state import OmniModelState

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _state(model, max_num_reqs=4):
    # DefaultModelState.__init__ needs a full engine config; these are the
    # attributes OmniModelState's preprocess and output paths read.
    state = object.__new__(OmniModelState)
    state.model = model
    state.has_preprocess = True
    state.has_postprocess = False
    state.have_multimodal_outputs = True
    state.intermediate_buffer = OmniIntermediateBuffer(max_num_reqs)
    state._static_inputs_embeds = None
    return state


@dataclass
class _MRv2Batch:
    """The InputBatch fields read by run_preprocess and postprocess_model_output."""

    idx_mapping_np: np.ndarray
    num_reqs: int
    num_scheduled_tokens: list[int]
    query_start_loc_np: list[int]
    logits_indices: torch.Tensor
    num_computed_tokens_cpu: np.ndarray


@dataclass
class _ReqStates:
    """The RequestState fields read by run_preprocess."""

    prompt_len: np.ndarray
    num_computed_tokens: np.ndarray | None = None


def _batch(slots, tokens, computed):
    starts = np.concatenate(([0], np.cumsum(tokens)[:-1])).tolist()
    return _MRv2Batch(
        idx_mapping_np=np.array(slots),
        num_reqs=len(slots),
        num_scheduled_tokens=list(tokens),
        query_start_loc_np=starts,
        logits_indices=torch.tensor([start + count - 1 for start, count in zip(starts, tokens)]),
        num_computed_tokens_cpu=np.array(computed, dtype=np.int32),
    )


def _add_runner_request(state, slot, req_id, num_tokens):
    # Like vLLM's kernel warmup: the buffer holds only req_id and sampling params.
    request = Request(req_id, list(range(num_tokens)), SamplingParams.for_sampler_warmup(), None)
    state.intermediate_buffer.add_request(
        slot, NewRequestData.from_request(request, block_ids=([1],), prefill_token_ids=request.prompt_token_ids)
    )


def _text_talker(mocker):
    model = _embedding_talker(torch.float32)
    # The T5Gemma2 text encoder is not under test; return one row per token.
    graph = mocker.Mock(spec=BreezeTextEncoderGraph)
    graph.run_batch.side_effect = lambda prompts: [
        prompt.float()[:, None].expand(-1, model.hidden_size) for prompt in prompts
    ]
    model._text_graphs = {(batch, 32): graph for batch in range(1, 5)}
    return model, graph


def _prefill_info(request_id, target_ids, seed, *, role="cond", guidance_scale=1.0):
    return {
        "req_id": request_id,
        "global_request_id": [request_id],
        "breeze_prompt": {"role": role, "guidance_scale": guidance_scale, "target_ids": target_ids},
        "breeze_sampling": {"temperature": 0.9, "top_k": 4, "top_p": 1.0, "repetition_penalty": 1.1},
        "sampling_params": SamplingParams(seed=seed),
    }


@pytest.mark.parametrize(
    "req_ids",
    [("_warmup_0_", "_warmup_1_"), ("_v2_mixed_warmup_decode_", "_v2_mixed_warmup_prefill_")],
)
def test_mrv2_warmup_rows_skip_breeze_output(req_ids):
    model = _small_talker()
    model._next_tokens = torch.tensor([1, 2])
    state = _state(model)
    tokens = [2, 3]
    for slot, (req_id, count) in enumerate(zip(req_ids, tokens)):
        _add_runner_request(state, slot, req_id, count)
    hidden = torch.randn(sum(tokens), model.hidden_size)

    text_hidden, payload = state.postprocess_model_output(hidden, _batch([0, 1], tokens, [0, 0]), None)

    assert text_hidden is hidden
    assert payload == {}
    assert model._next_tokens is None
    assert model.compute_logits(hidden[[1, 4]]).argmax(-1).tolist() == [0, 0]
    model.depth_decoder.generate_frames.assert_not_called()


def test_mrv2_real_row_without_breeze_metadata_fails_closed():
    model = _small_talker()
    state = _state(model)
    _add_runner_request(state, 0, "speech-0", 1)

    with pytest.raises(KeyError, match="global_request_id"):
        state.postprocess_model_output(torch.zeros(1, model.hidden_size), _batch([0], [1], [0]), None)


@pytest.mark.parametrize("leading_prefill", [False, True])
@torch.inference_mode()
def test_mrv2_decode_matches_v1_decode_hook(mocker, leading_prefill):
    model, _ = _text_talker(mocker)
    state = _state(model)
    frame = torch.arange(16, dtype=torch.long).reshape(1, 16) % 7
    for slot, offset in zip((0, 1, 2), (1, 3, 5)):
        info = _decode_info(f"speech-{slot}", (frame + offset) % 7, slot)
        # The runner derives the prefill flag each step; buffers never hold it.
        del info["_omni_is_prefill"]
        state.intermediate_buffer.buffers[slot] = {"req_id": f"speech-{slot}", **info}
    slots, tokens, computed = [2, 0, 1], [1, 1, 1], [9, 9, 9]
    if leading_prefill:
        state.intermediate_buffer.buffers[3] = _prefill_info("speech-3", [4, 5], seed=3)
        slots, tokens, computed = [3, *slots], [2, *tokens], [0, *computed]
    num_tokens = sum(tokens)
    placeholder_ids = torch.arange(num_tokens, dtype=torch.long) + 100
    inputs = {"input_ids": placeholder_ids.clone(), "inputs_embeds": torch.zeros(num_tokens, model.hidden_size)}
    req_states = _ReqStates(prompt_len=np.array([4, 4, 4, 2]))

    state.run_preprocess(_batch(slots, tokens, computed), inputs, req_states)

    decode_infos = [state.intermediate_buffer.buffers[slot] for slot in (2, 0, 1)]
    _, expected, _ = model.preprocess_decode_batch(input_ids=placeholder_ids[-3:], req_infos=decode_infos)
    torch.testing.assert_close(inputs["inputs_embeds"][-3:], expected, rtol=0, atol=0)
    assert torch.equal(inputs["input_ids"], placeholder_ids)
    if leading_prefill:
        expected_prefill = torch.tensor([4.0, 5.0])[:, None].expand(-1, model.hidden_size)
        torch.testing.assert_close(inputs["inputs_embeds"][:2], expected_prefill, rtol=0, atol=0)


@torch.inference_mode()
def test_mrv2_cfg_cycle_keeps_breeze_state_in_runner_buffers(mocker):
    model, graph = _text_talker(mocker)
    model.config = TalkerConfig(vocab_size=8, eos_token_id=7)
    model.codebook_size = 6
    model._first_code_sampler = BreezeFirstCodeSampler()
    model.lm_head = torch.nn.Linear(model.hidden_size, 8, bias=False)
    with torch.no_grad():
        # Keep generation EOS out of reach so every step produces a frame.
        model.lm_head.weight.uniform_(0.0, 1.0, generator=torch.Generator().manual_seed(5))
        model.lm_head.weight[7] = -10.0
        # Column 0 tags hidden rows without changing logits, so CFG cannot pick EOS.
        model.lm_head.weight[:, 0] = 0.0
    frames = mocker.Mock(side_effect=lambda hidden, first, **_: first[:, None].repeat(1, 16))
    model.depth_decoder.generate_frames = frames
    state = _state(model)
    buffers = state.intermediate_buffer.buffers
    cond, plain, uncond = 2, 0, 1
    buffers[cond] = _prefill_info("cfg", [1, 2, 3], 1, guidance_scale=2.0)
    buffers[plain] = _prefill_info("plain", [4, 5], 2)
    buffers[uncond] = _prefill_info(f"cfg{CFG_UNCOND_SUFFIX}", [6], 1, role="uncond", guidance_scale=2.0)
    seeds = {cond: 1, plain: 2, uncond: 1}
    req_states = _ReqStates(prompt_len=np.array([2, 1, 3, 0]))
    slots, tokens, computed = [uncond, cond, plain], [1, 3, 2], [0, 0, 0]

    for step in range(4):
        num_tokens = sum(tokens)
        batch = _batch(slots, tokens, computed)
        input_ids = torch.zeros(num_tokens, dtype=torch.long)
        inputs = {"input_ids": input_ids, "inputs_embeds": torch.zeros(num_tokens, model.hidden_size)}
        state.run_preprocess(batch, inputs, req_states)
        assert inputs["input_ids"] is input_ids
        if step == 0:
            # Prefill encodes every scheduled prompt in one text-encoder batch.
            graph.run_batch.assert_called_once()
            target_rows = torch.tensor([6, 1, 2, 3, 4, 5], dtype=torch.float32)
            expected = target_rows[:, None].expand(-1, model.hidden_size)
            torch.testing.assert_close(inputs["inputs_embeds"], expected, rtol=0, atol=0)
            for slot, seed in seeds.items():
                assert buffers[slot]["breeze_prepared"] == {}
                breeze_state = buffers[slot]["breeze_state"]
                assert breeze_state["history"].shape == (1, 0)
                assert breeze_state["current"].shape == (1, model.num_codebooks)
                assert breeze_state["generator"].initial_seed() == seed
        else:
            # Both CFG branches embed the shared frame.
            rows = {slot: row for row, slot in enumerate(slots)}
            assert torch.equal(inputs["inputs_embeds"][rows[cond]], inputs["inputs_embeds"][rows[uncond]])
        hidden = torch.ones(num_tokens, model.hidden_size)
        hidden[:, 0] = torch.arange(num_tokens)
        _, payload = state.postprocess_model_output(hidden, batch, req_states)

        codes = dict(zip(slots, payload["codes"]["audio"], strict=True))
        assert model._next_tokens.tolist() == [0, 0, 0]
        assert codes[uncond].shape == (0, 16)
        for slot in (cond, plain):
            breeze_state = buffers[slot]["breeze_state"]
            assert breeze_state["history"].shape == (1, step + 1)
            assert torch.equal(codes[slot], breeze_state["current"])
        assert buffers[uncond]["breeze_state"]["current"] is buffers[cond]["breeze_state"]["current"]
        # The companion pairs with its conditioned row through global_request_id.
        guided = next(call for call in frames.call_args_list[-2:] if call.kwargs["guidance_scale"] == 2.0)
        last_rows = dict(zip(slots, (np.cumsum(tokens) - 1).tolist(), strict=True))
        torch.testing.assert_close(guided.args[0], hidden[[last_rows[cond], last_rows[uncond]]], rtol=0, atol=0)
        slots, tokens, computed = [plain, uncond, cond] if step % 2 == 0 else [cond, plain, uncond], [1, 1, 1], [9] * 3

    # Decode steps never re-encode the text prompts.
    graph.run_batch.assert_called_once()
