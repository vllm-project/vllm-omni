# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Real MRv2 adapter contracts without allocating the large checkpoint."""

import numpy as np
import pytest
import torch
from torch import nn
from vllm import SamplingParams
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.states import RequestState

from vllm_omni.data_entry_keys import LAST_OUTPUT_TOKEN_KEY, OUTPUT_TOKEN_COUNT_KEY, REQUEST_SAMPLING_PARAMS_KEY
from vllm_omni.model_executor.models.zonos2.configuration_zonos2 import Zonos2Config
from vllm_omni.model_executor.models.zonos2.model_state import Zonos2ModelState, Zonos2Sampler
from vllm_omni.model_executor.models.zonos2.zonos2_keys import FRAMES
from vllm_omni.model_executor.models.zonos2.zonos2_sampler import sample_frame
from vllm_omni.model_executor.models.zonos2.zonos2_talker import (
    Zonos2MultiEmbedder,
    Zonos2TalkerForConditionalGeneration,
)
from vllm_omni.worker_v2.model_states.intermediate_buffer import OmniIntermediateBuffer
from vllm_omni.worker_v2.model_states.omni_model_state import OmniModelState

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def state(mocker):
    model = Zonos2TalkerForConditionalGeneration.__new__(Zonos2TalkerForConditionalGeneration)
    nn.Module.__init__(model)
    model.config = Zonos2Config(dim=2)
    model.multi_embedder = Zonos2MultiEmbedder(model.config)
    model.multi_output = nn.Linear(2, 9 * 1026, bias=False)
    nn.init.zeros_(model.multi_output.weight)
    model._request_states = {}
    model._sampling_plan = []
    model._step_codes = {}
    model._step_payload = None
    instance = Zonos2ModelState.__new__(Zonos2ModelState)
    instance.model = model
    instance.intermediate_buffer = OmniIntermediateBuffer(4)
    instance.last_tokens = {}
    instance._mtp_state = mocker.Mock()
    mocker.patch.object(
        OmniModelState,
        "add_request",
        side_effect=lambda slot, request: instance.intermediate_buffer.add_request(slot, request),
    )
    return instance


def admit(state, mocker, key, slot=0, *, seed=42, replay=0, prompt_len=3):
    frames = torch.full((prompt_len, 10), 1025, dtype=torch.int32)
    frames[:, 9] = 2
    request = mocker.Mock(
        spec=NewRequestData,
        req_id=key,
        prompt_token_ids=[2] * prompt_len,
        prefill_token_ids=[2] * prompt_len + [0] * replay,
        sampling_params=SamplingParams(temperature=1, seed=seed, max_tokens=32),
        additional_information={FRAMES: frames},
        prompt_embeds=None,
        model_intermediate_buffer=None,
        mm_features=[],
    )
    state.add_request(slot, request)


def batch(mocker, keys, slots, offsets, spans, prefill):
    offsets, spans, prefill = (np.asarray(x, dtype=np.int32) for x in (offsets, spans, prefill))
    starts = np.concatenate(([0], spans.cumsum()))
    return mocker.Mock(
        spec=InputBatch,
        req_ids=keys,
        num_reqs=len(keys),
        idx_mapping_np=np.asarray(slots),
        num_computed_tokens_np=offsets,
        num_scheduled_tokens=spans,
        prefill_len_np=prefill,
        is_prefilling_np=offsets < prefill,
        num_computed_prefill_tokens_np=np.minimum(offsets, prefill),
        query_start_loc_np=starts,
        num_tokens=int(spans.sum()),
        num_draft_tokens=0,
        logits_indices=torch.tensor(starts[1:] - 1),
    )


def preprocess(state, mocker, keys, slots, offsets, spans, prefill):
    step = batch(mocker, keys, slots, offsets, spans, prefill)
    requests = mocker.Mock(spec=RequestState)
    requests.prompt_len = mocker.Mock(np=np.full(4, 3, dtype=np.int32))
    inputs = {
        "input_ids": torch.zeros(step.num_tokens, dtype=torch.long),
        "inputs_embeds": torch.empty(step.num_tokens, 2),
    }
    state.run_preprocess(step, inputs, requests)
    # The backbone consumes the same snapshots; use the ordinary forward's
    # request-plan contract without allocating its transformer layers.
    state.model._sampling_plan = [
        (key, inputs["request_sample_eligible"][i], info)
        for i, (key, info) in enumerate(zip(keys, inputs["model_intermediate_buffer"]))
    ]
    return step, inputs


def sample(state, mocker, step):
    return Zonos2Sampler(mocker.Mock(), state).sample_step(
        torch.ones(step.num_tokens, 2), step, mocker.Mock(spec=RequestState), None, mocker.Mock()
    )


def test_final_prefill_and_decode_publish_current_frame_without_transient_buffer_leaks(state, mocker):
    admit(state, mocker, "a")
    step, _ = preprocess(state, mocker, ["a"], [0], [0], [3], [3])
    first = sample(state, mocker, step)
    assert first.num_sampled.tolist() == [1]
    assert first.multimodal_outputs["codes"]["audio"][0].shape == (1, 9)
    first_codes = first.multimodal_outputs["codes"]["audio"][0].clone()
    assert len(state.model._request_states["a"].history) == 1
    step, _ = preprocess(state, mocker, ["a"], [0], [3], [1], [3])
    sample(state, mocker, step)
    assert len(state.model._request_states["a"].history) == 2
    torch.testing.assert_close(first.multimodal_outputs["codes"]["audio"][0], first_codes, rtol=0, atol=0)
    buffer = state.intermediate_buffer.buffers[0]
    assert not {REQUEST_SAMPLING_PARAMS_KEY, OUTPUT_TOKEN_COUNT_KEY, LAST_OUTPUT_TOKEN_KEY}.intersection(buffer)


def test_partial_prefill_does_not_sample_or_advance_codec_history(state, mocker):
    admit(state, mocker, "a")
    step, _ = preprocess(state, mocker, ["a"], [0], [0], [2], [3])
    output = sample(state, mocker, step)
    assert output.num_sampled.tolist() == [0] and output.num_rejected.tolist() == [0]
    assert output.multimodal_outputs["codes"]["audio"][0].numel() == 0
    assert len(state.model._request_states["a"].history) == 0
    assert state.last_tokens["a"] is None


@pytest.mark.parametrize("seed", [42, None])
def test_preemption_rebuilds_full_codec_history_in_a_different_slot(state, mocker, seed):
    admit(state, mocker, "a", seed=seed)
    for offset, span in [(0, 3), (3, 1)]:
        step, _ = preprocess(state, mocker, ["a"], [0], [offset], [span], [3])
        sample(state, mocker, step)
    history = state.model._request_states["a"].history.clone()
    original = state.model._request_states["a"]
    next_codes = sample_frame(
        state.model.fused_audio_logits()[0], history, original.params, original.seed, len(history)
    )
    state.remove_request(0)
    admit(state, mocker, "other", 0, seed=99)
    admit(state, mocker, "a", 2, replay=2, seed=seed)
    partial, _ = preprocess(state, mocker, ["a"], [2], [0], [2], [5])
    assert sample(state, mocker, partial).num_sampled.tolist() == [0]
    torch.testing.assert_close(state.model._request_states["a"].history, history, rtol=0, atol=0)
    final, _ = preprocess(state, mocker, ["a"], [2], [2], [3], [5])
    sample(state, mocker, final)
    assert len(state.model._request_states["a"].history) == 3
    torch.testing.assert_close(state.model._request_states["a"].history[:2], history, rtol=0, atol=0)
    torch.testing.assert_close(state.model._request_states["a"].history[2], next_codes, rtol=0, atol=0)
    assert state.model._request_states["a"].seed == original.seed


@pytest.mark.parametrize("replacement", ["a", "c"])
def test_batch_reordering_and_cancelled_slot_reuse_keep_request_rng_separate(state, mocker, replacement):
    admit(state, mocker, "a", 0, seed=42)
    admit(state, mocker, "b", 3, seed=99)
    step, _ = preprocess(state, mocker, ["b", "a"], [3, 0], [0, 0], [3, 3], [3, 3])
    output = sample(state, mocker, step)
    for row, key in enumerate(["b", "a"]):
        torch.testing.assert_close(
            output.multimodal_outputs["codes"]["audio"][row], state.model._request_states[key].history
        )
    state.model.on_requests_finished({"a"})
    state.on_requests_finished({"a"})
    state.remove_request(0)
    assert "a" not in state.last_tokens and "a" not in state.model._request_states
    admit(state, mocker, replacement, 0, seed=7)
    step, _ = preprocess(state, mocker, [replacement, "b"], [0, 3], [0, 3], [3, 1], [3, 3])
    sample(state, mocker, step)
    assert state.model._request_states[replacement].seed == 7
    assert len(state.model._request_states[replacement].history) == 1
    assert len(state.model._request_states["b"].history) == 2


def test_slot_identity_mismatch_fails_before_preprocess(state, mocker):
    admit(state, mocker, "a", 0)
    with pytest.raises(ValueError, match="identity"):
        preprocess(state, mocker, ["a"], [1], [0], [3], [3])


def test_cancellation_cleans_preempted_request_without_a_live_slot(state, mocker):
    admit(state, mocker, "a")
    step, _ = preprocess(state, mocker, ["a"], [0], [0], [3], [3])
    sample(state, mocker, step)
    state.remove_request(0)
    state.model.on_requests_finished({"a"})
    state.on_requests_finished({"a"})
    assert not state.model._request_states and not state.last_tokens
    assert state.intermediate_buffer.buffers[0] == {}
