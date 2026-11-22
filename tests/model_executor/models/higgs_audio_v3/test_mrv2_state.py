# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from vllm.sampling_params import SamplingParams

from vllm_omni.model_executor.models.higgs_audio_v3.higgs_audio_v3_talker import (
    HiggsAudioV3TalkerForConditionalGeneration,
)
from vllm_omni.model_executor.models.higgs_audio_v3.model_state import HiggsModelState
from vllm_omni.worker_v2.model_states.intermediate_buffer import OmniIntermediateBuffer

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def make_state():
    state = object.__new__(HiggsModelState)
    state.sampling_slots = None
    state.request_slots = {}
    state.device = torch.device("cpu")
    state.rope_state = state.prompt_embeds_state = state._static_inputs_embeds = None
    state.requests = {}
    state.generators = {}
    state.metadata_key = state.metadata = None
    state.intermediate_buffer = OmniIntermediateBuffer(4)
    state.model = SimpleNamespace(
        on_requests_finished=Mock(),
        _resolve_token_ids=Mock(),
        _audio_continuation_id=99,
        update_decode_step_metadata=Mock(),
    )
    return state


def request(name, seed, **kwargs):
    return SimpleNamespace(
        req_id=name, prompt_token_ids=[1, 2, 99], mm_features=[], sampling_params=SamplingParams(seed=seed, **kwargs)
    )


@pytest.mark.parametrize("slots", [False, True])
def test_sampling_follows_request_identity_and_reinitializes_reused_slots(slots):
    s = make_state()
    if slots:
        s.sampling_slots = {
            name: torch.zeros(4, dtype=torch.long if name == "top_k" else torch.float32)
            for name in ("temperature", "top_p", "top_k", "zeros", "ones")
        }
        s.sampling_slots["ones"].fill_(1)
    s.add_request(0, request("a", 42, temperature=0.5))
    s.add_request(2, request("b", 99, temperature=0.9))
    a = s.generators["a"]
    b = s.generators["b"]
    first = s.sampling_metadata(["a", "b"])
    assert first.generators == {0: a, 1: b}
    torch.testing.assert_close(first.temperature, torch.tensor([0.5, 0.9]))
    swapped = s.sampling_metadata(["b", "a"])
    assert swapped.generators == {0: b, 1: a}
    torch.testing.assert_close(swapped.temperature, torch.tensor([0.9, 0.5]))
    s.remove_request("a")
    s.add_request(0, request("new", 7))
    assert "a" not in s.generators and "a" not in s.requests
    s.model.on_requests_finished.assert_called_once_with({"a"})
    assert s.sampling_metadata(["new"]).generators[0].initial_seed() == 7


@pytest.mark.parametrize(
    "kwargs",
    [
        {"repetition_penalty": 1.2},
        {"logprobs": 1},
        {"min_p": 0.1},
        {"frequency_penalty": 0.2},
        {"prompt_logprobs": 1},
        {"min_tokens": 1},
    ],
)
def test_unsupported_sampling_is_rejected_before_admission(kwargs):
    s = make_state()
    with pytest.raises(ValueError, match="temperature/top-k/top-p"):
        s.add_request(0, request("a", 42, **kwargs))
    assert not s.requests


def test_partial_prefill_not_classified_as_audio_and_slot_order_is_authoritative():
    s = make_state()
    s.add_request(0, request("a", 42))
    s.add_request(2, request("b", 99))
    batch = SimpleNamespace(
        req_ids=["b", "a"],
        idx_mapping_np=np.array([2, 0]),
        num_scheduled_tokens=np.array([1, 1]),
        positions=torch.tensor([2, 1]),
        query_start_loc=torch.tensor([0, 1, 2]),
        num_reqs=2,
        num_tokens=2,
    )
    states = SimpleNamespace(num_computed_prefill_tokens=np.array([1, 0, 2, 0]))
    s.run_preprocess(batch, {"input_ids": torch.tensor([99, 2])}, states)
    assert s.model.update_decode_step_metadata.call_args.kwargs["audio_prompt_mode_rows"] == 0
    states.num_computed_prefill_tokens[0] = 2
    s.run_preprocess(batch, {"input_ids": torch.tensor([99, 99])}, states)
    assert s.model.update_decode_step_metadata.call_args.kwargs["audio_prompt_mode_rows"] == 2


def test_finalizer_filters_unsampled_prefill_and_keeps_owned_audio():
    s = make_state()
    s.model = object.__new__(HiggsAudioV3TalkerForConditionalGeneration)
    torch.nn.Module.__init__(s.model)
    s.model.num_codebooks = 2
    snapshot = torch.tensor([[11, 12, 1, 0], [21, 22, 1, 0], [31, 32, 0, 1]])
    payload = {"_higgs_audio_snapshot": snapshot, "_higgs_invalid_rows": ()}
    result = s.finalize_audio_snapshot(payload, [1, 0, 1])
    snapshot.zero_()
    assert [x.tolist() for x in result["codes"]["audio"]] == [[[11, 12]], [], []]
    assert payload["_higgs_invalid_rows"] == ()


def test_upstream_warmup_retains_generic_text_sampling():
    s = make_state()
    s.add_request(0, request("_warmup_0_", 0, repetition_penalty=1.2, logprobs=1))
    batch = SimpleNamespace(req_ids=["_warmup_0_"])
    sampler, rejection = s.custom_sampler(Mock())
    expected = (object(), torch.ones(1), torch.zeros(1))
    standard = Mock(return_value=expected)
    output = sampler.sample_step(None, batch, None, None, standard)
    assert output.sampler_output is expected[0] and output.multimodal_outputs is None
    assert rejection is None
    standard.assert_called_once_with(None, batch, None)
    s.remove_request("_warmup_0_")
    assert not s.requests


def test_static_embeddings_exclude_padding_and_preserve_storage():
    s = make_state()
    embed = torch.nn.Embedding(100, 3)
    s.model.model = SimpleNamespace(embed_tokens=embed)
    s.model._apply_audio_feedback = Mock(side_effect=lambda x, ids: x + 2)
    s.model._apply_ref_audio_substitution = Mock(side_effect=lambda x, *args: x + 5)
    s._static_inputs_embeds = torch.full((8, 3), 999.0)
    ptr = s._static_inputs_embeds.data_ptr()
    ids = torch.tensor([1, 2, 3, 0])
    batch = SimpleNamespace(num_tokens=3, num_reqs=3, num_tokens_after_padding=4, has_prefill=False)
    s._prepare_static_embeddings(batch, {"input_ids": ids})
    assert s._static_inputs_embeds.data_ptr() == ptr
    torch.testing.assert_close(s._static_inputs_embeds[:3], embed(ids[:3]) + 2)
    assert s._static_inputs_embeds[3].count_nonzero() == 0
    assert s.model._apply_audio_feedback.call_args.args[1].numel() == 3
    s.model._apply_ref_audio_substitution.assert_not_called()
    batch.has_prefill = True
    batch.positions = torch.arange(4)
    s._prepare_static_embeddings(batch, {"input_ids": ids, "model_intermediate_buffer": [{}]})
    torch.testing.assert_close(s._static_inputs_embeds[:3], embed(ids[:3]) + 7)


def test_mixed_feedback_updates_only_single_token_spans_with_codes():
    s = make_state()
    s.model._ensure_decode_state_capacity = Mock()
    s.model._decode_last_codes = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]])
    s.model._decode_has_codes = torch.tensor([True, True, False, True])
    s.model.multimodal_embedding = lambda codes: codes.sum(1)[:, None].expand(-1, 3)
    batch = SimpleNamespace(num_tokens=7, num_reqs=4, query_start_loc=torch.tensor([0, 1, 4, 5, 7]))
    embeds = torch.arange(21).reshape(7, 3).float()
    output = s._apply_audio_feedback(embeds, torch.ones(7).long(), batch)
    expected = embeds.clone()
    expected[0] = 3
    torch.testing.assert_close(output, expected)
    assert embeds[0].tolist() == [0, 1, 2]


def test_direct_payload_owns_rows_and_filters_partial_prefill():
    s = make_state()
    s.direct_payload = True
    s.model.num_codebooks = 2
    snapshot = torch.tensor([[11, 12, 1, 0], [21, 22, 1, 0], [31, 32, 0, 1]])
    out = s.finalize_audio_snapshot({"_higgs_audio_snapshot": snapshot}, [1, 0, 1])
    snapshot.zero_()
    assert out.inter_stage[0]["codes.audio"].tolist() == [[11, 12]]
    assert out.inter_stage[1:] == [None, None]


def test_registered_audio_sampler_preserves_rows_counts_and_owned_output(mocker):
    state = make_state()
    metadata, payload = object(), {"_higgs_audio_snapshot": torch.ones(2, 4)}
    state.sampling_metadata = Mock(return_value=metadata)
    state.model.compute_logits = Mock(return_value=torch.ones(2, 8))
    state.model.sample = Mock(
        return_value=SimpleNamespace(sampled_token_ids=torch.tensor([[9], [8]]), logprobs_tensors=None)
    )
    state.model.post_sample_multimodal_outputs = Mock(return_value=payload)
    count, rejected = torch.tensor([1, 0]), torch.tensor([0, 0])
    mocker.patch(
        "vllm_omni.model_executor.models.higgs_audio_v3.model_state.get_num_sampled_and_rejected",
        return_value=(count, rejected),
    )
    batch = SimpleNamespace(
        req_ids=["b", "a"],
        num_reqs=2,
        num_draft_tokens=0,
        idx_mapping=torch.tensor([3, 1]),
        logits_indices=torch.tensor([2, 0]),
        seq_lens=None,
        cu_num_logits=None,
    )
    hidden = torch.arange(12).reshape(3, 4).float()
    standard = Mock(side_effect=AssertionError("must not sample twice"))
    sampler, rejection = state.custom_sampler(Mock())
    result = sampler.sample_step(hidden, batch, SimpleNamespace(prefill_len=SimpleNamespace(gpu=None)), None, standard)
    state.sampling_metadata.assert_called_once()
    torch.testing.assert_close(state.model.compute_logits.call_args.args[0], hidden[[2, 0]])
    assert state.model.compute_logits.call_args.args[1] is metadata
    assert result.num_sampled is count and result.num_rejected is rejected
    assert result.multimodal_outputs is payload and result.owns_multimodal_outputs
    assert not result.include_hidden_states and result.finalize_multimodal == state.finalize_audio_snapshot
    assert rejection is None
    standard.assert_not_called()


@pytest.mark.parametrize("draft,grammar", [(1, None), (0, object())])
def test_registered_audio_sampler_rejects_unsupported_modes(draft, grammar):
    sampler, _ = make_state().custom_sampler(Mock())
    batch = SimpleNamespace(req_ids=["real-request"], num_draft_tokens=draft)
    with pytest.raises(ValueError, match="grammar or speculative"):
        sampler.sample_step(None, batch, None, grammar, Mock())
