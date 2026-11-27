# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu.sample.output import SamplerOutput

from vllm_omni.model_executor.models.higgs_audio_v3.higgs_audio_v3_talker import (
    HiggsAudioV3TalkerForConditionalGeneration,
)
from vllm_omni.model_executor.models.higgs_audio_v3.model_state import HiggsModelState
from vllm_omni.model_executor.models.higgs_audio_v3.spec_decode import DraftBatch, HiggsAudioSpeculator
from vllm_omni.worker_v2.model_states.intermediate_buffer import OmniIntermediateBuffer
from vllm_omni.worker_v2.model_states.mtp_model_state import MTPState

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture(autouse=True)
def disable_pin_memory(monkeypatch):
    monkeypatch.setattr("vllm.utils.torch_utils.PIN_MEMORY", False)


def make_state():
    state = object.__new__(HiggsModelState)
    state.sampling_slots = None
    state.request_slots = {}
    state.device = torch.device("cpu")
    state.rope_state = state.prompt_embeds_state = state._static_inputs_embeds = None
    state.requests = {}
    state.generators = {}
    state.audio_history = None
    state.replay = {}
    state.parked = {}
    state.restored_rows = None
    state.metadata_key = state.metadata = None
    state.intermediate_buffer = OmniIntermediateBuffer(4)
    state.model = SimpleNamespace(
        on_requests_finished=Mock(),
        _resolve_token_ids=Mock(),
        _audio_continuation_id=99,
        update_decode_step_metadata=Mock(),
    )
    state._embed_dim = 0
    state.device = torch.device("cpu")
    state.model.has_talker_mtp = False
    state._mtp_state = MTPState(state)
    return state


def request(name, seed, **kwargs):
    return SimpleNamespace(
        req_id=name,
        prompt_token_ids=[1, 2, 99],
        prefill_token_ids=[1, 2, 99],
        mm_features=[],
        sampling_params=SamplingParams(seed=seed, **kwargs),
    )


def test_higgs_request_does_not_reserve_a_draft_during_kv_replay():
    state = make_state()
    spec = object.__new__(HiggsAudioSpeculator)
    spec.pending, spec.device_rng, spec.single_step = {}, {}, set()
    spec.max_model_len = 128
    spec.first_window_rows = 0
    state.audio_speculator = spec
    state.add_request(0, request("fresh", 42, max_tokens=100))
    resumed = request("resumed", 43)
    resumed.prefill_token_ids = [1, 2, 99, 99]
    state.add_request(1, resumed)
    batch = SimpleNamespace(
        req_ids=["resumed", "fresh"],
        num_reqs=2,
        idx_mapping_np=np.array([1, 0]),
        logits_indices=torch.arange(3),
        cu_num_logits=torch.tensor([0, 1, 3]),
        positions=torch.tensor([17, 21, 22]),
    )
    model = state.model
    model.num_codebooks = 8
    model._step_audio_mode_rows = 2
    model._decode_has_codes = torch.ones(2, dtype=torch.bool)
    model._decode_delay_count = torch.full((2,), 7)
    model._decode_eoc_countdown = torch.full((2,), -1)
    rows, token = spec.reserve_audio_drafts(state, batch, torch.tensor([1, 2]), 1)
    assert rows.tolist() == [1] and token == 99
    slot, position = spec.pending["fresh"]
    assert slot == 0 and position.item() == 23
    assert "resumed" not in spec.pending
    state.remove_request("resumed")
    assert not spec.single_step


def test_higgs_first_window_gate_waits_for_one_ready_request():
    state = make_state()
    spec = object.__new__(HiggsAudioSpeculator)
    spec.pending, spec.single_step = {}, set()
    spec.max_model_len = 128
    spec.first_window_rows = 15
    state.audio_speculator = spec
    for slot, name in enumerate(("ready", "early")):
        state.add_request(slot, request(name, 42, max_tokens=100))
    model = state.model
    model.num_codebooks = 8
    model._step_audio_mode_rows = 2
    model._decode_has_codes = torch.ones(2, dtype=torch.bool)
    model._decode_delay_count = torch.full((2,), 7)
    model._decode_eoc_countdown = torch.full((2,), -1)
    batch = SimpleNamespace(
        req_ids=["early", "ready"],
        num_reqs=2,
        idx_mapping_np=np.array([1, 0]),
        num_computed_tokens_np=np.array([15, 15]),
        num_scheduled_tokens=np.array([2, 1]),
        num_draft_tokens=1,
        num_draft_tokens_per_req=np.array([1, 0]),
        logits_indices=torch.arange(3),
        cu_num_logits=torch.tensor([0, 2, 3]),
        positions=torch.tensor([15, 16, 15]),
    )
    assert spec.reserve_audio_drafts(state, batch, torch.tensor([1, 1]), 1) is None
    assert not spec.pending

    batch.num_computed_tokens_np[1] = 16
    batch.positions[2] = 16
    rows, _ = spec.reserve_audio_drafts(state, batch, torch.tensor([1, 1]), 1)
    assert rows.tolist() == [0, 1]
    assert set(spec.pending) == {"early", "ready"}


def test_verifier_commits_audio_and_rng_only_for_an_accepted_row(monkeypatch):
    state = make_state()
    spec = object.__new__(HiggsAudioSpeculator)
    rows = torch.arange(3)
    candidates = torch.full((3, 8), 11)
    delay, remaining = torch.full((3,), 7), torch.full((3,), -1)
    spec.batch = DraftBatch(
        ["accept", "reject", "stale"], rows, rows, candidates, delay, remaining, torch.tensor([True, True, False])
    )
    model = state.model
    model.num_codebooks = 8
    model.multimodal_embedding = SimpleNamespace(vocab_size=1026)
    model._get_codebook_indices = lambda device: torch.arange(8, device=device)[None]
    first = candidates.clone()
    first[1, 7] = 90
    model._decode_last_codes = first.clone()
    model._decode_delay_count = delay.clone()
    model._decode_eoc_countdown = remaining.clone()
    model._decode_has_codes = torch.ones(3, dtype=torch.bool)
    model._decode_generation_done = torch.zeros(3, dtype=torch.bool)
    state.generators = {r: Mock(get_offset=Mock(return_value=100)) for r in spec.batch.req_ids}
    state.sampling_metadata = Mock()
    bonus = torch.full((3, 8), 22)
    spec._replay_bonus = Mock(return_value=(bonus, bonus, delay, remaining, torch.zeros(3, dtype=torch.bool)))
    monkeypatch.setattr(
        "vllm_omni.model_executor.models.higgs_audio_v3.spec_decode.fill_exponential_steps", lambda *args: 4
    )
    counts = torch.ones(3, dtype=torch.int32)
    payload = {"_higgs_audio_snapshot": torch.cat((first, torch.ones(3, 2, dtype=first.dtype)), dim=1)}
    spec.verify(state, torch.zeros((3, 8)), torch.full((3, 1), 99), counts, payload)
    assert counts.tolist() == [2, 1, 1]
    assert spec.batch is None
    assert torch.equal(model._decode_last_codes, torch.stack((bonus[0], first[1], first[2])))
    state.generators["accept"].set_offset.assert_called_once_with(104)
    for req_id in ("reject", "stale"):
        state.generators[req_id].set_offset.assert_not_called()
    output = state.finalize_audio_snapshot(payload, counts.tolist()).inter_stage
    assert [item["codes.audio"].tolist() for item in output] == [
        [first[0].tolist(), bonus[0].tolist()],
        [first[1].tolist()],
        [first[2].tolist()],
    ]


@pytest.mark.parametrize("slots", [False, True])
def test_sampling_follows_request_identity_and_reinitializes_reused_slots(slots):
    s = make_state()
    if slots:
        s.sampling_slots = {
            name: torch.zeros(4, dtype=torch.long if name == "top_k" else torch.float32)
            for name in ("temperature", "top_p", "top_k")
        }
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


@pytest.mark.parametrize("req_id", ["_warmup_0_", "_v2_mixed_warmup_decode_", "_v2_mixed_warmup_prefill_"])
def test_upstream_warmup_retains_generic_text_sampling(req_id):
    s = make_state()
    s.add_request(0, request(req_id, 0, repetition_penalty=1.2, logprobs=1))
    batch = SimpleNamespace(req_ids=[req_id])
    sampler, rejection = s.custom_sampler(Mock())
    expected = (object(), torch.ones(1), torch.zeros(1))
    standard = Mock(return_value=expected)
    output = sampler.sample_step(None, batch, None, None, standard)
    assert output.sampler_output is expected[0] and output.multimodal_outputs is None
    assert rejection is None
    standard.assert_called_once_with(None, batch, None)
    s.remove_request(req_id)
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


@pytest.mark.parametrize("has_reserved_draft", [False, True])
def test_mixed_feedback_keeps_base_embedding_for_an_unsupplied_draft(has_reserved_draft):
    s = make_state()
    s.model._ensure_decode_state_capacity = Mock()
    s.model._decode_last_codes = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]])
    s.model._decode_has_codes = torch.tensor([True, True, False, True])
    s.model.multimodal_embedding = lambda codes: codes.sum(1)[:, None].expand(-1, 3)
    batch = SimpleNamespace(
        num_tokens=7,
        num_reqs=4,
        query_start_loc=torch.tensor([0, 1, 4, 5, 7]),
        num_draft_tokens=int(has_reserved_draft),
        num_draft_tokens_per_req=np.array([0, 0, 0, int(has_reserved_draft)]),
    )
    embeds = torch.arange(21).reshape(7, 3).float()
    output = s._apply_audio_feedback(embeds, torch.ones(7).long(), batch)
    expected = embeds.clone()
    expected[0] = 3
    if has_reserved_draft:
        expected[5] = 15
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


@pytest.mark.parametrize("stock_fallback", [False, True])
def test_registered_audio_sampler_preserves_rows_counts_and_owned_output(mocker, stock_fallback):
    state = make_state()
    metadata, payload = object(), {"_higgs_audio_snapshot": torch.ones(2, 4)}
    state.sampling_metadata = Mock(return_value=metadata)
    state.model.compute_logits = Mock(return_value=torch.ones(2, 8))
    sampled_ids = torch.tensor([[9], [8]])
    stock = SamplerOutput(sampled_ids, None, torch.tensor([0, 1]), torch.ones(2), torch.zeros(2))
    base_sampler = Mock(return_value=stock)

    def sample(logits, params, *, sample_text, audio_noise=None):
        return sample_text(logits) if stock_fallback else sampled_ids

    state.model.sample = Mock(side_effect=sample)
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
    sampler, rejection = state.custom_sampler(base_sampler)
    result = sampler.sample_step(hidden, batch, SimpleNamespace(prefill_len=SimpleNamespace(gpu=None)), None, standard)
    state.sampling_metadata.assert_called_once()
    torch.testing.assert_close(state.model.compute_logits.call_args.args[0], hidden[[2, 0]])
    assert state.model.compute_logits.call_args.args[1] is metadata
    assert result.num_sampled is count and result.num_rejected is rejected
    assert result.sampler_output.sampled_token_ids is sampled_ids
    assert result.sampler_output.num_sampled is count and result.sampler_output.num_rejected is rejected
    if stock_fallback:
        base_sampler.assert_called_once_with(state.model.compute_logits.return_value, batch)
        assert result.sampler_output.num_nans is stock.num_nans
    else:
        base_sampler.assert_not_called()
        assert result.sampler_output.num_nans is None
    state.model.compute_logits.assert_called_once()
    assert result.multimodal_outputs is payload and result.owns_multimodal_outputs
    assert not result.include_hidden_states and result.finalize_multimodal == state.finalize_audio_snapshot
    assert rejection is None
    standard.assert_not_called()


@pytest.mark.parametrize("candidate_ready", [False, True])
def test_prefill_joining_drafted_batch_samples_text_with_matching_rows(mocker, candidate_ready):
    """A new request can join a step that verifies another request's draft."""
    from dataclasses import replace

    from vllm.v1.worker.gpu.input_batch import InputBatch

    buffers = SimpleNamespace(
        device=torch.device("cpu"),
        seq_lens=torch.zeros(4, dtype=torch.int32),
        query_start_loc=torch.zeros(5, dtype=torch.int32),
        input_ids=torch.zeros(16, dtype=torch.int32),
        positions=torch.zeros(16, dtype=torch.int64),
        is_padding=torch.zeros(16, dtype=torch.bool),
    )
    # Request "a" verifies one draft (two logits); "new" is still prefilling.
    batch = replace(
        InputBatch.make_dummy(2, 10, buffers),
        req_ids=["a", "new"],
        expanded_idx_mapping=torch.tensor([0, 0, 1]),
        expanded_local_pos=torch.tensor([0, 1, 0], dtype=torch.int32),
        num_draft_tokens=1,
        num_draft_tokens_per_req=np.array([1, 0]),
        num_scheduled_tokens=np.array([2, 8]),
        logits_indices=torch.tensor([3, 4, 9]),
        cu_num_logits=torch.tensor([0, 2, 3], dtype=torch.int32),
        cu_num_logits_np=np.array([0, 2, 3], dtype=np.int32),
    )
    state = make_state()
    state.audio_speculator = speculator = Mock()
    speculator.batch = SimpleNamespace(batch_rows=torch.tensor([0])) if candidate_ready else None
    speculator.preview_noise.return_value = None, None, 0
    state.sampling_metadata = Mock(return_value=object())
    state.model.compute_logits = Mock(side_effect=lambda hidden, meta: hidden)
    state.model._step_audio_mode_rows = 1
    state.model.post_sample_multimodal_outputs = Mock(return_value={})
    state.model.sample = Mock(
        side_effect=lambda logits, meta, *, sample_text, draft_rows, audio_noise: sample_text(logits)
    )
    mocker.patch(
        "vllm_omni.model_executor.models.higgs_audio_v3.model_state.get_num_sampled_and_rejected",
        return_value=(torch.ones(2), torch.zeros(2)),
    )

    def base_sampler(logits, text_batch):
        assert text_batch.expanded_idx_mapping.shape[0] == logits.shape[0] == text_batch.num_reqs
        assert text_batch.logits_indices.tolist() == [3, 9]
        assert text_batch.cu_num_logits_np.tolist() == [0, 1, 2]
        return SamplerOutput(torch.tensor([[5], [6]]), None, None, torch.ones(2), torch.zeros(2))

    sampler, _ = state.custom_sampler(Mock(side_effect=base_sampler))
    hidden = torch.arange(16, dtype=torch.float32)[:, None]
    result = sampler.sample_step(hidden, batch, SimpleNamespace(prefill_len=SimpleNamespace(gpu=None)), None, Mock())
    assert result.sampler_output.sampled_token_ids.tolist() == [[5], [6]]
    assert state.model.sample.call_args.kwargs["draft_rows"].tolist() == [0]
    speculator.verify.assert_not_called()
    assert speculator.batch is None


@pytest.mark.parametrize("draft,grammar,message", [(1, None, "audio candidates"), (0, object(), "grammar sampling")])
def test_registered_audio_sampler_rejects_unsupported_modes(draft, grammar, message):
    sampler, _ = make_state().custom_sampler(Mock())
    batch = SimpleNamespace(req_ids=["real-request"], num_draft_tokens=draft)
    with pytest.raises(ValueError, match=message):
        sampler.sample_step(None, batch, None, grammar, Mock())


def test_preemption_replays_feedback_on_reused_slot_and_restores_rng_at_last_chunk():
    state = make_state()
    spec = SimpleNamespace(pending={}, device_rng={}, single_step=set())
    state.audio_speculator = spec
    state.audio_history = torch.zeros((4, 8, 2), dtype=torch.int16)
    state.audio_history_len = torch.zeros(4, dtype=torch.int32)
    state.add_request(0, request("replayed", None))
    state.audio_history[0, :2] = torch.tensor([[11, 12], [21, 22]])
    state.audio_history_len[0] = 2
    model = state.model
    model._last_batch_req_ids = ["replayed"]
    model._decode_has_codes = torch.tensor([True])
    model._decode_last_codes = torch.tensor([[21, 22]])
    model._decode_delay_count = torch.tensor([2])
    model._decode_eoc_countdown = torch.tensor([-1])
    model._decode_generation_done = torch.tensor([False])
    model.multimodal_embedding = lambda codes: codes.float()
    state.generators["replayed"] = Mock(get_offset=Mock(return_value=8))
    state.on_request_preempted("replayed", 0)
    state.remove_request(0)
    state.audio_history[0].fill_(-1)
    model._decode_last_codes.zero_()

    resumed = request("replayed", None)
    resumed.prefill_token_ids = [1, 2, 99, 99, 99]
    state.add_request(2, resumed)
    generator = state.generators["replayed"] = Mock()
    batch = SimpleNamespace(
        req_ids=["replayed"],
        idx_mapping_np=np.array([2]),
        num_computed_tokens_np=np.array([2]),
        num_scheduled_tokens=np.array([2]),
        query_start_loc_np=np.array([0, 2]),
    )
    state._restore_replayed_rows(batch)
    generator.set_offset.assert_not_called()
    assert spec.single_step == {"replayed"}
    first_chunk = torch.zeros(2, 2)
    state._replace_replay_feedback(first_chunk, batch)
    assert first_chunk.tolist() == [[0, 0], [11, 12]]
    batch.num_computed_tokens_np[0] = 4
    batch.num_scheduled_tokens[0] = 1
    state._restore_replayed_rows(batch)
    assert state.restored_rows.tolist() == [0]
    assert model._decode_last_codes.tolist() == [[21, 22]]
    generator.set_offset.assert_called_once_with(8)
    assert not spec.single_step
    last_chunk = torch.zeros(1, 2)
    state._replace_replay_feedback(last_chunk, batch)
    assert last_chunk.tolist() == [[21, 22]]
    assert not state.replay
