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
    EOC_ID,
    HiggsAudioV3TalkerForConditionalGeneration,
    _scan_draft_rows,
    _verified_audio_prefix,
)
from vllm_omni.model_executor.models.higgs_audio_v3.model_state import HiggsModelState
from vllm_omni.model_executor.models.higgs_audio_v3.spec_config import (
    HIGGS_AUDIO_SPEC_METHOD,
    HiggsAudioSpeculatorConfig,
)
from vllm_omni.model_executor.models.higgs_audio_v3.spec_decode import HiggsAudioSpeculator
from vllm_omni.worker_v2.model_states.intermediate_buffer import OmniIntermediateBuffer
from vllm_omni.worker_v2.model_states.mtp_model_state import MTPState

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_omni_plugin_registers_the_weightless_higgs_speculator():
    from vllm.utils.import_utils import resolve_obj_by_qualname
    from vllm.v1.worker.gpu.spec_decode.speculator_config import get_custom_speculator_config_class

    from vllm_omni.engine.arg_utils import register_omni_models_to_vllm

    register_omni_models_to_vllm()
    config = get_custom_speculator_config_class(HIGGS_AUDIO_SPEC_METHOD)
    assert config is HiggsAudioSpeculatorConfig
    assert config.base_method is None
    assert resolve_obj_by_qualname(config.speculator_class) is HiggsAudioSpeculator


def test_speculation_requires_batch_invariant_math(monkeypatch):
    monkeypatch.setattr(
        "vllm_omni.model_executor.models.higgs_audio_v3.spec_decode.current_platform.is_cuda", lambda: True
    )
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "0")
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(tensor_parallel_size=1, data_parallel_size=1),
        scheduler_config=SimpleNamespace(async_scheduling=False),
        cache_config=SimpleNamespace(enable_prefix_caching=False),
        model_config=SimpleNamespace(max_model_len=128),
        num_speculative_tokens=1,
    )
    with pytest.raises(ValueError, match="VLLM_BATCH_INVARIANT=1"):
        HiggsAudioSpeculator(config, torch.device("cpu"))
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    config.num_speculative_tokens = 2
    with pytest.raises(ValueError, match="num_speculative_tokens=1"):
        HiggsAudioSpeculator(config, torch.device("cpu"))


def test_speculation_rejects_resumed_audio_prefix():
    s = make_state()
    s.audio_speculator = Mock()
    resumed = request("r", 1)
    resumed.prefill_token_ids = [1, 2, 99, 1024]
    with pytest.raises(ValueError, match="preempted audio prefix"):
        s.add_request(0, resumed)


def test_verified_prefix_ends_at_correction_bonus_or_eoc():
    drafts = torch.tensor([11, 22, 33]).view(1, 3, 1).expand(4, 3, 8).clone()
    drafts[3, 1, 0] = EOC_ID
    target = torch.cat((drafts.clone(), torch.full((4, 1, 8), 44)), dim=1)
    target[0, 0, 7] = 90
    target[1, 1, 7] = 91
    codebooks = torch.arange(8)[None]
    delay = torch.full((4,), 7)
    remaining = torch.full((4,), -1)
    effective, delays, countdowns = _scan_draft_rows(drafts, delay, remaining, codebooks, 8)
    torch.testing.assert_close(effective[:, :2], drafts[:, :2])
    torch.testing.assert_close(countdowns[3], torch.tensor([-1, -1, 7, 6]))

    codes, _, next_remaining, _, last = _verified_audio_prefix(
        target, drafts, torch.full((4,), 3), delays, countdowns, codebooks, 8
    )
    assert last.tolist() == [0, 1, 3, 1]
    batch = torch.arange(4)
    assert codes[batch, last, 7].tolist() == [90, 91, 44, 22]
    assert next_remaining[batch, last].tolist() == [-1, -1, -1, 7]
    assert remaining.tolist() == [-1] * 4


def test_draft_feedback_tracks_request_slot_and_position_after_reorder():
    s = make_state()
    s.add_request(0, request("a", 42))
    s.add_request(2, request("b", 43))
    spec = object.__new__(HiggsAudioSpeculator)
    s.audio_speculator = spec
    a, b = torch.full((1, 8), 10), torch.full((2, 8), 20)
    b[1] = 21
    spec.pending = {"a": (0, 18, a), "b": (2, 15, b)}
    s.model.num_codebooks = 8
    s.model.multimodal_embedding = lambda codes: codes.float()
    s.model._get_codebook_indices = lambda device: torch.arange(8, device=device)[None]
    s.model._decode_delay_count = torch.full((3,), 7)
    s.model._decode_eoc_countdown = torch.full((3,), -1)
    s.model._decode_last_codes = torch.tensor([4, 5, 6])[:, None].expand(-1, 8)
    batch = SimpleNamespace(
        req_ids=["b", "a", "prefill"],
        num_draft_tokens=3,
        num_draft_tokens_per_req=np.array([2, 1, 0]),
        idx_mapping_np=np.array([2, 0, 1]),
        num_computed_tokens_np=np.array([15, 18, 0]),
        query_start_loc_np=np.array([0, 3, 5, 8]),
        cu_num_logits_np=np.array([0, 3, 5, 6]),
    )
    embeddings = torch.zeros((8, 8))
    spec.prepare(s, batch, embeddings)
    assert embeddings[:, 0].tolist() == [4, 20, 21, 5, 10, 0, 0, 0]
    assert spec.batch.logit_rows.tolist() == [[0, 1, 2], [3, 4, 4]]
    assert not spec.pending

    spec.pending["a"] = (0, 23, a)
    s.remove_request("a")
    s.add_request(0, request("new", 7))
    batch.req_ids[1] = "new"
    spec.pending["b"] = (2, 15, b)
    with pytest.raises(RuntimeError, match="slot/position for new"):
        spec.prepare(s, batch, embeddings)


def test_proposer_reanchors_to_last_commit_and_reuses_leftover_rows():
    s = make_state()
    for i in range(3):
        s.add_request(i, request(str(i), i, max_tokens=100))
    spec = object.__new__(HiggsAudioSpeculator)
    spec.max_model_len, spec.pending = 128, {}
    spec.guesses = {"1": torch.full((2, 8), 5)}
    s.model.num_codebooks = 8
    s.model._decode_has_codes = torch.ones(3, dtype=torch.bool)
    s.model._decode_delay_count = torch.full((3,), 7)
    s.model._decode_eoc_countdown = torch.full((3,), -1)
    s.model._decode_last_codes = torch.arange(24).view(3, 8)
    s.model._step_audio_mode_rows = 3
    batch = SimpleNamespace(
        req_ids=["0", "1", "2"],
        num_reqs=3,
        idx_mapping_np=np.arange(3),
        cu_num_logits_np=np.array([0, 4, 8, 12]),
        num_computed_tokens_np=np.full(3, 20),
        num_scheduled_tokens=np.full(3, 4),
        logits_indices=torch.arange(4, 16),
    )
    hidden = torch.arange(16)[:, None].expand(-1, 8).float()
    spec.propose_audio(s, batch, hidden, torch.tensor([1, 2, 4]), 1)
    assert [record[1] for record in spec.pending.values()] == [21, 22, 24]
    # A leftover verified row is the candidate; otherwise the committed row reserves the slot.
    assert [record[2][:, 0].tolist() for record in spec.pending.values()] == [[0], [5], [16]]
    assert spec.guesses == {}


def test_verifier_commits_only_the_verified_rows_and_rng_prefix(monkeypatch):
    from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch

    monkeypatch.setattr(
        "vllm_omni.model_executor.models.higgs_audio_v3.higgs_audio_v3_talker.apply_top_k_top_p",
        apply_top_k_top_p_pytorch,
    )
    s = make_state()
    for i in range(4):
        s.add_request(i, request(str(i), i, temperature=0))
    spec = object.__new__(HiggsAudioSpeculator)
    spec.guesses = {}
    spec.pending = {str(i): (i, 20, torch.tensor([11, 22])[:, None].expand(-1, 8)) for i in range(3)}
    s.audio_speculator = spec
    model = s.model
    model.num_codebooks = 8
    model.multimodal_embedding = Mock(side_effect=lambda codes: codes.float())
    model.multimodal_embedding.vocab_size = 1026
    model._get_codebook_indices = lambda device: torch.arange(8, device=device)[None]
    model._decode_delay_count = torch.full((4,), 7)
    model._decode_eoc_countdown = torch.full((4,), -1)
    model._decode_last_codes = torch.full((4, 8), 10)
    model._decode_generation_done = torch.zeros(4, dtype=torch.bool)
    model._decode_has_codes = torch.ones(4, dtype=torch.bool)
    model.modality_head = SimpleNamespace(
        generate=lambda hidden: torch.nn.functional.one_hot(hidden.long(), 1026).float() * 20
    )
    model._sample_audio_codes = HiggsAudioV3TalkerForConditionalGeneration._sample_audio_codes.__get__(model)
    model._expand_audio_sampling_params = HiggsAudioV3TalkerForConditionalGeneration._expand_audio_sampling_params
    batch = SimpleNamespace(
        req_ids=[str(i) for i in range(4)],
        num_reqs=4,
        num_draft_tokens=6,
        num_draft_tokens_per_req=np.array([2, 2, 2, 0]),
        idx_mapping_np=np.arange(4),
        num_computed_tokens_np=np.full(4, 20),
        query_start_loc_np=np.array([0, 3, 6, 9, 10]),
        cu_num_logits_np=np.array([0, 3, 6, 9, 10]),
    )
    spec.prepare(s, batch, torch.zeros(10, 8))
    target = torch.tensor([90, 22, 33, 11, 91, 33, 11, 22, 33, 11])[:, None].expand(-1, 8).float()
    snapshot = torch.zeros(4, 10, dtype=torch.int32)
    snapshot[:, :8] = target[torch.tensor([0, 3, 6, 9])].int()
    snapshot[:, 8] = 1
    payload = {"_higgs_audio_snapshot": snapshot}
    counts = torch.ones(4, dtype=torch.int32)
    s.generators = {str(i): Mock(get_offset=Mock(return_value=100)) for i in range(4)}
    s.metadata_key = None

    def preview(noise, generators):
        noise.fill_(1)
        return 4

    monkeypatch.setattr("vllm_omni.model_executor.models.higgs_audio_v3.spec_decode.fill_exponential_steps", preview)
    tokens = spec.verify(s, target, batch, torch.full((4, 1), 99), counts, payload)
    assert counts.tolist() == [1, 2, 3, 1]
    assert tokens.shape == (4, 3)
    assert model._decode_last_codes[:, 0].tolist() == [90, 91, 33, 10]
    # Rows sampled after the first mismatch become the next candidates.
    assert {k: v[:, 0].tolist() for k, v in spec.guesses.items()} == {"0": [22, 33], "1": [33]}
    for i, offset in enumerate([100, 104, 108]):
        s.generators[str(i)].set_offset.assert_called_once_with(offset)
    s.generators["3"].set_offset.assert_not_called()
    output = s.finalize_audio_snapshot(payload, counts.tolist())
    assert [row["codes.audio"][:, 0].tolist() for row in output.inter_stage] == [[90], [11, 91], [11, 22, 33], [11]]


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
    batch = SimpleNamespace(req_ids=[req_id], num_tokens=3, num_reqs=1, num_tokens_after_padding=3, has_prefill=False)
    s.model.model = SimpleNamespace(embed_tokens=torch.nn.Embedding(100, 3))
    s._static_inputs_embeds = torch.zeros(3, 3)
    s._apply_audio_feedback = lambda embeds, ids, batch: embeds
    s.audio_speculator = Mock()
    s._prepare_static_embeddings(batch, {"input_ids": torch.tensor([1, 2, 3])})
    s.audio_speculator.prepare.assert_not_called()
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


@pytest.mark.parametrize("stock_fallback", [False, True])
def test_registered_audio_sampler_preserves_rows_counts_and_owned_output(mocker, stock_fallback):
    state = make_state()
    metadata, payload = object(), {"_higgs_audio_snapshot": torch.ones(2, 4)}
    state.sampling_metadata = Mock(return_value=metadata)
    state.model.compute_logits = Mock(return_value=torch.ones(2, 8))
    sampled_ids = torch.tensor([[9], [8]])
    stock = SamplerOutput(sampled_ids, None, torch.tensor([0, 1]), torch.ones(2), torch.zeros(2))
    base_sampler = Mock(return_value=stock)

    def sample(logits, params, *, sample_text):
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


@pytest.mark.parametrize("draft,grammar,message", [(1, None, "audio candidates"), (0, object(), "grammar sampling")])
def test_registered_audio_sampler_rejects_unsupported_modes(draft, grammar, message):
    sampler, _ = make_state().custom_sampler(Mock())
    batch = SimpleNamespace(req_ids=["real-request"], num_draft_tokens=draft)
    with pytest.raises(ValueError, match=message):
        sampler.sample_step(None, batch, None, grammar, Mock())
