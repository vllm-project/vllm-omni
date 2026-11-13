# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MRv2 Talker K-step wiring: draft alias, spans, and multi-frame gating."""

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_tts import (
    MiniCPMO45OmniTTSForConditionalGeneration,
)
from vllm_omni.worker_v2.omni_ar_model_runner import OmniARModelRunner
from vllm_omni.worker_v2.omni_model_runner import OmniGPUModelRunner

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture(autouse=True)
def _cuda_runner_platform(monkeypatch):
    from vllm_omni.platforms import current_omni_platform

    monkeypatch.setattr(current_omni_platform, "is_cuda", lambda: True)


def _runner():
    runner = OmniARModelRunner.__new__(OmniARModelRunner)
    model = MiniCPMO45OmniTTSForConditionalGeneration.__new__(MiniCPMO45OmniTTSForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model._request_audio_states = {}
    model._request_codec_history = {}
    model._codec_eos_id = 0
    runner.model = model
    return runner


@pytest.mark.parametrize("frames", [0, 8])
def test_model_load_skips_rejection_sampler_only_for_talker_kstep(monkeypatch, frames):
    runner = _runner()
    runner.num_speculative_steps = 7
    runner.is_last_pp_rank = True
    runner.rejection_sampler = None
    monkeypatch.setattr(OmniGPUModelRunner, "load_model", lambda self: None)
    monkeypatch.setattr(runner, "_talker_kstep_frames", lambda: frames)
    if frames:
        runner.load_model()
    else:
        with pytest.raises(ValueError, match="requires a rejection sampler"):
            runner.load_model()


def _scheduled_batch(*counts):
    starts = [0]
    for count in counts:
        starts.append(starts[-1] + count)
    return SimpleNamespace(num_reqs=len(counts), num_scheduled_tokens=counts, query_start_loc_np=starts)


def test_kstep_spans_from_schedule():
    runner = _runner()
    batch = _scheduled_batch(8, 8)
    buffers = [{"audio_state": {"step": 1}}, {"audio_state": {"step": 1}}]
    extra = runner._build_kstep_model_kwargs_extra({"model_intermediate_buffer": buffers}, batch)
    assert extra["request_token_spans"] == [(0, 8), (8, 16)]
    assert extra["request_sample_eligible"] == [True, True]
    assert extra["request_sampling_params"] == [None, None]


@pytest.mark.parametrize(
    "handoff",
    [
        {"audio_codes": {"accumulated": torch.tensor([17, 23])}},
        {"codes": {"current": [17, 23]}},
        {"codes": {"audio": torch.tensor([23])}},
    ],
)
def test_initial_seed_uses_handoff(handoff):
    runner = _runner()
    state: dict[str, Any] = {}
    info = {"req_id": "seed-test", "audio_state": state, **handoff}
    batch = _scheduled_batch(8)
    extra = runner._build_kstep_model_kwargs_extra(
        {"model_intermediate_buffer": [info], "input_ids": torch.zeros(8, dtype=torch.long)}, batch
    )
    assert extra["request_sample_eligible"] == [True]
    assert info["audio_state"] is state
    assert info["request_id"] == "seed-test"
    assert state["last_code"] == 23
    assert state["_kstep_last_code"].item() == 23
    assert state["step"] == 1


def test_missing_seed_declines_kstep():
    runner = _runner()
    state: dict[str, Any] = {}
    info = {"req_id": "missing-seed", "audio_state": state}
    batch = _scheduled_batch(8)
    extra = runner._build_kstep_model_kwargs_extra(
        {"model_intermediate_buffer": [info], "input_ids": torch.zeros(8, dtype=torch.long)}, batch
    )
    assert extra["request_sample_eligible"] == [False]
    assert "last_code" not in state


def test_existing_state_is_not_reseeded():
    runner = _runner()
    state = {"step": 9, "last_code": 37}
    info = {"req_id": "existing-state", "audio_state": state, "codes": {"audio": [23]}}
    batch = _scheduled_batch(8)
    runner._build_kstep_model_kwargs_extra({"model_intermediate_buffer": [info]}, batch)
    assert info["audio_state"] is state
    assert state == {"step": 9, "last_code": 37}


def test_initial_embedding_uses_confirmed_history_only():
    runner = _runner()
    embedding = torch.nn.Embedding(64, 3)
    runner.model.emb_code = [embedding]
    runner.model._request_codec_history = {"embedding-test": [17, 23]}
    state = {"step": 2, "last_code": 11, "codes": torch.tensor([61, 62])}
    info = {"req_id": "embedding-test", "audio_state": state}
    embeds = torch.zeros(8, 3)
    ids = torch.zeros(8, dtype=torch.long)
    runner.model.rewrite_kstep_initial_embeds(embeds, {"model_intermediate_buffer": [info]}, [(0, 8)], ids)
    torch.testing.assert_close(embeds[0], embedding.weight[23])
    assert torch.count_nonzero(embeds[1:]) == 0
    assert ids.tolist() == [23, 0, 0, 0, 0, 0, 0, 0]
    assert state["last_code"] == 23


def test_initial_embedding_resolves_history_by_request_id():
    """Find the committed codec under the request ID used by the model."""
    runner = _runner()
    embedding = torch.nn.Embedding(64, 3)
    runner.model.emb_code = [embedding]
    runner.model._request_codec_history = {"req-abc-1": [17, 23]}
    state = {"step": 10, "last_code": 11}
    info = {"req_id": "sched-internal-9", "request_id": "req-abc-1", "audio_state": state}
    embeds = torch.zeros(8, 3)
    ids = torch.zeros(8, dtype=torch.long)
    runner.model.rewrite_kstep_initial_embeds(embeds, {"model_intermediate_buffer": [info]}, [(0, 8)], ids)
    torch.testing.assert_close(embeds[0], embedding.weight[23])
    assert torch.count_nonzero(embeds[1:]) == 0
    assert ids.tolist() == [23, 0, 0, 0, 0, 0, 0, 0]
    assert state["last_code"] == 23


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_initial_embedding_falls_back_to_row_token_on_first_kstep(dtype):
    runner = _runner()
    embedding = torch.nn.Embedding(64, 3)
    runner.model.emb_code = [embedding]
    state = {"step": 1}
    info = {"req_id": "first-kstep", "audio_state": state}
    embeds = torch.zeros(8, 3)
    ids = torch.tensor([23, 0, 0, 0, 0, 0, 0, 0], dtype=dtype)
    runner.model.rewrite_kstep_initial_embeds(embeds, {"model_intermediate_buffer": [info]}, [(0, 8)], ids)
    torch.testing.assert_close(embeds[0], embedding.weight[23])
    assert torch.count_nonzero(embeds[1:]) == 0
    assert ids.tolist() == [23, 0, 0, 0, 0, 0, 0, 0]
    assert state["last_code"] == 23
    assert state["_kstep_last_code"] == 23


def test_initial_embedding_declines_control_row_token_after_first_step():
    """Later K-steps must not treat the 2-wide control token as a codec.

    Once past the first K-step the row's first slot is the previous step's
    continue/stop sample (0/1), not a codec id; conditioning frame 1 on it
    would poison the state. The fallback is gated on ``step <= 1``.
    """
    runner = _runner()
    embedding = torch.nn.Embedding(64, 3)
    runner.model.emb_code = [embedding]
    state = {"step": 9}
    info = {"req_id": "later-kstep", "audio_state": state}
    embeds = torch.zeros(8, 3)
    ids = torch.tensor([1, 0, 0, 0, 0, 0, 0, 0], dtype=torch.long)
    runner.model.rewrite_kstep_initial_embeds(embeds, {"model_intermediate_buffer": [info]}, [(0, 8)], ids)
    assert torch.count_nonzero(embeds) == 0
    assert "last_code" not in state


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("width", [7, 8])
def test_propose_kstep_drafts_records_booked_width(dtype, width):
    runner = _runner()
    runner._talker_kstep_frames = lambda: 8
    runner.num_speculative_steps = width
    runner.model._request_audio_states = {"request-a": {"step": 1}}
    buffer = torch.full((3, width), -1, dtype=dtype)
    runner.req_states = SimpleNamespace(draft_tokens=buffer)
    captured = []
    runner.draft_tokens_handler = SimpleNamespace(set_draft_tokens=lambda batch, tokens: captured.append(tokens))
    batch = SimpleNamespace(num_reqs=1, req_ids=["request-a"], idx_mapping=torch.tensor([1]))
    runner._propose_draft_tokens(batch, SimpleNamespace(num_spec_tokens=7), torch.tensor([1]), torch.tensor([0]), None)

    assert len(captured) == 1
    assert captured[0].dtype == dtype
    assert captured[0].device == buffer.device
    assert captured[0].tolist() == [[0] * 7]
    assert buffer.tolist() == [[-1] * width, [0] * 7 + [-1] * (width - 7), [-1] * width]


def test_propose_drafts_unready_request_emits_empty_drafts():
    """An unseeded request keeps the next batch in single-frame mode."""
    runner = _runner()
    runner._talker_kstep_frames = lambda: 8
    runner.num_speculative_steps = 7
    runner.model._request_audio_states = {"request-a": {"step": 0}}
    runner.req_states = SimpleNamespace(draft_tokens=torch.full((3, 7), -1, dtype=torch.int32))
    captured = []
    runner.draft_tokens_handler = SimpleNamespace(set_draft_tokens=lambda batch, tokens: captured.append(tokens))

    def _fail(*args, **kwargs):
        raise AssertionError("ngram speculator must not propose for Talker K-step")

    runner.speculator = SimpleNamespace(propose=_fail)
    runner.sampler = SimpleNamespace()
    batch = SimpleNamespace(
        num_reqs=1,
        req_ids=["request-a"],
        idx_mapping=torch.tensor([1]),
        input_ids=torch.zeros(1, dtype=torch.int32),
    )
    runner._propose_draft_tokens(batch, SimpleNamespace(num_spec_tokens=7), torch.tensor([1]), torch.tensor([0]), None)
    assert len(captured) == 1
    assert captured[0].shape == (1, 0)
    # The draft buffer itself must stay untouched (no stale row rewrite).
    assert runner.req_states.draft_tokens.tolist() == [[-1] * 7] * 3


def test_propose_drafts_nonkstep_model_keeps_generic_buffer_slice():
    """With no speculator, the generic handler records the existing drafts."""
    runner = _runner()
    runner._talker_kstep_frames = lambda: 0
    runner.num_speculative_steps = 7
    runner.model._request_audio_states = {"request-a": {"step": 1}}
    runner.req_states = SimpleNamespace(draft_tokens=torch.tensor([[5, 5, 5, 5, 5, 5, 5]] * 3, dtype=torch.int32))
    captured = []
    runner.draft_tokens_handler = SimpleNamespace(set_draft_tokens=lambda batch, tokens: captured.append(tokens))
    runner.speculator = None
    runner.pp_handler = None
    batch = SimpleNamespace(
        num_reqs=1,
        req_ids=["request-a"],
        idx_mapping=torch.tensor([1]),
        input_ids=torch.zeros(1, dtype=torch.int32),
    )
    runner._propose_draft_tokens(batch, SimpleNamespace(num_spec_tokens=7), torch.tensor([1]), torch.tensor([0]), None)
    # Not the (1, 0) empty-draft shape: the generic tail recorded a slice.
    assert len(captured) == 1
    assert captured[0].shape == (1, 7)
    assert captured[0].tolist() == [[5] * 7]


def test_kstep_control_sampler_packs_and_truncates(monkeypatch):
    from vllm_omni.model_executor.models.minicpmo_4_5 import minicpmo_4_5_omni_tts as tts

    def counts(sampled, seq_lens, cu, mapping, prefill):
        unfinished = seq_lens < prefill[mapping]
        sampled = sampled.masked_fill(unfinished, 0)
        return sampled, (cu[1:] - cu[:-1] - sampled).masked_fill(unfinished, 0)

    monkeypatch.setattr(tts, "get_num_sampled_and_rejected", counts)
    sampler = object.__new__(tts.MiniCPMO45TalkerSampler)
    sampler.talker = SimpleNamespace(_k_step_frames=8)
    sampler.req_states = SimpleNamespace(prefill_len=SimpleNamespace(gpu=torch.tensor([1, 1, 10])))
    batch = SimpleNamespace(
        cu_num_logits=torch.tensor([0, 8, 16, 17]),
        seq_lens=torch.tensor([8, 8, 5]),
        idx_mapping=torch.tensor([0, 1, 2]),
    )
    tokens = torch.tensor([0] * 8 + [0, 0, 1, 0, 1, 0, 0, 0] + [0])
    logits = torch.nn.functional.one_hot(tokens, 2).float()
    output = sampler._sample_kstep_controls(logits, batch)
    assert output.num_sampled.tolist() == [8, 3, 0]
    assert output.num_rejected.tolist() == [0, 5, 0]
    assert output.sampled_token_ids.tolist() == [[0] * 8, [0, 0, 1] + [-1] * 5, [-1] * 8]
