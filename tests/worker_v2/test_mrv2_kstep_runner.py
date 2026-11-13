# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MRv2 Talker K-step wiring: draft alias, spans, and multi-frame gating."""

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vllm_omni.platforms.npu.worker import talker_multiframe
from vllm_omni.worker_v2.omni_ar_model_runner import OmniARModelRunner
from vllm_omni.worker_v2.omni_model_runner import OmniGPUModelRunner

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture(autouse=True)
def _cuda_runner_platform(monkeypatch):
    from vllm_omni.platforms import current_omni_platform

    monkeypatch.setattr(current_omni_platform, "is_cuda", lambda: True)


def _runner():
    return OmniARModelRunner.__new__(OmniARModelRunner)


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


def test_kstep_spans_from_schedule():
    runner = _runner()
    batch = SimpleNamespace(num_reqs=2, num_scheduled_tokens=[8, 8])
    buffers = [{"audio_state": {"step": 1}}, {"audio_state": {"step": 1}}]
    extra = runner._build_kstep_model_kwargs_extra({"model_intermediate_buffer": buffers}, batch)
    assert extra["request_token_spans"] == [(0, 8), (8, 16)]
    assert extra["request_sample_eligible"] == [True, True]
    assert extra["request_sampling_params"] == [None, None]


def test_kstep_spans_decline_on_mixed_batch():
    runner = _runner()
    # One request prefilling while another decodes: spans differ, so the loop
    # must decline and the single forward runs.
    batch = SimpleNamespace(num_reqs=2, num_scheduled_tokens=[3, 8])
    extra = runner._build_kstep_model_kwargs_extra({"model_intermediate_buffer": [{}, {}]}, batch)
    assert extra["request_token_spans"] == [(0, 3), (3, 11)]


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
    batch = SimpleNamespace(num_reqs=1, num_scheduled_tokens=[8])
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
    batch = SimpleNamespace(num_reqs=1, num_scheduled_tokens=[8])
    extra = runner._build_kstep_model_kwargs_extra(
        {"model_intermediate_buffer": [info], "input_ids": torch.zeros(8, dtype=torch.long)}, batch
    )
    assert extra["request_sample_eligible"] == [False]
    assert state["_kstep_seed_missing"]
    assert "last_code" not in state


def test_existing_state_is_not_reseeded():
    runner = _runner()
    state = {"step": 9, "last_code": 37}
    info = {"req_id": "existing-state", "audio_state": state, "codes": {"audio": [23]}}
    batch = SimpleNamespace(num_reqs=1, num_scheduled_tokens=[8])
    runner._build_kstep_model_kwargs_extra({"model_intermediate_buffer": [info]}, batch)
    assert info["audio_state"] is state
    assert state == {"step": 9, "last_code": 37}


def test_initial_embedding_uses_confirmed_history_only():
    runner = _runner()
    embedding = torch.nn.Embedding(64, 3)
    runner.model = SimpleNamespace(
        talker=SimpleNamespace(emb_code=[embedding]),
        _request_codec_history={"embedding-test": [17, 23]},
    )
    state = {"step": 2, "last_code": 11, "codes": torch.tensor([61, 62])}
    info = {"req_id": "embedding-test", "audio_state": state}
    embeds = torch.zeros(8, 3)
    ids = torch.zeros(8, dtype=torch.long)
    runner._rewrite_kstep_initial_embeds(embeds, {"model_intermediate_buffer": [info]}, [(0, 8)], ids)
    torch.testing.assert_close(embeds[0], embedding.weight[23])
    assert torch.count_nonzero(embeds[1:]) == 0
    assert ids.tolist() == [23, 0, 0, 0, 0, 0, 0, 0]
    assert state["last_code"] == 23


def test_initial_embedding_resolves_history_by_request_id():
    """The history is keyed by the user-facing request_id, not req_id.

    The model registers ``_request_codec_history`` under ``request_id`` (the
    prefill's frame-1 seed and the K-step host replay both key on it), while
    the buffer entries carry the scheduler-internal ``req_id``. Resolving
    ``req_id`` first missed on every step, so every decode step after the
    first fell back to the stale ``last_code`` mirror -- the frozen frame-1
    seed -- and re-conditioned the whole K-row on the same first-frame
    embedding: the context never advanced, eos_p froze at ~0.012 at the
    sentence end (V1: 0.36-0.74), and the stream rambled to the frame
    ceiling (observed 2026-11-11: 11.6-11.9 s mean audio, repeated phrases).
    """
    runner = _runner()
    embedding = torch.nn.Embedding(64, 3)
    # Mirror the real nesting: the registry lives on the Talker submodule,
    # NOT on the wrapper -- looking it up on ``runner.model`` returned {}
    # and kept the frozen-seed miss alive even after the key order was
    # fixed (fix-4 bench: still 11.92 s mean audio with ``src=lc``).
    runner.model = SimpleNamespace(
        talker=SimpleNamespace(
            emb_code=[embedding],
            _request_codec_history={"req-abc-1": [17, 23]},
        ),
    )
    state = {"step": 10, "last_code": 11}
    info = {"req_id": "sched-internal-9", "request_id": "req-abc-1", "audio_state": state}
    embeds = torch.zeros(8, 3)
    ids = torch.zeros(8, dtype=torch.long)
    runner._rewrite_kstep_initial_embeds(embeds, {"model_intermediate_buffer": [info]}, [(0, 8)], ids)
    torch.testing.assert_close(embeds[0], embedding.weight[23])
    assert torch.count_nonzero(embeds[1:]) == 0
    assert ids.tolist() == [23, 0, 0, 0, 0, 0, 0, 0]
    assert state["last_code"] == 23


def test_initial_embedding_stamps_int32_input_ids():
    """The runner's input_ids buffer is int32; the codec stamp must cast.

    Writing the int64 codec tensor into the int32 buffer raised on every
    decode step and flooded the log with "kstep input_ids stamp failed"
    tracebacks (observed 2026-11-11: hundreds per run).
    """
    runner = _runner()
    embedding = torch.nn.Embedding(64, 3)
    runner.model = SimpleNamespace(talker=SimpleNamespace(emb_code=[embedding]))
    state = {"step": 1}
    info = {"req_id": "int32-stamp", "audio_state": state}
    embeds = torch.zeros(8, 3)
    ids = torch.zeros(8, dtype=torch.int32)
    ids[0] = 23
    runner._rewrite_kstep_initial_embeds(embeds, {"model_intermediate_buffer": [info]}, [(0, 8)], ids)
    torch.testing.assert_close(embeds[0], embedding.weight[23])
    assert ids.tolist() == [23, 0, 0, 0, 0, 0, 0, 0]


def test_initial_embedding_falls_back_to_row_token_on_first_kstep():
    """A first K-step row's first slot is the prefill step's own codec sample.

    The mrv2 chain bookkeeps ``last_code`` from the *input* ids, which at the
    prefill step are placeholder zeros -- so the state reaches the first
    K-step without a usable codec. V1 parity: the row's committed token (the
    standard sampler's prefill sample) is that frame's condition.
    """
    runner = _runner()
    embedding = torch.nn.Embedding(64, 3)
    runner.model = SimpleNamespace(talker=SimpleNamespace(emb_code=[embedding]))
    state = {"step": 1}
    info = {"req_id": "first-kstep", "audio_state": state}
    embeds = torch.zeros(8, 3)
    ids = torch.tensor([23, 0, 0, 0, 0, 0, 0, 0], dtype=torch.long)
    runner._rewrite_kstep_initial_embeds(embeds, {"model_intermediate_buffer": [info]}, [(0, 8)], ids)
    torch.testing.assert_close(embeds[0], embedding.weight[23])
    assert torch.count_nonzero(embeds[1:]) == 0
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
    runner.model = SimpleNamespace(talker=SimpleNamespace(emb_code=[embedding]))
    state = {"step": 9}
    info = {"req_id": "later-kstep", "audio_state": state}
    embeds = torch.zeros(8, 3)
    ids = torch.tensor([1, 0, 0, 0, 0, 0, 0, 0], dtype=torch.long)
    runner._rewrite_kstep_initial_embeds(embeds, {"model_intermediate_buffer": [info]}, [(0, 8)], ids)
    assert torch.count_nonzero(embeds) == 0
    assert "last_code" not in state


def test_applies_engages_on_uniform_kstep_batch(monkeypatch):
    monkeypatch.setattr(talker_multiframe, "supports_multi_frame_decode", lambda model: True)
    model = SimpleNamespace()
    extra = {
        "request_token_spans": [(0, 8), (8, 16)],
        # A live decode row carries its Talker state, already stepped at least
        # once so frame 0 has a previous codec id to re-embed.
        "model_intermediate_buffer": [
            {"audio_state": {"step": 1}},
            {"audio_state": {"step": 1}},
        ],
    }
    assert talker_multiframe.applies(model, extra) == 8


def test_applies_declines_on_non_uniform_spans(monkeypatch):
    monkeypatch.setattr(talker_multiframe, "supports_multi_frame_decode", lambda model: True)
    model = SimpleNamespace()
    extra = {
        "request_token_spans": [(0, 3), (3, 11)],
        "model_intermediate_buffer": [{}, {}],
    }
    assert talker_multiframe.applies(model, extra) == 0


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_propose_kstep_drafts_preserves_buffer_dtype(monkeypatch, dtype):
    runner = _runner()
    runner._talker_kstep_frames = lambda: 8
    runner.num_speculative_steps = 7
    runner.model = torch.nn.Module()
    runner.model._request_audio_states = {"request-a": {"step": 1}}
    runner.req_states = SimpleNamespace(draft_tokens=torch.full((3, 7), -1, dtype=dtype))
    captured = []
    runner.draft_tokens_handler = SimpleNamespace(set_draft_tokens=lambda batch, tokens: captured.append(tokens))
    batch = SimpleNamespace(
        num_reqs=1,
        req_ids=["request-a"],
        idx_mapping=torch.tensor([1]),
        input_ids=torch.zeros(1, dtype=torch.int32),
    )
    runner._propose_draft_tokens(batch, SimpleNamespace(num_spec_tokens=7), torch.tensor([1]), torch.tensor([0]), None)
    assert len(captured) == 1
    assert captured[0].shape == (1, 7)
    assert captured[0].dtype == dtype
    assert captured[0].device == runner.req_states.draft_tokens.device
    assert captured[0].tolist() == [[0] * 7]
    assert runner.req_states.draft_tokens.tolist() == [[-1] * 7, [0] * 7, [-1] * 7]


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_propose_kstep_drafts_slices_handler_to_booked_width(monkeypatch, dtype):
    """The handler must record the booked draft width (K - 1), not the span K.

    Native parity (vllm/v1/worker/gpu/model_runner.py:2333) slices the buffer
    to ``num_spec_tokens`` before the handler records it; a K-wide record
    books K + 1 query rows per request on the next step (9 for K = 8) and
    overruns the frame loop's span. The unbooked buffer tail carries -1.
    """
    runner = _runner()
    runner._talker_kstep_frames = lambda: 8
    runner.num_speculative_steps = 8
    runner.model = torch.nn.Module()
    runner.model._request_audio_states = {"request-a": {"step": 1}}
    runner.req_states = SimpleNamespace(draft_tokens=torch.full((3, 8), -1, dtype=dtype))
    captured = []
    runner.draft_tokens_handler = SimpleNamespace(set_draft_tokens=lambda batch, tokens: captured.append(tokens))
    batch = SimpleNamespace(
        num_reqs=1,
        req_ids=["request-a"],
        idx_mapping=torch.tensor([1]),
        input_ids=torch.zeros(1, dtype=torch.int32),
    )
    runner._propose_draft_tokens(batch, SimpleNamespace(num_spec_tokens=7), torch.tensor([1]), torch.tensor([0]), None)
    assert len(captured) == 1
    assert captured[0].shape == (1, 7)
    assert captured[0].dtype == dtype
    assert captured[0].tolist() == [[0] * 7]
    # The buffer row keeps the full span K with the unbooked tail padded -1.
    assert runner.req_states.draft_tokens.tolist() == [
        [-1] * 8,
        [0] * 7 + [-1],
        [-1] * 8,
    ]


def test_propose_drafts_unready_request_emits_empty_drafts():
    """An unseeded request keeps the next batch in single-frame mode."""
    runner = _runner()
    runner._talker_kstep_frames = lambda: 8
    runner.num_speculative_steps = 7
    runner.model = torch.nn.Module()
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


def test_propose_drafts_nonkstep_model_keeps_speculator_path(monkeypatch):
    """A model without multi-frame support keeps the generic speculator path.

    Patch B gates on ``drafts_this_step > 0``; for a deployment whose model
    does not support multi-frame decode (plain ngram/DSpark deployments),
    the branch must stay closed so the real speculator keeps proposing.
    With ``speculator=None`` the trailing handler records the existing
    buffer slice -- exactly the pre-patch behavior.
    """
    runner = _runner()
    runner._talker_kstep_frames = lambda: 0
    runner.num_speculative_steps = 7
    runner.model = torch.nn.Module()
    runner.model._request_audio_states = {"request-a": {"step": 1}}
    runner._kstep_saw_concurrent = True
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
