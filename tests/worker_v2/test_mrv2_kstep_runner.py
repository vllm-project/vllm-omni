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


def test_cuda_kstep_requires_model_hooks_before_execution():
    runner = _runner()
    runner.model_config = SimpleNamespace(model_stage="tts")
    runner.vllm_config = SimpleNamespace(speculative_config={"method": "ngram", "num_speculative_tokens": 7})
    runner.model = SimpleNamespace(
        supports_multi_frame_decode=True,
        batch_stop_logits=None,
        take_batch_stop_logits=lambda: None,
        set_batch_stop_logits=lambda logits: None,
        merge_frame_outputs=lambda outputs, stop_logits: outputs,
    )
    with pytest.raises(TypeError, match="kstep_rows_ready"):
        runner._talker_kstep_frames()


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
