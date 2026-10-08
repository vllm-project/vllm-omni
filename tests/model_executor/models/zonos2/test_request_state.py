# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""P3 integration contracts: replay, stable IDs, same-step payload and cleanup."""

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

from vllm_omni.model_executor.models.zonos2.configuration_zonos2 import Zonos2Config
from vllm_omni.model_executor.models.zonos2.zonos2_sampler import Zonos2RequestState, Zonos2SamplingParams
from vllm_omni.model_executor.models.zonos2.zonos2_talker import (
    Zonos2MultiEmbedder,
    Zonos2TalkerForConditionalGeneration,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def model():
    m = Zonos2TalkerForConditionalGeneration.__new__(Zonos2TalkerForConditionalGeneration)
    nn.Module.__init__(m)
    m.config = Zonos2Config(dim=2)
    m.multi_embedder = Zonos2MultiEmbedder(m.config)
    m.speaker_lda_projection = nn.Linear(2048, 2)
    m.speaker_projection = nn.Linear(2, 2)
    m._request_states = {}
    m._sampling_plan = []
    m._step_codes = {}
    m._step_payload = None
    with torch.no_grad():
        for col, table in enumerate(m.multi_embedder.embedders):
            table.weight[:, 0] = torch.arange(len(table.weight)) * (col + 1)
            table.weight[:, 1] = col
    return m


def info(key, *, prompt_len=3, offset=0, history=None, count=0, max_tokens=20, seed=42):
    frames = torch.full((prompt_len, 10), 1025, dtype=torch.int32)
    frames[:, 9] = 2
    return {
        "_omni_req_id": key,
        "request_id": key,
        "zonos2_frames": frames,
        "_omni_num_computed_tokens": offset,
        "_omni_is_prefill": offset < prompt_len,
        "_omni_output_token_ids": [0] * count,
        "_omni_sampling_params": {
            "temperature": 0.0,
            "top_k": -1,
            "top_p": 1.0,
            "min_p": 0.0,
            "seed": seed,
            "max_tokens": max_tokens,
        },
        "zonos2": {} if history is None else {"history": history, "seed": seed},
    }


def logits_for_codes(values):
    x = torch.full((len(values), 9, 1026), -20.0)
    for i, value in enumerate(values):
        x[i, :, value] = 20.0
    return x


def test_reprefill_crosses_prompt_and_completion_without_resetting_history(model):
    history = torch.tensor([[5] * 9, [7] * 9, [9] * 9], dtype=torch.int32)
    data = info("a", offset=1, history=history, count=3)
    _, embeds, update = model.preprocess(torch.zeros(5, dtype=torch.int64), None, **data)
    stream = torch.cat((data["zonos2_frames"], torch.cat((history, torch.full((3, 1), 519, dtype=torch.int32)), dim=1)))
    torch.testing.assert_close(embeds, model.multi_embedder(stream[1:6]), rtol=0, atol=0)
    assert len(model._request_states["a"].history) == 3
    assert update["_omni_req_id"] == "a" and update["_zonos2_scheduled_span"] == 5


def test_missing_nine_codebook_history_fails_instead_of_restarting(model):
    with pytest.raises(ValueError, match="lifecycle"):
        model.preprocess(torch.zeros(3, dtype=torch.int64), None, **info("a", count=4))


def test_retraction_truncates_stale_extra_frame_before_next_sample(model):
    old = torch.zeros((5, 9), dtype=torch.int32)
    options = Zonos2SamplingParams(temperature=0, max_tokens=20)
    model._request_states["a"] = Zonos2RequestState.rebuild("a", options, old)
    kept = torch.zeros((4, 9), dtype=torch.int32)
    model.preprocess(torch.zeros(3, dtype=torch.int64), None, **info("a", history=kept, count=4))
    assert len(model._request_states["a"].history) == 4


def test_batch_reordering_routes_current_codes_by_request_id(model):
    a = info("a")
    b = info("b")
    model.preprocess(torch.zeros(3, dtype=torch.int64), None, **a)
    model.preprocess(torch.zeros(3, dtype=torch.int64), None, **b)
    for data in (a, b):
        data.update(_zonos2_scheduled_span=3)
    model._sampling_plan = [("b", True, b), ("a", True, a)]
    model._last_fused_logits = logits_for_codes([8, 7])
    output = model.make_omni_output(torch.zeros(6, 2))
    sampled = model.sample(torch.zeros(2, 1026), None)
    assert sampled.sampled_token_ids.tolist() == [[0], [0]]
    assert output.multimodal_outputs["codes"]["audio"][0].tolist() == [[8] * 9]
    assert model.postprocess(torch.empty(0), _omni_req_id="a")["codes"]["audio"].tolist() == [[7] * 9]
    assert model.postprocess(torch.empty(0), _omni_req_id="b")["codes"]["audio"].tolist() == [[8] * 9]


def test_terminal_frame_is_present_in_same_output_as_stop_token(model):
    data = info("a", max_tokens=1)
    data["_zonos2_scheduled_span"] = 3
    model.preprocess(torch.zeros(3, dtype=torch.int64), None, **data)
    model._sampling_plan = [("a", True, data)]
    model._last_fused_logits = logits_for_codes([7])
    output = model.make_omni_output(torch.zeros(3, 2))
    assert output.multimodal_outputs["codes"]["audio"][0].numel() == 0
    sampled = model.sample(torch.zeros(1, 1026), None)
    assert sampled.sampled_token_ids.tolist() == [[1]]
    assert output.multimodal_outputs["codes"]["audio"][0].tolist() == [[7] * 9]
    assert len(model.postprocess(torch.empty(0), _omni_req_id="a")["zonos2"]["history"]) == 1


def test_partial_prefill_does_not_sample_advance_seed_or_emit_codes(model):
    data = info("a", prompt_len=6)
    data["_zonos2_scheduled_span"] = 3
    model.preprocess(torch.zeros(3, dtype=torch.int64), None, **data)
    model._sampling_plan = [("a", False, data)]
    model._last_fused_logits = logits_for_codes([7])
    output = model.make_omni_output(torch.zeros(3, 2))
    model.sample(torch.zeros(1, 1026), None)
    assert len(model._request_states["a"].history) == 0
    assert output.multimodal_outputs["codes"]["audio"][0].numel() == 0
    assert model.postprocess(torch.empty(0), _omni_req_id="a") == {}


@pytest.mark.parametrize("reason", ["finish", "cancel", "error"])
def test_cleanup_releases_state_and_seed_and_allows_id_reuse(model, reason):
    data = info("a", seed=42)
    model.preprocess(torch.zeros(3, dtype=torch.int64), None, **data)
    model.on_requests_finished(iter(["a"]))
    assert model._request_states == {} and model._step_codes == {}
    model.preprocess(torch.zeros(3, dtype=torch.int64), None, **info("a", seed=99))
    assert model._request_states["a"].seed == 99


def test_sampler_error_cleans_active_batch_state(model, monkeypatch):
    data = info("a")
    data["_zonos2_scheduled_span"] = 3
    model.preprocess(torch.zeros(3, dtype=torch.int64), None, **data)
    model._sampling_plan = [("a", True, data)]
    model._last_fused_logits = logits_for_codes([7])

    def fail(*args, **kwargs):
        raise ValueError("sampling failed")

    monkeypatch.setattr("vllm_omni.model_executor.models.zonos2.zonos2_talker.sample_frame", fail)
    with pytest.raises(ValueError, match="sampling failed"):
        model.sample(torch.zeros(1, 1026), None)
    assert model._request_states == {} and model._step_codes == {}


def test_unseeded_preemption_reuses_persisted_base_seed(model):
    data = info("a", seed=None)
    model.preprocess(torch.zeros(3, dtype=torch.int64), None, **data)
    seed = model._request_states["a"].seed
    model.preprocess(torch.zeros(3, dtype=torch.int64), None, **data)
    assert model._request_states["a"].seed == seed


def test_preprocess_error_cleans_previous_request_state(model):
    data = info("a")
    model.preprocess(torch.zeros(3, dtype=torch.int64), None, **data)
    data["_omni_output_token_ids"] = [0] * 5
    with pytest.raises(ValueError, match="lifecycle"):
        model.preprocess(torch.zeros(3, dtype=torch.int64), None, **data)
    assert "a" not in model._request_states


def test_lifecycle_vocabulary_covers_stop_sentinel_and_text_padding():
    config = Zonos2Config()
    assert config.vocab_size == 1026
    assert 1 < config.vocab_size and config.text_vocab < config.vocab_size


def test_terminal_replay_does_not_append_duplicate_frame_or_consume_rng(model):
    history = torch.zeros((2, 9), dtype=torch.int32)
    data = info("a", history=history, count=2, max_tokens=2)
    data["_omni_output_token_ids"] = [0, 1]
    _, _, updates = model.preprocess(torch.zeros(5, dtype=torch.int64), None, **data)
    data.update(updates)
    model._sampling_plan = [("a", True, data)]
    model._last_fused_logits = logits_for_codes([7])
    output = model.make_omni_output(torch.zeros(5, 2))
    assert model.sample(torch.zeros(1, 1026), None).sampled_token_ids.tolist() == [[1]]
    assert len(model._request_states["a"].history) == 2
    assert output.multimodal_outputs["codes"]["audio"][0].numel() == 0


@pytest.mark.parametrize(
    "async_scheduling,prefix,eager,message",
    [(True, False, True, "async_scheduling"), (False, True, True, "prefix"), (False, False, False, "eager")],
)
def test_unsupported_capture_or_scheduler_modes_fail_before_allocating_model(async_scheduling, prefix, eager, message):
    cfg: Any = SimpleNamespace(
        scheduler_config=SimpleNamespace(async_scheduling=async_scheduling),
        cache_config=SimpleNamespace(enable_prefix_caching=prefix),
        model_config=SimpleNamespace(enforce_eager=eager),
    )
    with pytest.raises(ValueError, match=message):
        Zonos2TalkerForConditionalGeneration(vllm_config=cfg)


def test_decode_slices_one_frame_without_rebuilding_prompt_and_history(model, monkeypatch):
    history = torch.arange(64 * 9, dtype=torch.int32).reshape(64, 9) % 1024
    data = info("a", prompt_len=1000, offset=1063, history=history, count=64, max_tokens=128)
    shapes = []
    original = torch.cat

    def counted(tensors, *args, **kwargs):
        shapes.append([tuple(t.shape) for t in tensors])
        return original(tensors, *args, **kwargs)

    monkeypatch.setattr(torch, "cat", counted)
    _, actual, _ = model.preprocess(torch.zeros(1, dtype=torch.int64), None, **data)
    expected = original((history[-1:], torch.full((1, 1), 519, dtype=torch.int32)), dim=1)
    torch.testing.assert_close(actual, model.multi_embedder(expected), rtol=0, atol=0)
    assert shapes == [[(1, 9), (1, 1)]]
