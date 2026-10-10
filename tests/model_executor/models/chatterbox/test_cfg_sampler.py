# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.chatterbox.cfg_sampler import (
    DEFAULT_CFG_WEIGHT,
    SPEECH_START_TOKEN,
    SPEECH_STOP_TOKEN,
    ChatterboxCFGSampler,
    ChatterboxMRv2CFGSampler,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class RecordingSampler:
    """A delegate that exposes its input and deliberately samples distinct rows."""

    def __call__(self, logits, input_batch=None, *, sampling_metadata=None):
        self.logits = logits.clone()
        self.metadata = sampling_metadata
        self.input_batch = input_batch
        return SimpleNamespace(sampled_token_ids=torch.arange(logits.shape[0]).reshape(-1, 1))


@pytest.mark.parametrize("mrv2", [False, True])
def test_guidance_precedes_delegate_and_tokens_follow_conditional(mrv2):
    registry = {"c": ("pair", "cond", DEFAULT_CFG_WEIGHT), "u": ("pair", "uncond", DEFAULT_CFG_WEIGHT)}
    delegate = RecordingSampler()
    sampler = ChatterboxMRv2CFGSampler(delegate, registry) if mrv2 else ChatterboxCFGSampler(delegate, registry)
    batch = SimpleNamespace(req_ids=["u", "c"])
    logits = torch.zeros(2, SPEECH_STOP_TOKEN + 3)
    logits[0, :2] = torch.tensor([2.0, -4.0])
    logits[1, :2] = torch.tensor([4.0, -2.0])
    metadata = object()
    output = sampler(logits, batch) if mrv2 else sampler.sample(logits, metadata, batch)
    torch.testing.assert_close(delegate.logits[:, :2], torch.tensor([[5.0, -1.0], [5.0, -1.0]]))
    assert not torch.isnan(delegate.logits).any()
    assert torch.isneginf(delegate.logits[:, SPEECH_START_TOKEN]).all()
    assert torch.isneginf(delegate.logits[:, SPEECH_STOP_TOKEN + 1 :]).all()
    assert torch.isfinite(delegate.logits[:, SPEECH_STOP_TOKEN]).all()
    assert output.sampled_token_ids.tolist() == [[1], [1]]
    assert logits[:, SPEECH_START_TOKEN].tolist() == [0, 0]
    if mrv2:
        assert delegate.input_batch is batch
    else:
        assert delegate.metadata is metadata


def test_batch_reordering_and_registry_cleanup():
    registry = {"c": ("a", "cond", 0.5), "u": ("a", "uncond", 0.5)}
    sampler = ChatterboxCFGSampler(RecordingSampler(), registry)
    logits = torch.zeros(2, SPEECH_STOP_TOKEN + 1)
    assert sampler.prepare_logits(logits, ["c", "u"])[1] == [(0, 1)]
    assert sampler.prepare_logits(logits, ["u", "c"])[1] == [(1, 0)]
    registry.clear()
    registry.update({"new-u": ("b", "uncond", 0.0), "new-c": ("b", "cond", 0.0)})
    assert sampler.prepare_logits(logits, ["new-u", "new-c"])[1] == [(1, 0)]


@pytest.mark.parametrize(
    ("registry", "req_ids", "message"),
    [
        ({}, ["c"], "Missing Chatterbox CFG registration"),
        ({"c": ("a", "cond", 0.5)}, ["c"], "Missing Chatterbox CFG companion"),
        ({"c": ("a", "invalid", 0.5)}, ["c"], "Unsupported Chatterbox CFG role"),
        ({"c": ("a", "cond", 0.5), "u": ("a", "cond", 0.5)}, ["c", "u"], "Duplicate"),
        ({"c": ("a", "cond", 0.5), "u": ("a", "uncond", 0.4)}, ["c", "u"], "Mismatched"),
    ],
)
def test_invalid_pairs_fail_before_sampling(registry, req_ids, message):
    sampler = ChatterboxCFGSampler(RecordingSampler(), registry)
    with pytest.raises(ValueError, match=message):
        sampler.prepare_logits(torch.zeros(len(req_ids), SPEECH_STOP_TOKEN + 1), req_ids)


def test_expanded_logits_are_unsupported():
    sampler = ChatterboxCFGSampler(RecordingSampler(), {})
    with pytest.raises(ValueError, match="one logits row per request"):
        sampler.prepare_logits(torch.zeros(3, SPEECH_STOP_TOKEN + 1), ["c", "u"])


def test_multiple_pairs_do_not_share_guidance_or_tokens():
    registry = {
        "ac": ("a", "cond", 0.5),
        "au": ("a", "uncond", 0.5),
        "bc": ("b", "cond", 1.0),
        "bu": ("b", "uncond", 1.0),
    }
    delegate = RecordingSampler()
    sampler = ChatterboxCFGSampler(delegate, registry)
    batch = SimpleNamespace(req_ids=["au", "bc", "ac", "bu"])
    logits = torch.zeros(4, SPEECH_STOP_TOKEN + 1)
    logits[:, 0] = torch.tensor([2.0, 10.0, 4.0, 6.0])
    output = sampler.sample(logits, object(), batch)
    torch.testing.assert_close(delegate.logits[:, 0], torch.tensor([5.0, 14.0, 5.0, 14.0]))
    assert output.sampled_token_ids.tolist() == [[2], [1], [2], [1]]


def test_zero_weight_conditional_request_needs_no_companion():
    sampler = ChatterboxCFGSampler(RecordingSampler(), {"c": ("a", "cond", 0.0)})
    logits = torch.ones(1, SPEECH_STOP_TOKEN + 1)
    guided, pairs = sampler.prepare_logits(logits, ["c"])
    assert pairs == []
    torch.testing.assert_close(guided[:, :SPEECH_START_TOKEN], logits[:, :SPEECH_START_TOKEN])


def test_synthetic_mrv2_warmup_delegates_without_registration():
    delegate = RecordingSampler()
    sampler = ChatterboxMRv2CFGSampler(delegate, {})
    logits = torch.ones(2, SPEECH_STOP_TOKEN + 3)
    batch = SimpleNamespace(req_ids=["_warmup_0", "_warmup_1"])
    output = sampler(logits, batch)
    assert output.sampled_token_ids.tolist() == [[0], [1]]
    torch.testing.assert_close(delegate.logits[:, :SPEECH_START_TOKEN], logits[:, :SPEECH_START_TOKEN])
    assert torch.isneginf(delegate.logits[:, SPEECH_START_TOKEN]).all()
    assert torch.isneginf(delegate.logits[:, SPEECH_STOP_TOKEN + 1 :]).all()
