# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.v1.worker.gpu.sample.output import SamplerOutput

from vllm_omni.model_executor.models.chatterbox.cfg_sampler import ChatterboxCFGSampler, ChatterboxMRv2CFGSampler
from vllm_omni.model_executor.models.chatterbox.chatterbox_original_t3 import ChatterboxOriginalT3
from vllm_omni.model_executor.models.chatterbox.chatterbox_t3 import ChatterboxT3ForConditionalGeneration
from vllm_omni.transformers_utils.configs.chatterbox import ORIGINAL_DEFAULT_CFG_WEIGHT as DEFAULT_CFG_WEIGHT
from vllm_omni.transformers_utils.configs.chatterbox import START_SPEECH_TOKEN, STOP_SPEECH_TOKEN
from vllm_omni.worker.sampling_utils import call_model_sampler

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("stage_type", [ChatterboxT3ForConditionalGeneration, ChatterboxOriginalT3])
def test_mrv2_sampling_transports_tokens_without_unused_hidden_states(stage_type):
    stage = object.__new__(stage_type)
    torch.nn.Module.__init__(stage)
    stage.cfg_pairs = {}
    sampler, _ = stage.mrv2_custom_sampler(None)
    hidden = torch.randn(2, 16)
    counts, rejected = torch.ones(2, dtype=torch.int32), torch.zeros(2, dtype=torch.int32)
    sampled = SamplerOutput(torch.tensor([[4], [4]]), None, None, counts, rejected)
    calls = []

    def standard_sample(hidden_states, input_batch, grammar_output):
        calls.append(hidden_states)
        return sampled, counts, rejected

    result = sampler.sample_step(hidden, None, None, None, standard_sample)
    assert len(calls) == 1 and calls[0] is hidden
    assert result.sampler_output is sampled
    assert result.num_sampled is counts and result.num_rejected is rejected
    assert result.multimodal_outputs is None  # Preserve forward's per-token streaming metadata.
    assert result.include_hidden_states is False
    assert result.owns_multimodal_outputs is True


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
    batch = SimpleNamespace(req_ids=["u", "c"], prefill_len_np=np.array([42, 42]))
    logits = torch.zeros(2, STOP_SPEECH_TOKEN + 3)
    logits[0, :2] = torch.tensor([2.0, -4.0])
    logits[1, :2] = torch.tensor([4.0, -2.0])
    metadata = object()
    output = (
        sampler(logits, batch)
        if mrv2
        else call_model_sampler(
            SimpleNamespace(model_sampler_wants_extra_args=True),
            sampler.sample,
            logits,
            metadata,
            input_batch=batch,
            requests={
                rid: SimpleNamespace(sampling_params=SimpleNamespace(extra_args={"chatterbox_cfg": registry[rid]}))
                for rid in batch.req_ids
            },
        )
    )
    torch.testing.assert_close(delegate.logits[:, :2], torch.tensor([[5.0, -1.0], [5.0, -1.0]]))
    assert not torch.isnan(delegate.logits).any()
    assert torch.isneginf(delegate.logits[:, START_SPEECH_TOKEN]).all()
    assert torch.isneginf(delegate.logits[:, STOP_SPEECH_TOKEN + 1 :]).all()
    assert torch.isfinite(delegate.logits[:, STOP_SPEECH_TOKEN]).all()
    assert output.sampled_token_ids.tolist() == [[1], [1]]
    assert logits[:, START_SPEECH_TOKEN].tolist() == [0, 0]
    if mrv2:
        assert delegate.input_batch is batch
    else:
        assert delegate.metadata is metadata


def test_batch_reordering_and_registry_cleanup():
    registry = {"c": ("a", "cond", 0.5), "u": ("a", "uncond", 0.5)}
    sampler = ChatterboxCFGSampler(RecordingSampler(), registry)
    logits = torch.zeros(2, STOP_SPEECH_TOKEN + 1)
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
        sampler.prepare_logits(torch.zeros(len(req_ids), STOP_SPEECH_TOKEN + 1), req_ids)


def test_expanded_logits_are_unsupported():
    sampler = ChatterboxCFGSampler(RecordingSampler(), {})
    with pytest.raises(ValueError, match="one logits row per request"):
        sampler.prepare_logits(torch.zeros(3, STOP_SPEECH_TOKEN + 1), ["c", "u"])


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
    logits = torch.zeros(4, STOP_SPEECH_TOKEN + 1)
    logits[:, 0] = torch.tensor([2.0, 10.0, 4.0, 6.0])
    output = sampler.sample(logits, object(), [{"chatterbox_cfg": registry[rid]} for rid in batch.req_ids])
    torch.testing.assert_close(delegate.logits[:, 0], torch.tensor([5.0, 14.0, 5.0, 14.0]))
    assert output.sampled_token_ids.tolist() == [[2], [1], [2], [1]]


def test_zero_weight_conditional_request_needs_no_companion():
    sampler = ChatterboxCFGSampler(RecordingSampler(), {"c": ("a", "cond", 0.0)})
    logits = torch.ones(1, STOP_SPEECH_TOKEN + 1)
    guided, pairs = sampler.prepare_logits(logits, ["c"])
    assert pairs == []
    torch.testing.assert_close(guided[:, :START_SPEECH_TOKEN], logits[:, :START_SPEECH_TOKEN])


def test_synthetic_mrv2_warmup_delegates_without_registration():
    delegate = RecordingSampler()
    sampler = ChatterboxMRv2CFGSampler(delegate, {})
    logits = torch.ones(2, STOP_SPEECH_TOKEN + 3)
    batch = SimpleNamespace(req_ids=["req_0_random", "req_1_random"], prefill_len_np=np.zeros(2, dtype=np.int32))
    output = sampler(logits, batch)
    assert output.sampled_token_ids.tolist() == [[0], [1]]
    torch.testing.assert_close(delegate.logits[:, :START_SPEECH_TOKEN], logits[:, :START_SPEECH_TOKEN])
    assert torch.isneginf(delegate.logits[:, START_SPEECH_TOKEN]).all()
    assert torch.isneginf(delegate.logits[:, STOP_SPEECH_TOKEN + 1 :]).all()


def test_mrv2_real_prompt_requires_registration():
    sampler = ChatterboxMRv2CFGSampler(RecordingSampler(), {})
    batch = SimpleNamespace(req_ids=["req_0_random"], prefill_len_np=np.array([42]))
    with pytest.raises(ValueError, match="Missing Chatterbox CFG registration"):
        sampler(torch.zeros(1, STOP_SPEECH_TOKEN + 1), batch)


def test_registered_kernel_warmup_delegates_without_guidance():
    delegate = RecordingSampler()
    sampler = ChatterboxMRv2CFGSampler(delegate, {})
    batch = SimpleNamespace(req_ids=["_warmup_0_"], prefill_len_np=np.array([1]))
    sampler(torch.ones(1, STOP_SPEECH_TOKEN + 3), batch)
    assert torch.isneginf(delegate.logits[:, START_SPEECH_TOKEN]).all()
    assert torch.isneginf(delegate.logits[:, STOP_SPEECH_TOKEN + 1 :]).all()


@pytest.mark.parametrize("weight", [0.0, 0.5])
def test_v1_shared_logit_bias_masks_survive_guidance(weight):
    masked_tokens = [2, 17, 123, STOP_SPEECH_TOKEN]
    registry = {"c": ("pair", "cond", weight), "u": ("pair", "uncond", weight)}
    delegate = RecordingSampler()
    sampler = ChatterboxCFGSampler(delegate, registry)
    # V1's runner applies min_tokens/allowed_token_ids before model sampling.
    logits = torch.ones(2, STOP_SPEECH_TOKEN + 3)
    logits[0, 0], logits[1, 0] = 4.0, 2.0
    logits[:, masked_tokens] = -torch.inf
    original = logits.clone()
    output = sampler.sample(logits, object(), [{"chatterbox_cfg": registry[rid]} for rid in ["c", "u"]])
    assert not torch.isnan(delegate.logits).any()
    assert torch.isneginf(delegate.logits[:, masked_tokens]).all()
    assert torch.isneginf(delegate.logits[:, START_SPEECH_TOKEN]).all()
    assert torch.isneginf(delegate.logits[:, STOP_SPEECH_TOKEN + 1 :]).all()
    torch.testing.assert_close(delegate.logits[:, 0], torch.full((2,), 4.0 + weight * 2.0))
    torch.testing.assert_close(logits, original)
    assert output.sampled_token_ids.tolist() == [[0], [0]]
