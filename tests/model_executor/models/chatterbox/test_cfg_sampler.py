# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Original's guidance logits, on the CPU.

Model Runner V2's sampler runs on the GPU; the tests that sample through it
are in ``test_chatterbox_gpu.py``.
"""

import pytest
import torch

from vllm_omni.model_executor.models.chatterbox.cfg_sampler import ChatterboxCFGSampler
from vllm_omni.transformers_utils.configs.chatterbox import START_SPEECH_TOKEN, STOP_SPEECH_TOKEN

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_guidance_blends_both_rows_and_masks_non_speech_ids() -> None:
    """Either row alone would pick token 1; the guided logits, the same in both rows, pick 0."""
    sampler = ChatterboxCFGSampler({"c": ("pair", "cond", 0.5), "u": ("pair", "uncond", 0.5)})
    logits = torch.zeros(2, STOP_SPEECH_TOKEN + 3)
    logits[0, :2] = torch.tensor([-2.0, 1.5])
    logits[1, :2] = torch.tensor([1.0, 1.2])
    original = logits.clone()

    guided, pairs = sampler.prepare_logits(logits, ["u", "c"])

    # c + 0.5 * (c - u) in both rows.
    torch.testing.assert_close(guided[:, :2], torch.tensor([[2.5, 1.05], [2.5, 1.05]]))
    assert pairs == [(1, 0)]
    assert not torch.isnan(guided).any()
    assert torch.isneginf(guided[:, START_SPEECH_TOKEN]).all()
    assert torch.isneginf(guided[:, STOP_SPEECH_TOKEN + 1 :]).all()
    assert torch.isfinite(guided[:, STOP_SPEECH_TOKEN]).all()
    assert logits.argmax(dim=1).tolist() == [1, 1]
    assert guided.argmax(dim=1).tolist() == [0, 0]
    assert torch.equal(logits, original)


def test_pairs_are_found_by_request_id_in_any_row_order_and_in_the_current_registry() -> None:
    registry = {"c": ("a", "cond", 0.5), "u": ("a", "uncond", 0.5)}
    sampler = ChatterboxCFGSampler(registry)
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
    sampler = ChatterboxCFGSampler(registry)
    with pytest.raises(ValueError, match=message):
        sampler.prepare_logits(torch.zeros(len(req_ids), STOP_SPEECH_TOKEN + 1), req_ids)


@pytest.mark.parametrize("req_id", ["_warmup_0_", "_v2_mixed_warmup_decode_"])
def test_runner_warmup_rows_need_no_registration_and_stay_masked(req_id: str) -> None:
    """Model Runner V2 runs these before any request; refusing one stops the stage from starting."""
    logits = torch.ones(1, STOP_SPEECH_TOKEN + 3)

    guided, pairs = ChatterboxCFGSampler({}).prepare_logits(logits, [req_id])

    assert pairs == []
    assert torch.isneginf(guided[:, START_SPEECH_TOKEN]).all()
    assert torch.isneginf(guided[:, STOP_SPEECH_TOKEN + 1 :]).all()


def test_expanded_logits_are_unsupported() -> None:
    sampler = ChatterboxCFGSampler({})
    with pytest.raises(ValueError, match="one logits row per request"):
        sampler.prepare_logits(torch.zeros(3, STOP_SPEECH_TOKEN + 1), ["c", "u"])


def test_multiple_pairs_do_not_share_guidance() -> None:
    """Each pair is guided by its own weight."""
    registry = {
        "ac": ("a", "cond", 0.5),
        "au": ("a", "uncond", 0.5),
        "bc": ("b", "cond", 1.0),
        "bu": ("b", "uncond", 1.0),
    }
    sampler = ChatterboxCFGSampler(registry)
    logits = torch.zeros(4, STOP_SPEECH_TOKEN + 1)
    logits[:, 0] = torch.tensor([2.0, 10.0, 4.0, 6.0])

    guided, pairs = sampler.prepare_logits(logits, ["au", "bc", "ac", "bu"])

    torch.testing.assert_close(guided[:, 0], torch.tensor([5.0, 14.0, 5.0, 14.0]))
    assert sorted(pairs) == [(1, 3), (2, 0)]


def test_zero_weight_conditional_request_needs_no_companion() -> None:
    sampler = ChatterboxCFGSampler({"c": ("a", "cond", 0.0)})
    logits = torch.ones(1, STOP_SPEECH_TOKEN + 1)
    guided, pairs = sampler.prepare_logits(logits, ["c"])
    assert pairs == []
    torch.testing.assert_close(guided[:, :START_SPEECH_TOKEN], logits[:, :START_SPEECH_TOKEN])


@pytest.mark.parametrize("weight", [0.0, 0.5])
def test_shared_restrictions_survive_guidance(weight: float) -> None:
    masked_tokens = [2, 17, 123, STOP_SPEECH_TOKEN]
    sampler = ChatterboxCFGSampler({"c": ("pair", "cond", weight), "u": ("pair", "uncond", weight)})
    logits = torch.ones(2, STOP_SPEECH_TOKEN + 3)
    logits[0, 0], logits[1, 0] = 4.0, 2.0
    logits[:, masked_tokens] = -torch.inf
    original = logits.clone()

    guided, _ = sampler.prepare_logits(logits, ["c", "u"])

    assert not torch.isnan(guided).any()
    assert torch.isneginf(guided[:, masked_tokens]).all()
    assert torch.isneginf(guided[:, START_SPEECH_TOKEN]).all()
    assert torch.isneginf(guided[:, STOP_SPEECH_TOKEN + 1 :]).all()
    torch.testing.assert_close(guided[:, 0], torch.full((2,), 4.0 + weight * 2.0))
    torch.testing.assert_close(logits, original)
    assert guided.argmax(dim=1).tolist() == [0, 0]


@pytest.mark.parametrize("restricted_row", [0, 1], ids=["cond", "uncond"])
def test_an_id_restricted_on_one_row_is_banned_from_the_blend(restricted_row: int) -> None:
    """Unmasked, an id only the unconditional row restricts would blend to cond - (-inf) = +inf and be forced."""
    sampler = ChatterboxCFGSampler({"c": ("pair", "cond", 0.5), "u": ("pair", "uncond", 0.5)})
    logits = torch.ones(2, STOP_SPEECH_TOKEN + 3)
    logits[:, 0] = 4.0
    logits[restricted_row, 7] = -torch.inf

    guided, _ = sampler.prepare_logits(logits, ["c", "u"])

    assert not torch.isnan(guided).any() and not torch.isposinf(guided).any()
    assert torch.isneginf(guided[:, 7]).all()
    assert guided.argmax(dim=1).tolist() == [0, 0]


def test_a_pair_restricted_to_a_few_ids_keeps_them_sampleable() -> None:
    """allowed_token_ids masks every other id on both rows; the blend leaves the allowed ones finite.

    The companion's parameters are a clone of the conditional request's, so
    the rows carry the same mask and the union bans nothing more than either.
    """
    allowed = [5, 9]
    sampler = ChatterboxCFGSampler({"c": ("pair", "cond", 0.5), "u": ("pair", "uncond", 0.5)})
    logits = torch.full((2, STOP_SPEECH_TOKEN + 3), -torch.inf)
    logits[:, allowed] = torch.tensor([[1.0, 3.0], [2.0, 0.0]])

    guided, _ = sampler.prepare_logits(logits, ["c", "u"])

    assert torch.isfinite(guided[:, allowed]).all()
    assert torch.isfinite(guided).sum(dim=1).tolist() == [len(allowed)] * 2
    # 1 + 0.5 * (1 - 2) = 0.5 against 3 + 0.5 * (3 - 0) = 4.5.
    assert guided.argmax(dim=1).tolist() == [9, 9]
