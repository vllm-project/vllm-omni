# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Original's guidance around vLLM's V1 sampler, sampling for real on the CPU.

Model Runner V2's sampler runs on the GPU; its guidance tests are in
``test_chatterbox_gpu.py``.
"""

from collections.abc import Iterator

import pytest
import torch
from vllm.v1.sample.logits_processor import LogitsProcessors
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler

from vllm_omni.model_executor.models.chatterbox.cfg_sampler import ChatterboxCFGSampler
from vllm_omni.transformers_utils.configs.chatterbox import START_SPEECH_TOKEN, STOP_SPEECH_TOKEN

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture(scope="module")
def v1_sampler() -> Iterator[Sampler]:
    """vLLM's V1 sampler, the one Original's stage builds as its delegate."""
    with pytest.MonkeyPatch.context() as patch:
        # It otherwise asks the GPU whether FlashInfer can sample; these logits are on the CPU.
        patch.setenv("VLLM_USE_FLASHINFER_SAMPLER", "0")
        yield Sampler()


def sampling_metadata(rows: int, seeds: list[int] | None = None) -> SamplingMetadata:
    """The V1 runner's metadata for a step: greedy rows, or random rows each with its own seeded generator.

    The V1 input batch that builds it pins host memory, which needs a GPU; the
    GPU tests sample through it.
    """
    return SamplingMetadata(
        temperature=None if seeds is None else torch.ones(rows),
        all_greedy=seeds is None,
        all_random=seeds is not None,
        top_p=None,
        top_k=None,
        generators={}
        if seeds is None
        else {row: torch.Generator().manual_seed(seed) for row, seed in enumerate(seeds)},
        max_num_logprobs=None,
        no_penalties=True,
        prompt_token_ids=None,
        frequency_penalties=torch.zeros(rows),
        presence_penalties=torch.zeros(rows),
        repetition_penalties=torch.ones(rows),
        output_token_ids=[[] for _ in range(rows)],
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=LogitsProcessors(),
    )


def drawn_alone(sampler: Sampler, row: torch.Tensor, seed: int) -> int:
    """The token ``row`` samples by itself with a request's own seed."""
    return int(sampler(logits=row[None].clone(), sampling_metadata=sampling_metadata(1, [seed])).sampled_token_ids)


def test_guidance_precedes_delegate_and_tokens_follow_conditional(v1_sampler: Sampler) -> None:
    """Both rows are sampled from the guided logits, and the companion takes the conditional's token.

    Greedy: either row alone would pick token 1, the guided logits pick 0.
    Seeded: each row draws from its own generator, and the unconditional row
    still ends with the token the conditional row draws by itself.
    """
    registry = {"c": ("pair", "cond", 0.5), "u": ("pair", "uncond", 0.5)}
    sampler = ChatterboxCFGSampler(v1_sampler, registry)
    extra_args = [{"chatterbox_cfg": registry[rid]} for rid in ("u", "c")]
    logits = torch.zeros(2, STOP_SPEECH_TOKEN + 3)
    logits[0, :2] = torch.tensor([-2.0, 1.5])
    logits[1, :2] = torch.tensor([1.0, 1.2])
    original = logits.clone()

    guided, pairs = sampler.prepare_logits(logits, ["u", "c"])
    greedy = sampler.sample(logits, sampling_metadata(2), extra_args)

    # c + 0.5 * (c - u) in both rows.
    torch.testing.assert_close(guided[:, :2], torch.tensor([[2.5, 1.05], [2.5, 1.05]]))
    assert pairs == [(1, 0)]
    assert not torch.isnan(guided).any()
    assert torch.isneginf(guided[:, START_SPEECH_TOKEN]).all()
    assert torch.isneginf(guided[:, STOP_SPEECH_TOKEN + 1 :]).all()
    assert torch.isfinite(guided[:, STOP_SPEECH_TOKEN]).all()
    assert logits.argmax(dim=1).tolist() == [1, 1]
    assert greedy.sampled_token_ids.tolist() == [[0], [0]]
    assert torch.equal(logits, original)

    flat = torch.zeros(2, STOP_SPEECH_TOKEN + 3)
    flat_guided, _ = sampler.prepare_logits(flat, ["u", "c"])
    seeded = sampler.sample(flat, sampling_metadata(2, [1, 2]), extra_args)

    conditional = drawn_alone(v1_sampler, flat_guided[1], 2)
    assert drawn_alone(v1_sampler, flat_guided[0], 1) != conditional
    assert seeded.sampled_token_ids.tolist() == [[conditional], [conditional]]


def test_pairs_are_found_by_request_id_in_any_row_order_and_in_the_current_registry(v1_sampler: Sampler) -> None:
    registry = {"c": ("a", "cond", 0.5), "u": ("a", "uncond", 0.5)}
    sampler = ChatterboxCFGSampler(v1_sampler, registry)
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
def test_invalid_pairs_fail_before_sampling(v1_sampler: Sampler, registry, req_ids, message):
    sampler = ChatterboxCFGSampler(v1_sampler, registry)
    with pytest.raises(ValueError, match=message):
        sampler.prepare_logits(torch.zeros(len(req_ids), STOP_SPEECH_TOKEN + 1), req_ids)


def test_expanded_logits_are_unsupported(v1_sampler: Sampler) -> None:
    sampler = ChatterboxCFGSampler(v1_sampler, {})
    with pytest.raises(ValueError, match="one logits row per request"):
        sampler.prepare_logits(torch.zeros(3, STOP_SPEECH_TOKEN + 1), ["c", "u"])


def test_multiple_pairs_do_not_share_guidance_or_tokens(v1_sampler: Sampler) -> None:
    """Each pair is guided by its own weight, and each companion takes its own conditional's token."""
    registry = {
        "ac": ("a", "cond", 0.5),
        "au": ("a", "uncond", 0.5),
        "bc": ("b", "cond", 1.0),
        "bu": ("b", "uncond", 1.0),
    }
    sampler = ChatterboxCFGSampler(v1_sampler, registry)
    req_ids = ["au", "bc", "ac", "bu"]
    extra_args = [{"chatterbox_cfg": registry[rid]} for rid in req_ids]
    logits = torch.zeros(4, STOP_SPEECH_TOKEN + 1)
    logits[:, 0] = torch.tensor([2.0, 10.0, 4.0, 6.0])

    guided, _ = sampler.prepare_logits(logits, req_ids)
    torch.testing.assert_close(guided[:, 0], torch.tensor([5.0, 14.0, 5.0, 14.0]))

    flat = torch.zeros(4, STOP_SPEECH_TOKEN + 1)
    flat_guided, _ = sampler.prepare_logits(flat, req_ids)
    seeds = [11, 12, 13, 14]
    output = sampler.sample(flat, sampling_metadata(4, seeds), extra_args)

    a, b = drawn_alone(v1_sampler, flat_guided[2], seeds[2]), drawn_alone(v1_sampler, flat_guided[1], seeds[1])
    assert a != b
    assert output.sampled_token_ids.tolist() == [[a], [b], [a], [b]]


def test_zero_weight_conditional_request_needs_no_companion(v1_sampler: Sampler) -> None:
    sampler = ChatterboxCFGSampler(v1_sampler, {"c": ("a", "cond", 0.0)})
    logits = torch.ones(1, STOP_SPEECH_TOKEN + 1)
    guided, pairs = sampler.prepare_logits(logits, ["c"])
    assert pairs == []
    torch.testing.assert_close(guided[:, :START_SPEECH_TOKEN], logits[:, :START_SPEECH_TOKEN])


@pytest.mark.parametrize("weight", [0.0, 0.5])
def test_v1_shared_logit_bias_masks_survive_guidance(v1_sampler: Sampler, weight: float) -> None:
    masked_tokens = [2, 17, 123, STOP_SPEECH_TOKEN]
    registry = {"c": ("pair", "cond", weight), "u": ("pair", "uncond", weight)}
    sampler = ChatterboxCFGSampler(v1_sampler, registry)
    # V1's runner applies min_tokens/allowed_token_ids before model sampling.
    logits = torch.ones(2, STOP_SPEECH_TOKEN + 3)
    logits[0, 0], logits[1, 0] = 4.0, 2.0
    logits[:, masked_tokens] = -torch.inf
    original = logits.clone()

    guided, _ = sampler.prepare_logits(logits, ["c", "u"])
    output = sampler.sample(logits, sampling_metadata(2), [{"chatterbox_cfg": registry[rid]} for rid in ["c", "u"]])

    assert not torch.isnan(guided).any()
    assert torch.isneginf(guided[:, masked_tokens]).all()
    assert torch.isneginf(guided[:, START_SPEECH_TOKEN]).all()
    assert torch.isneginf(guided[:, STOP_SPEECH_TOKEN + 1 :]).all()
    torch.testing.assert_close(guided[:, 0], torch.full((2,), 4.0 + weight * 2.0))
    torch.testing.assert_close(logits, original)
    assert output.sampled_token_ids.tolist() == [[0], [0]]


@pytest.mark.parametrize("restricted_row", [0, 1], ids=["cond", "uncond"])
def test_an_id_restricted_on_one_row_is_banned_from_the_blend(v1_sampler: Sampler, restricted_row: int) -> None:
    """Unmasked, an id only the unconditional row restricts would blend to cond - (-inf) = +inf and be forced."""
    registry = {"c": ("pair", "cond", 0.5), "u": ("pair", "uncond", 0.5)}
    sampler = ChatterboxCFGSampler(v1_sampler, registry)
    logits = torch.ones(2, STOP_SPEECH_TOKEN + 3)
    logits[:, 0] = 4.0
    logits[restricted_row, 7] = -torch.inf

    guided, _ = sampler.prepare_logits(logits, ["c", "u"])
    output = sampler.sample(logits, sampling_metadata(2), [{"chatterbox_cfg": registry[rid]} for rid in ["c", "u"]])

    assert not torch.isnan(guided).any() and not torch.isposinf(guided).any()
    assert torch.isneginf(guided[:, 7]).all()
    assert output.sampled_token_ids.tolist() == [[0], [0]]
