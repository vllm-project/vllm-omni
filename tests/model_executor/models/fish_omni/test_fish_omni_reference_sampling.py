# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Drama 3 reference sampling (CPU): the keep rule, epsilon draws over hundreds of
candidates, the repetition check with both fallbacks, residual codes inside the
codec's codebooks, and the switch and epsilon_cutoff parameter."""

import math
import types

import pytest
import torch

from vllm_omni.model_executor.models.fish_omni.reference_sampling import (
    ReferenceFrameSampler,
    reference_sampling_enabled,
    reference_sampling_fallback,
    resolve_epsilon_cutoff,
    truncate,
    truncated_sample,
)
from vllm_omni.model_executor.models.fish_omni.sampling import RAS_TOP_K
from vllm_omni.model_executor.models.fish_speech.dac_utils import DAC_RESIDUAL_CODEBOOK_SIZE

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

EOS = 5
SEMANTIC_BEGIN = 100
CODEBOOK_SIZE = 64
CANDIDATES = torch.tensor([EOS, *range(SEMANTIC_BEGIN, SEMANTIC_BEGIN + CODEBOOK_SIZE)])
EMPTY = -1
# The released checkpoint's main-token candidates: end of speech and semantic codes 1..4095.
CANDIDATE_COUNT = 4096
HEAD_WIDTH = 4096
# Six columns' probabilities; descending they are columns 3, 0, 5, 1, 4, 2 with
# cumulative sums 0.40, 0.65, 0.80, 0.90, 0.96, 1.00.
SIX_COLUMNS = [0.25, 0.10, 0.04, 0.40, 0.06, 0.15]
DESCENDING = [3, 0, 5, 1, 4, 2]


def _rows(value, rows, dtype=torch.float32):
    return torch.full((rows,), value, dtype=dtype)


def _uniform(rows, width, seed):
    return torch.rand((rows, width), generator=torch.Generator().manual_seed(seed))


def _kept(logits, temperature, top_k, top_p, epsilon_cutoff, legal_ids=None):
    """The columns of a one-row ``logits`` the rules keep."""
    _, order, keep = truncate(logits.unsqueeze(0), temperature, top_k, top_p, epsilon_cutoff, legal_ids)
    return set(order[keep].tolist())


def _expected(logits, temperature, top_k, top_p, epsilon_cutoff, legal_ids=None):
    """Float64 kept probabilities of one row, renormalised, rank by rank."""
    probs = torch.softmax(logits.double() / max(temperature, 1e-5), dim=-1)
    order = torch.argsort(probs, descending=True, stable=True).tolist()
    keep = torch.zeros_like(probs, dtype=torch.bool)
    cumulative = 0.0
    for rank, column in enumerate(order):
        cumulative += float(probs[column])
        keep[column] = (top_k < 1 or rank < top_k) and (
            rank == 0 or ((top_p >= 1.0 or cumulative <= top_p) and float(probs[column]) >= epsilon_cutoff)
        )
    if legal_ids is not None:
        keep[legal_ids:] = False
        keep[next(column for column in order if column < legal_ids)] = True
    kept = probs * keep
    return kept / kept.sum()


def _assert_follows(draws, expected):
    """Every frequency within five standard errors of its kept probability, and no draw outside the kept set."""
    frequencies = torch.bincount(draws, minlength=expected.numel()).double() / draws.numel()
    errors = (expected * (1 - expected) / draws.numel()).sqrt()
    assert bool(((frequencies - expected).abs() <= 5 * errors + 1e-9).all())
    assert bool((frequencies[expected == 0] == 0).all())


@pytest.mark.parametrize(
    ("temperature", "top_k", "top_p", "epsilon_cutoff", "kept_ranks"),
    [
        pytest.param(1.0, -1, 1.0, 0.0, 6, id="every-rule-off"),
        pytest.param(1.0, 0, 1.0, 0.0, 6, id="top-k-zero-is-off"),
        pytest.param(1.0, 2, 1.0, 0.0, 2, id="top-k"),
        pytest.param(1.0, -1, 0.85, 0.0, 3, id="top-p-drops-the-crossing-rank"),
        pytest.param(1.0, -1, 0.3, 0.0, 1, id="top-p-keeps-rank-zero"),
        pytest.param(1.0, -1, 1.0, 0.12, 3, id="epsilon"),
        pytest.param(1.0, -1, 1.0, 0.05, 5, id="epsilon-lower"),
        pytest.param(1.0, -1, 1.0, 0.5, 1, id="epsilon-keeps-rank-zero"),
        pytest.param(1.0, 3, 0.95, 0.05, 3, id="joint"),
        # 0.25 would pass 0.3 only if top-k renormalised the two ranks it keeps.
        pytest.param(1.0, 2, 1.0, 0.3, 1, id="rules-read-one-distribution"),
        # At temperature 2 the five most probable columns reach 0.1.
        pytest.param(2.0, -1, 1.0, 0.1, 5, id="epsilon-after-temperature"),
    ],
)
def test_the_kept_set_follows_the_keep_rule(temperature, top_k, top_p, epsilon_cutoff, kept_ranks):
    logits = torch.tensor([math.log(p) for p in SIX_COLUMNS])
    assert _kept(logits, temperature, top_k, top_p, epsilon_cutoff) == set(DESCENDING[:kept_ranks])


def test_an_epsilon_only_row_keeps_hundreds_of_candidates_in_proportion():
    """No top-k or top-p, epsilon 1e-3: the 300 candidates above it (far more than the fused
    sampler's 64) are drawn in proportion to p, and nothing else is."""
    generator = torch.Generator().manual_seed(4)
    probs = torch.cat(
        (
            torch.empty(300).uniform_(1.5, 2.5, generator=generator),
            torch.empty(CANDIDATE_COUNT - 300).uniform_(0.5, 1.5, generator=generator),
        )
    )
    probs[:300] *= 0.6 / probs[:300].sum()
    probs[300:] *= 0.4 / probs[300:].sum()
    logits = probs[torch.randperm(CANDIDATE_COUNT, generator=generator)].log()
    expected = _expected(logits, 1.0, -1, 1.0, 1e-3)
    rows, chunks = 5000, 6

    draws = torch.cat(
        [
            truncated_sample(
                logits.repeat(rows, 1),
                _rows(1.0, rows),
                _rows(-1, rows, torch.int64),
                _rows(1.0, rows),
                _rows(1e-3, rows),
                _uniform(rows, CANDIDATE_COUNT, seed),
            )
            for seed in range(chunks)
        ]
    )

    assert int((expected > 0).sum()) == 300
    assert _kept(logits, 1.0, -1, 1.0, 1e-3) == set(torch.nonzero(expected).flatten().tolist())
    _assert_follows(draws, expected)
    kept = expected > 0
    frequencies = torch.bincount(draws, minlength=CANDIDATE_COUNT).double() / draws.numel()
    chi_square = float(((frequencies[kept] - expected[kept]) ** 2 / expected[kept]).sum()) * draws.numel()
    assert chi_square < 299 + 6 * math.sqrt(2 * 299)


@pytest.mark.parametrize(
    ("temperature", "top_k", "top_p", "epsilon_cutoff"),
    [
        pytest.param(1.0, -1, 1.0, 1e-3, id="recommended"),
        pytest.param(0.9, 40, 0.9, 2e-3, id="every-rule"),
        pytest.param(1.3, -1, 0.97, 5e-3, id="top-p-and-epsilon"),
    ],
)
def test_draws_follow_the_renormalised_kept_probabilities(temperature, top_k, top_p, epsilon_cutoff):
    logits = torch.randn(1024, generator=torch.Generator().manual_seed(11)) * 1.5
    expected = _expected(logits, temperature, top_k, top_p, epsilon_cutoff)
    rows = 20000

    draws = truncated_sample(
        logits.repeat(rows, 1),
        _rows(temperature, rows),
        _rows(top_k, rows, torch.int64),
        _rows(top_p, rows),
        _rows(epsilon_cutoff, rows),
        _uniform(rows, 1024, seed=1),
    )

    assert _kept(logits, temperature, top_k, top_p, epsilon_cutoff) == set(torch.nonzero(expected).flatten().tolist())
    _assert_follows(draws, expected)


def test_every_draw_stays_in_its_rows_kept_set():
    """Rows with mixed parameters, the legal-id rule included, never draw an index their rules drop."""
    generator = torch.Generator().manual_seed(5)
    rows, width = 3000, 256

    def pick(values):
        return torch.tensor(values)[torch.randint(0, len(values), (rows,), generator=generator)]

    logits = torch.randn(rows, width, generator=generator) * (torch.rand(rows, 1, generator=generator) * 4.0)
    params = (
        pick([0.0, 0.5, 1.0, 1.7]),
        pick([-1, 0, 1, 5, 50, 300]),
        pick([0.3, 0.8, 0.95, 1.0]),
        pick([0.0, 1e-3, 1e-2, 0.2]),
    )
    for legal_ids in (None, 100):
        draws = truncated_sample(logits, *params, _uniform(rows, width, seed=2), legal_ids=legal_ids)
        _, order, keep = truncate(logits, *params, legal_ids)
        kept = torch.zeros_like(keep).scatter_(-1, order, keep)
        assert bool(kept[torch.arange(rows), draws].all())


def _candidate_logits(probs, rows):
    """Candidate logits with these token probabilities; every other candidate is negligible."""
    logits = torch.full((rows, CANDIDATES.numel()), -30.0)
    for token, prob in probs.items():
        logits[:, int(torch.searchsorted(CANDIDATES, token))] = math.log(prob)
    return logits


def _window(tokens, rows):
    return torch.tensor([EMPTY] * (16 - len(tokens)) + tokens).expand(rows, -1)


def _main_tokens(logits, window, fallback, epsilon_cutoff=1e-3, seed=0):
    rows = logits.shape[0]
    sampler = ReferenceFrameSampler(fallback=fallback, num_codebooks=10, width=CANDIDATES.numel())
    noise = torch.rand((rows, sampler.num_draws, sampler.width), generator=torch.Generator().manual_seed(seed))
    tokens, code_zero = sampler.main_tokens(
        logits,
        CANDIDATES,
        window,
        _rows(1.0, rows),
        _rows(-1, rows, torch.int64),
        _rows(1.0, rows),
        _rows(epsilon_cutoff, rows),
        noise,
        SEMANTIC_BEGIN,
        CODEBOOK_SIZE,
    )
    assert torch.equal(code_zero, (tokens - SEMANTIC_BEGIN).clamp(0, CODEBOOK_SIZE - 1))
    return tokens


def test_a_repeated_first_draw_takes_the_most_probable_candidate_end_of_speech_included():
    """Drawn end-of-speech and semantic tokens found in the window both become the arg-max token;
    without them in the window all three are drawn."""
    top, other = SEMANTIC_BEGIN + 5, SEMANTIC_BEGIN + 9
    logits = _candidate_logits({top: 0.4, EOS: 0.35, other: 0.25}, 2000)

    repeated = _main_tokens(logits, _window([EOS, other], 2000), "argmax")
    free = _main_tokens(logits, _window([], 2000), "argmax")

    assert set(repeated.tolist()) == {top}
    assert set(free.tolist()) == {top, EOS, other}


def test_the_ras_fallback_draws_at_fixed_parameters_without_the_cutoff():
    """With epsilon 0.1 the first draw is always the top token, which the window holds; the
    fallback then keeps the top 30 candidates by 1.0 / 0.95 / 30 although all but the top one
    lie below the cutoff."""
    top = SEMANTIC_BEGIN
    others = range(SEMANTIC_BEGIN + 1, SEMANTIC_BEGIN + 41)
    weights = torch.linspace(1.0, 2.0, len(others))
    probs = {top: 0.5} | {token: float(0.5 * weight / weights.sum()) for token, weight in zip(others, weights)}
    logits = _candidate_logits(probs, 20000)
    expected = _expected(logits[0], 1.0, RAS_TOP_K, 0.95, 0.0)

    tokens = _main_tokens(logits, _window([top], 20000), "ras", epsilon_cutoff=0.1)

    assert int((expected > 0).sum()) == RAS_TOP_K
    _assert_follows(torch.searchsorted(CANDIDATES, tokens), expected)


def _codes(probs, rows, top_k=-1, top_p=1.0, seed=0):
    logits = torch.full((rows, HEAD_WIDTH), -30.0)
    for code, prob in probs.items():
        logits[:, code] = math.log(prob)
    sampler = ReferenceFrameSampler(fallback="argmax", num_codebooks=10, width=HEAD_WIDTH)
    noise = torch.rand((rows, sampler.num_draws, sampler.width), generator=torch.Generator().manual_seed(seed))
    return logits, sampler.code(logits, _rows(1.0, rows), _rows(top_k, rows, torch.int64), _rows(top_p, rows), noise, 3)


def test_untruncated_residual_draws_stay_inside_the_codec_codebook():
    """Codes past the codebook hold half of the mass; the draws follow the legal codes'
    probabilities renormalised."""
    probs = {code: 0.04 for code in range(10)} | {DAC_RESIDUAL_CODEBOOK_SIZE + code: 0.06 for code in range(10)}
    logits, codes = _codes(probs, 20000)

    assert int(codes.max()) < DAC_RESIDUAL_CODEBOOK_SIZE
    _assert_follows(codes, _expected(logits[0], 1.0, -1, 1.0, 0.0, legal_ids=DAC_RESIDUAL_CODEBOOK_SIZE))


@pytest.mark.parametrize(
    ("probs", "top_k", "top_p", "expected"),
    [
        pytest.param({3000: 0.4, 5: 0.3, 7: 0.2, 9: 0.1}, 2, 1.0, {5}, id="top-k"),
        pytest.param({3000: 0.4, 5: 0.3, 7: 0.2, 9: 0.1}, -1, 0.95, {5, 7}, id="top-p"),
        pytest.param({3000: 0.6, 2500: 0.3, 5: 0.06, 7: 0.04}, 2, 1.0, {5}, id="all-illegal"),
        pytest.param({3000: 0.6, 2500: 0.3, 5: 0.06, 7: 0.04}, -1, 0.5, {5}, id="all-illegal-top-p"),
    ],
)
def test_residual_rules_run_on_the_whole_row_before_illegal_codes_drop(probs, top_k, top_p, expected):
    """Illegal codes count toward top-k and top-p; when no survivor is legal, the most probable
    legal code is kept."""
    _, codes = _codes(probs, 2000, top_k=top_k, top_p=top_p)
    assert set(codes.tolist()) == expected


def test_a_rows_draw_depends_only_on_its_own_noise():
    """A seeded request's frame noise is keyed by its seed and frame, so the same row noise
    must give the same frame in any batch row next to any other request."""
    generator = torch.Generator().manual_seed(12)
    logits = torch.randn(4, CANDIDATES.numel(), generator=generator) * 2.0
    sampler = ReferenceFrameSampler(fallback="ras", num_codebooks=10, width=CANDIDATES.numel())
    noise = torch.rand((4, sampler.num_draws, sampler.width), generator=generator)
    window = _window([int(CANDIDATES[logits[0].argmax()])], 4)
    params = (_rows(1.0, 4), _rows(-1, 4, torch.int64), _rows(1.0, 4), _rows(1e-3, 4))
    order = torch.tensor([2, 0, 3, 1])

    tokens, _ = sampler.main_tokens(logits, CANDIDATES, window, *params, noise, SEMANTIC_BEGIN, CODEBOOK_SIZE)
    shuffled, _ = sampler.main_tokens(
        logits[order], CANDIDATES, window[order], *params, noise[order], SEMANTIC_BEGIN, CODEBOOK_SIZE
    )

    assert torch.equal(shuffled, tokens[order])


def test_the_switch_and_fallback_are_validated_and_off_by_default():
    assert (
        reference_sampling_enabled(types.SimpleNamespace()),
        reference_sampling_fallback(types.SimpleNamespace()),
    ) == (False, "argmax")
    config = types.SimpleNamespace(fish_omni_reference_sampling=True, fish_omni_reference_sampling_fallback="ras")
    assert (reference_sampling_enabled(config), reference_sampling_fallback(config)) == (True, "ras")
    with pytest.raises(ValueError, match="fish_omni_reference_sampling="):
        reference_sampling_enabled(types.SimpleNamespace(fish_omni_reference_sampling="yes"))
    with pytest.raises(ValueError, match="fish_omni_reference_sampling_fallback="):
        reference_sampling_fallback(types.SimpleNamespace(fish_omni_reference_sampling_fallback="greedy"))


@pytest.mark.parametrize(
    ("value", "reference_sampling", "resolved"),
    [
        pytest.param(None, True, 1e-3, id="reference-unset"),
        pytest.param(0, True, 0.0, id="reference-zero"),
        pytest.param(0.02, True, 0.02, id="reference-set"),
        pytest.param(None, False, 0.0, id="default-unset"),
        pytest.param(0.0, False, 0.0, id="default-zero"),
    ],
)
def test_an_unset_epsilon_cutoff_defaults_by_sampler(value, reference_sampling, resolved):
    assert resolve_epsilon_cutoff(value, reference_sampling) == resolved


@pytest.mark.parametrize("value", [float("nan"), -0.001, 1.0, 1.5, True, "0.1"])
def test_an_epsilon_cutoff_outside_zero_to_one_is_refused(value):
    with pytest.raises(ValueError, match=r"epsilon_cutoff must be in \[0, 1\)"):
        resolve_epsilon_cutoff(value, True)


def test_a_non_zero_epsilon_cutoff_without_the_switch_names_it():
    with pytest.raises(ValueError, match="hf_overrides fish_omni_reference_sampling: true"):
        resolve_epsilon_cutoff(0.002, False)
