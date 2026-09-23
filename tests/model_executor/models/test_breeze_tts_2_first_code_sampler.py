# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""First-code sampler token, RNG, graph-cache and fallback contracts."""

import pytest
import torch

from tests.helpers.mark import hardware_test
from vllm_omni.model_executor.models.breeze_tts_2.depth_decoder import sample_logits
from vllm_omni.model_executor.models.breeze_tts_2.first_code_sampler import BreezeFirstCodeSampler

pytestmark = [pytest.mark.core_model]

PARAMETERS = (0.9, 50, 1.0)
VOCAB = 2052
EOS = 2051


def _logits(count: int, *, device: str, seed: int = 100, vocab: int = VOCAB) -> list[torch.Tensor]:
    source = torch.Generator(device="cpu").manual_seed(seed)
    values = torch.randn((count, vocab), generator=source, dtype=torch.float32) * 3
    if vocab == VOCAB:
        values[:, 2048:EOS] = -torch.inf
    return list(values.to(device).split(1))


def _generators(seeds: list[int], device: str) -> list[torch.Generator]:
    return [torch.Generator(device=device).manual_seed(seed) for seed in seeds]


def _reference(rows, parameters, generators) -> list[torch.Tensor]:
    return [sample_logits(row, *parameters, generator) for row, generator in zip(rows, generators, strict=True)]


def _assert_outputs(actual, expected, rows) -> None:
    assert isinstance(actual, list) and len(actual) == len(expected)
    for result, reference, row in zip(actual, expected, rows, strict=True):
        assert result.shape == (1,) and result.dtype == torch.long and result.device == row.device
        torch.testing.assert_close(result, reference, atol=0, rtol=0)


def _assert_states(actual, expected) -> None:
    for left, right in zip(actual, expected, strict=True):
        torch.testing.assert_close(left.get_state(), right.get_state(), atol=0, rtol=0)


@pytest.mark.cpu
def test_empty_and_cpu_sampling_preserve_eager_contract() -> None:
    sampler = BreezeFirstCodeSampler()
    assert sampler.sample([], PARAMETERS, []) == []
    assert not sampler._graphs
    rows = _logits(3, device="cpu")
    before = [row.clone() for row in rows]
    actual_generators = _generators([42, 2**63 + 19, 2**64 - 23], "cpu")
    expected_generators = _generators([42, 2**63 + 19, 2**64 - 23], "cpu")
    for parameters in (PARAMETERS, (0.7, 20, 0.8), (1.2, 0, 0.95)):
        expected = _reference(rows, parameters, expected_generators)
        _assert_outputs(sampler.sample(rows, parameters, actual_generators), expected, rows)
        _assert_states(actual_generators, expected_generators)
    for row, original in zip(rows, before, strict=True):
        torch.testing.assert_close(row, original, atol=0, rtol=0)
    assert not sampler._graphs


@pytest.mark.cpu
def test_greedy_never_samples_or_advances_rng(monkeypatch) -> None:
    sampler = BreezeFirstCodeSampler()
    rows = _logits(3, device="cpu")
    generators = _generators([42, 43, 44], "cpu")
    original_states = [generator.get_state().clone() for generator in generators]

    def unexpected_multinomial(*_args, **_kwargs):
        raise AssertionError("Greedy sampling must not call multinomial")

    monkeypatch.setattr(torch, "multinomial", unexpected_multinomial)
    for parameters in ((0.0, 0, 1.0), (0.0, 1, 0.2)):
        _assert_outputs(sampler.sample(rows, parameters, generators), [row.argmax(-1) for row in rows], rows)
    for generator, state in zip(generators, original_states, strict=True):
        torch.testing.assert_close(generator.get_state(), state, atol=0, rtol=0)
    assert not sampler._graphs


@pytest.mark.cpu
@pytest.mark.parametrize(
    "invalid", ["extra_generator", "missing_generator", "empty_rows", "multirow", "mixed_vocab", "mixed_dtype"]
)
def test_invalid_batch_fails_before_advancing_any_generator(invalid: str) -> None:
    sampler = BreezeFirstCodeSampler()
    rows = _logits(2, device="cpu")
    generators = _generators([42, 43], "cpu")
    if invalid == "extra_generator":
        rows = rows[:1]
    elif invalid == "missing_generator":
        generators = generators[:1]
    elif invalid == "empty_rows":
        rows = []
    elif invalid == "multirow":
        rows[1] = rows[1].expand(2, -1)
    elif invalid == "mixed_vocab":
        rows[1] = rows[1][:, :-1]
    else:
        rows[1] = rows[1].double()
    states = [generator.get_state().clone() for generator in generators]
    with pytest.raises(ValueError):
        sampler.sample(rows, PARAMETERS, generators)
    for generator, state in zip(generators, states, strict=True):
        torch.testing.assert_close(generator.get_state(), state, atol=0, rtol=0)
    assert not sampler._graphs


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@torch.inference_mode()
def test_cuda_replay_preserves_tokens_rng_and_owned_outputs(monkeypatch) -> None:
    sampler = BreezeFirstCodeSampler()
    seeds = list(range(42, 77)) + [2**63 + 19, 2**64 - 23, 101]
    actual_generators = _generators(seeds, "cuda")
    expected_generators = _generators(seeds, "cuda")
    # Torch also accepts an unspecified CUDA index. Exercise it alongside the
    # indexed generators created from serving request tensors.
    indexed_device = torch.device("cuda", torch.accelerator.current_device_index())
    for pool in (actual_generators, expected_generators):
        pool[1] = torch.Generator(device=indexed_device).manual_seed(seeds[1])
    orders = (
        [0],
        [2, 0, 1],
        list(range(16)),
        [36, *range(16)],
        [35, *range(30)],
        list(reversed(range(32))),
        [35, 1, 2],
    )
    retained: list[tuple[torch.Tensor, torch.Tensor]] = []
    global_state = torch.cuda.get_rng_state().clone()
    for sweep in range(2):
        if sweep:
            for pool in (actual_generators, expected_generators):
                pool[2] = torch.Generator(device="cuda").manual_seed(2**64 - 101)
        for case, order in enumerate(orders):
            rows = _logits(len(order), device="cuda", seed=1000 + sweep * 10 + case)
            if sweep and case == 3:
                for index, row in enumerate(rows):
                    row.fill_(-torch.inf)
                    row[0, EOS if index % 2 else 2] = 0
            original = [row.clone() for row in rows]
            live_actual = [actual_generators[index] for index in order]
            live_expected = [expected_generators[index] for index in order]
            expected = _reference(rows, PARAMETERS, live_expected)
            actual = sampler.sample(rows, PARAMETERS, live_actual)
            _assert_outputs(actual, expected, rows)
            _assert_states(actual_generators, expected_generators)
            retained.extend((value, reference.clone()) for value, reference in zip(actual, expected, strict=True))
            for row, before in zip(rows, original, strict=True):
                torch.testing.assert_close(row, before, atol=0, rtol=0)
            # Preserve the subsequent 15 independent depth draws, including
            # skipping them for EOS. All inactive/companion generators remain
            # in the full-pool comparison above and below.
            for index, token in enumerate(torch.cat(expected).cpu().tolist()):
                if token == EOS:
                    continue
                left = torch.empty((15, 2051), device="cuda")
                right = torch.empty_like(left)
                for codebook in range(15):
                    left[codebook].exponential_(generator=live_actual[index])
                    right[codebook].exponential_(generator=live_expected[index])
                torch.testing.assert_close(left, right, atol=0, rtol=0)
            _assert_states(actual_generators, expected_generators)
    # B1/3/16/17/31/32 must use exactly buckets 1/4/16/32.
    assert len(sampler._graphs) == 4
    rows = _logits(3, device="cuda", seed=2026)
    expected = _reference(rows, PARAMETERS, expected_generators[:3])

    def unexpected_multinomial(*_args, **_kwargs):
        raise AssertionError("Warmed graph replay must not call Python multinomial")

    with monkeypatch.context() as replay_guard:
        replay_guard.setattr(torch, "multinomial", unexpected_multinomial)
        _assert_outputs(sampler.sample(rows, PARAMETERS, actual_generators[:3]), expected, rows)
        _assert_outputs(
            sampler.sample(rows, (0.0, 1, 0.2), actual_generators[:3]), [row.argmax(-1) for row in rows], rows
        )
    _assert_states(actual_generators, expected_generators)
    assert len(sampler._graphs) == 4
    for value, expected in retained:
        torch.testing.assert_close(value, expected, atol=0, rtol=0)
    torch.testing.assert_close(torch.cuda.get_rng_state(), global_state, atol=0, rtol=0)


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@torch.inference_mode()
def test_cuda_parameters_vocab_and_missing_graph_api(monkeypatch) -> None:
    sampler = BreezeFirstCodeSampler()
    actual_generators = _generators([42, 43, 44], "cuda")
    expected_generators = _generators([42, 43, 44], "cuda")
    cases = ((0.9, 50, 1.0), (0.7, 100, 0.8), (1.2, 0, 0.95), (1.0, 1, 1.0), (1.0, 0, 0.5))
    for index, parameters in enumerate(cases):
        rows = _logits(3, device="cuda", seed=200 + index)
        if index >= 3:
            for row in rows:
                row.fill_(-torch.inf)
                row[0, :4] = 0  # top-k ties and exact top-p=0.5 boundary.
        expected = _reference(rows, parameters, expected_generators)
        _assert_outputs(sampler.sample(rows, parameters, actual_generators), expected, rows)
        _assert_states(actual_generators, expected_generators)
    assert len(sampler._graphs) == len(cases)
    rows = _logits(3, device="cuda", vocab=17)
    expected = _reference(rows, PARAMETERS, expected_generators)
    _assert_outputs(sampler.sample(rows, PARAMETERS, actual_generators), expected, rows)
    _assert_states(actual_generators, expected_generators)
    assert len(sampler._graphs) == len(cases) + 1

    class UnavailableGraph:
        def __init__(self, *_args, **_kwargs):
            raise AssertionError("Unsupported graph API must select the eager fallback")

    fallback = BreezeFirstCodeSampler()
    expected = _reference(rows, PARAMETERS, expected_generators)
    with monkeypatch.context() as api_guard:
        api_guard.setattr(torch.cuda, "CUDAGraph", UnavailableGraph)
        _assert_outputs(fallback.sample(rows, PARAMETERS, actual_generators), expected, rows)
    _assert_states(actual_generators, expected_generators)
    assert not fallback._graphs

    # Validate every row before the first eager/captured draw, including an
    # invalid device that appears only after an otherwise valid CUDA row.
    invalid = [rows[0], rows[1].cpu(), rows[2]]
    with pytest.raises(ValueError):
        sampler.sample(invalid, PARAMETERS, actual_generators)
    _assert_states(actual_generators, expected_generators)
    # A late invalid generator must likewise fail before row zero samples.
    cpu_generator = torch.Generator(device="cpu").manual_seed(77)
    cpu_reference = torch.Generator(device="cpu").manual_seed(77)
    mixed_generators = [actual_generators[0], cpu_generator, actual_generators[2]]
    with pytest.raises(ValueError):
        sampler.sample(rows, PARAMETERS, mixed_generators)
    _assert_states(mixed_generators, [expected_generators[0], cpu_reference, expected_generators[2]])


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@torch.inference_mode()
def test_cuda_shared_generator_uses_sequential_eager_fallback(monkeypatch) -> None:
    sampler = BreezeFirstCodeSampler()
    rows = _logits(3, device="cuda")
    # The duplicate-generator guard must precede lookup of an existing graph.
    sampler.sample(rows, PARAMETERS, _generators([100, 101, 102], "cuda"))
    cached = dict(sampler._graphs)
    assert len(cached) == 1
    actual = torch.Generator(device="cuda").manual_seed(2**64 - 23)
    reference = torch.Generator(device="cuda").manual_seed(2**64 - 23)
    multinomial = torch.multinomial
    calls = []

    def eager_spy(*args, **kwargs):
        calls.append(kwargs.get("generator"))
        return multinomial(*args, **kwargs)

    for order in ([0, 1, 2], [2, 0, 1]):
        ordered = [rows[index] for index in order]
        expected = _reference(ordered, PARAMETERS, [reference] * 3)
        calls.clear()
        with monkeypatch.context() as fallback_guard:
            fallback_guard.setattr(torch, "multinomial", eager_spy)
            _assert_outputs(sampler.sample(ordered, PARAMETERS, [actual] * 3), expected, ordered)
        assert len(calls) == 3 and all(generator is actual for generator in calls)
        _assert_states([actual], [reference])
        assert set(sampler._graphs) == set(cached)
        assert all(sampler._graphs[key] is value for key, value in cached.items())


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@torch.inference_mode()
def test_cuda_cache_limit_falls_back_without_evicting_warmed_graphs(monkeypatch) -> None:
    sampler = BreezeFirstCodeSampler()
    rows = _logits(1, device="cuda")
    actual_generators = _generators([42], "cuda")
    expected_generators = _generators([42], "cuda")
    for top_k in range(1, 33):
        parameters = (0.9, top_k, 1.0)
        expected = _reference(rows, parameters, expected_generators)
        _assert_outputs(sampler.sample(rows, parameters, actual_generators), expected, rows)
        _assert_states(actual_generators, expected_generators)
    cached = dict(sampler._graphs)
    assert len(cached) == 32
    parameters = (0.9, 33, 1.0)
    expected = _reference(rows, parameters, expected_generators)
    multinomial = torch.multinomial
    calls = []

    def eager_spy(*args, **kwargs):
        calls.append(kwargs.get("generator"))
        return multinomial(*args, **kwargs)

    with monkeypatch.context() as fallback_guard:
        fallback_guard.setattr(torch, "multinomial", eager_spy)
        _assert_outputs(sampler.sample(rows, parameters, actual_generators), expected, rows)
    assert len(calls) == 1 and calls[0] is actual_generators[0]
    _assert_states(actual_generators, expected_generators)
    assert set(sampler._graphs) == set(cached)
    assert all(sampler._graphs[key] is value for key, value in cached.items())
    expected = _reference(rows, (0.9, 1, 1.0), expected_generators)

    def unexpected_multinomial(*_args, **_kwargs):
        raise AssertionError("Cache exhaustion must retain existing captured entries")

    with monkeypatch.context() as replay_guard:
        replay_guard.setattr(torch, "multinomial", unexpected_multinomial)
        _assert_outputs(sampler.sample(rows, (0.9, 1, 1.0), actual_generators), expected, rows)
    _assert_states(actual_generators, expected_generators)
