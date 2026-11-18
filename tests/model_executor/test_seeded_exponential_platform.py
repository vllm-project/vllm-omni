# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from unittest.mock import Mock

import pytest
import torch

from tests.helpers.mark import hardware_test
from vllm_omni.utils import seeded_exponential

pytestmark = [pytest.mark.core_model]


@pytest.mark.cpu
@pytest.mark.parametrize("is_nvidia", [False, True], ids=["rocm", "cuda"])
def test_cuda_tensor_requires_nvidia_distribution_contract(monkeypatch, is_nvidia: bool) -> None:
    monkeypatch.setattr(seeded_exponential.current_platform, "is_cuda", lambda: is_nvidia)
    # HIP tensors expose the same Torch device predicate as NVIDIA tensors.
    q = Mock(spec=torch.Tensor)
    q.is_cuda = True
    q.dtype = torch.float32
    q.dim.return_value = 2
    q.is_contiguous.return_value = True
    generators = {0: object(), 1: object()}

    assert seeded_exponential.batched_seeded_exponential_supported(q, generators) is is_nvidia


@pytest.mark.cpu
def test_shared_generator_keeps_ordered_torch_draws(monkeypatch) -> None:
    monkeypatch.setattr(seeded_exponential.current_platform, "is_cuda", lambda: True)
    q = Mock(spec=torch.Tensor)
    q.is_cuda = True
    q.dtype = torch.float32
    q.dim.return_value = 2
    q.is_contiguous.return_value = True
    shared = object()

    assert not seeded_exponential.batched_seeded_exponential_supported(q, {0: shared, 1: shared})


@hardware_test(res={"cuda": "L4", "rocm": "MI325"}, num_cards=1)
@pytest.mark.parametrize("seeded_rows", [(0, 1, 2), (0, 2)], ids=["all_seeded", "mixed"])
@pytest.mark.parametrize("use_fp64_gumbel", [False, True])
def test_torch_fallback_preserves_samples_and_generator_state(monkeypatch, seeded_rows, use_fp64_gumbel) -> None:
    def reference(probs, generators, fp64):
        noise = torch.empty(probs.shape, dtype=torch.float64 if fp64 else probs.dtype, device=probs.device)
        if len(generators) != len(probs):
            noise.exponential_()
        for row, generator in generators.items():
            noise[row].exponential_(generator=generator)
        return (probs / noise).argmax(dim=-1)

    monkeypatch.setattr(seeded_exponential.current_platform, "is_cuda", lambda: False)

    def reject_kernel(*args, **kwargs):
        raise AssertionError("the CUDA distribution kernel must not run on the fallback path")

    monkeypatch.setattr(seeded_exponential, "fill_exponential_rows", reject_kernel)
    probs = torch.arange(1, 258, device="cuda", dtype=torch.float32).expand(3, -1).contiguous()
    probs /= probs.sum(dim=-1, keepdim=True)
    expected_gens = {row: torch.Generator(device="cuda").manual_seed(42 + row) for row in seeded_rows}
    actual_gens = {row: torch.Generator(device="cuda").manual_seed(42 + row) for row in seeded_rows}
    expected_default = torch.cuda.get_rng_state()
    actual_default = expected_default.clone()
    try:
        for _ in range(5):
            torch.cuda.set_rng_state(expected_default)
            # FP32 sampling divides probabilities in place. Give both paths
            # the same untouched input on every step.
            expected = reference(probs.clone(), expected_gens, use_fp64_gumbel)
            expected_default = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(actual_default)
            actual = seeded_exponential.categorical_sample(probs.clone(), actual_gens, use_fp64_gumbel)
            actual_default = torch.cuda.get_rng_state()
            assert torch.equal(actual, expected)
            assert torch.equal(actual_default, expected_default)
            for row in seeded_rows:
                assert torch.equal(actual_gens[row].get_state(), expected_gens[row].get_state())
    finally:
        torch.cuda.set_rng_state(expected_default)


@hardware_test(res={"cuda": "L4", "rocm": "MI325"}, num_cards=1)
@pytest.mark.parametrize("rows", [None, [0, 2, 4]], ids=["grouped_codebooks", "selected_rows"])
def test_direct_fill_fallback_preserves_ordered_draws(monkeypatch, rows) -> None:
    monkeypatch.setattr(seeded_exponential.current_platform, "is_cuda", lambda: False)

    def reject_kernel(*args, **kwargs):
        raise AssertionError("the NVIDIA kernel must not run for direct fallback callers")

    monkeypatch.setattr(seeded_exponential, "torch_exponential_policy", reject_kernel)
    expected = torch.ones(6, 257, device="cuda")
    actual = expected.clone()
    expected_gen = torch.Generator(device="cuda").manual_seed(71)
    actual_gen = torch.Generator(device="cuda").manual_seed(71)
    # A shared seeded generator and an unseeded request make loop order and
    # default-state advancement observable, including grouped codebooks.
    expected_gens = [expected_gen, None, expected_gen]
    actual_gens = [actual_gen, None, actual_gen]
    original_default = torch.cuda.get_rng_state()
    expected_default = original_default.clone()
    actual_default = original_default.clone()
    try:
        for _ in range(5):
            torch.cuda.set_rng_state(expected_default)
            targets = expected.reshape(3, -1) if rows is None else [expected[row] for row in rows]
            for target, generator in zip(targets, expected_gens):
                target.exponential_(generator=generator)
            expected_default = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(actual_default)
            assert seeded_exponential.fill_exponential_rows(actual, actual_gens, rows) is actual
            actual_default = torch.cuda.get_rng_state()
            assert torch.equal(actual, expected)
            assert torch.equal(actual_gen.get_state(), expected_gen.get_state())
            assert torch.equal(actual_default, expected_default)
    finally:
        torch.cuda.set_rng_state(original_default)


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("vocab", [6562, 300_000])
def test_batched_steps_match_torch_noise_at_each_request_offset(vocab):
    device = torch.device("cuda")
    generators = tuple(torch.Generator(device=device).manual_seed(seed) for seed in (42, 42, 2**63 + 5))
    offsets = (0, 12, 28)
    for generator, offset in zip(generators, offsets, strict=True):
        generator.set_offset(offset)
    references = tuple(generator.clone_state() for generator in generators)
    actual = torch.empty((3, len(generators), vocab), device=device, dtype=torch.float32)

    increment = seeded_exponential.fill_exponential_steps(actual, generators)
    assert tuple(generator.get_offset() for generator in generators) == offsets
    expected = torch.empty_like(actual)
    for frame in range(actual.shape[0]):
        for row, reference in enumerate(references):
            expected[frame, row].exponential_(generator=reference)
            assert reference.get_offset() == offsets[row] + (frame + 1) * increment
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.cpu
@pytest.mark.parametrize("fp64", [False, True])
@pytest.mark.parametrize("seeding", ["all", "mixed", "shared", "unseeded"])
def test_model_categorical_draw_preserves_ordered_generators_and_input_contract(fp64, seeding):
    # Exercise the removed native primitive's actual Torch RNG contract,
    # including a shared generator in non-sorted insertion order.
    probs = torch.arange(1, 22, dtype=torch.float32).reshape(3, 7)
    shared = torch.Generator().manual_seed(43)
    generators = {2: torch.Generator().manual_seed(42), 0: shared, 1: shared}
    if seeding == "mixed":
        generators.pop(1)
    elif seeding == "unseeded":
        generators = {}
    elif seeding == "all":
        generators[1] = torch.Generator().manual_seed(44)
    default_before = torch.get_rng_state()
    states = {id(g): (g, g.get_state()) for g in generators.values()}
    try:
        noise = torch.empty(probs.shape, dtype=torch.float64 if fp64 else probs.dtype)
        if len(generators) != len(probs):
            noise.exponential_()
        for row, generator in generators.items():
            noise[row].exponential_(generator=generator)
        expected = (probs / noise).argmax(-1)
        expected_default = torch.get_rng_state()
        expected_states = {key: g.get_state() for key, (g, _) in states.items()}
        torch.set_rng_state(default_before)
        for generator, state in states.values():
            generator.set_state(state)
        owned = probs.clone()
        actual = seeded_exponential.categorical_sample(owned, generators, fp64)
        assert actual.shape == (3,) and actual.dtype == torch.int64
        assert torch.equal(actual, expected)
        assert torch.equal(torch.get_rng_state(), expected_default)
        for key, (generator, _) in states.items():
            assert torch.equal(generator.get_state(), expected_states[key])
        if fp64:
            assert torch.equal(owned, probs)
        else:
            torch.testing.assert_close(owned, probs / noise, rtol=0, atol=0)
    finally:
        torch.set_rng_state(default_before)
