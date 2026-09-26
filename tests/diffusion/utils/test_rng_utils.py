# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from vllm_omni.diffusion.utils.rng_utils import seeded_global_rng

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _draw() -> torch.Tensor:
    """Sample off the global RNG, the way transformers' generate() does."""
    return torch.multinomial(torch.ones(8), num_samples=16, replacement=True)


def test_same_seed_reproduces_global_draws():
    with seeded_global_rng(torch.Generator().manual_seed(1234)):
        first = _draw()
    with seeded_global_rng(torch.Generator().manual_seed(1234)):
        second = _draw()

    assert torch.equal(first, second)


def test_different_seeds_produce_different_global_draws():
    with seeded_global_rng(torch.Generator().manual_seed(1234)):
        first = _draw()
    with seeded_global_rng(torch.Generator().manual_seed(4321)):
        second = _draw()

    assert not torch.equal(first, second)


def test_generator_stream_is_not_consumed():
    generator = torch.Generator().manual_seed(1234)
    expected = torch.randn(4, generator=generator)

    generator = torch.Generator().manual_seed(1234)
    with seeded_global_rng(generator):
        _draw()
    latents = torch.randn(4, generator=generator)

    assert torch.equal(expected, latents)


def test_global_rng_is_restored_on_exit():
    torch.manual_seed(99)
    expected = _draw()

    torch.manual_seed(99)
    with seeded_global_rng(torch.Generator().manual_seed(1234)):
        _draw()
    restored = _draw()

    assert torch.equal(expected, restored)


def test_missing_generator_leaves_global_rng_alone():
    torch.manual_seed(99)
    expected = _draw()

    torch.manual_seed(99)
    with seeded_global_rng(None):
        without_generator = _draw()
    torch.manual_seed(99)
    with seeded_global_rng([]):
        without_generators = _draw()

    assert torch.equal(expected, without_generator)
    assert torch.equal(expected, without_generators)


def test_generator_list_seeds_from_first_entry():
    with seeded_global_rng([torch.Generator().manual_seed(1234), torch.Generator().manual_seed(4321)]):
        from_list = _draw()
    with seeded_global_rng(torch.Generator().manual_seed(1234)):
        from_single = _draw()

    assert torch.equal(from_list, from_single)
