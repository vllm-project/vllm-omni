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
    from vllm.v1.sample.ops import topk_topp_sampler as sampler_ops

    import vllm_omni.patch  # noqa: F401

    original = sampler_ops.random_sample.__wrapped__
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
            expected = original(probs.clone(), expected_gens, use_fp64_gumbel)
            expected_default = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(actual_default)
            actual = sampler_ops.random_sample(probs.clone(), actual_gens, use_fp64_gumbel)
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
