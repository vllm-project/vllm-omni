# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Exact projection reuse without model-specific schedules or artifacts."""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.cache.exact_projection_cache import ExactProjectionCache

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.fixture(autouse=True)
def tp1(monkeypatch):
    import vllm.distributed

    monkeypatch.setattr(vllm.distributed, "get_tensor_model_parallel_world_size", lambda: 1)


@pytest.fixture(autouse=True)
def deterministic_weights():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        yield


@pytest.mark.parametrize("batch,in_features,out_features", [(2, 4, 8), (1, 8, 48), (4, 6, 18)])
def test_runtime_reuses_exact_outputs_and_tracks_changed_weights(mocker, batch, in_features, out_features):
    cache = ExactProjectionCache()
    linear = torch.nn.Linear(in_features, out_features)
    x = torch.randn(batch, in_features)
    compute = mocker.Mock(side_effect=lambda: linear(x))
    with torch.no_grad():
        cache.prepare(x)
        first = cache.project("block", linear, x, compute)
        first_copy = first.clone()
        first.zero_()  # The caller/provider cannot corrupt the owned cache.
        assert torch.equal(cache.project("block", linear, x, compute), first_copy)
        assert compute.call_count == 1
        linear.weight.add_(0.25)
        assert torch.equal(cache.project("block", linear, x, compute), linear(x))
        assert compute.call_count == 2
        other = x.clone()
        other[0, 0] += 1
        cache.prepare(other)
        cache.project("block", linear, other, lambda: linear(other))
        cache.prepare(x)
        cache.project("block", linear, x, compute)
        assert compute.call_count == 2
    assert cache.hits == 2


def test_runtime_budget_retains_reusable_entries_without_schedule_thrashing(mocker):
    cache = ExactProjectionCache(max_bytes=32)
    linear = torch.nn.Linear(2, 4)
    x = torch.ones(2, 2)
    compute = mocker.Mock(side_effect=lambda: linear(x))
    with torch.no_grad():
        cache.prepare(x)
        for name in ("first", "second", "first"):
            cache.project(name, linear, x, compute)
            assert cache._bytes <= 32
        assert compute.call_count == 2
        assert cache.hits == 1
        cache.clear()
        assert cache._bytes == 0 and not cache._entries


def test_dynamic_lora_buffers_and_suspend_mask_invalidate_entries(mocker):
    cache = ExactProjectionCache()
    linear = torch.nn.Linear(2, 2)
    linear.lora_a_stacked = (torch.ones(1, 2),)
    linear.lora_b_stacked = (torch.ones(2, 1),)
    linear._diffusion_lora_active_slices = (True,)
    x = torch.ones(1, 2)

    def compute():
        result = linear(x)
        if linear._diffusion_lora_active_slices[0]:
            result = result + x @ linear.lora_a_stacked[0].T @ linear.lora_b_stacked[0].T
        return result

    counted = mocker.Mock(side_effect=compute)
    with torch.no_grad():
        cache.prepare(x)
        cache.project("a", linear, x, counted)
        cache.project("a", linear, x, counted)
        assert counted.call_count == 1
        linear.lora_b_stacked[0].add_(1)
        assert torch.equal(cache.project("a", linear, x, counted), compute())
        linear._diffusion_lora_active_slices = (False,)
        assert torch.equal(cache.project("a", linear, x, counted), compute())
        assert counted.call_count == 3


@pytest.mark.parametrize("scope", ["local", "global", "child"])
@pytest.mark.parametrize("kind", ["pre", "post"])
def test_linear_hooks_run_even_after_cache_warmup(scope, kind):
    cache = ExactProjectionCache()
    linear = torch.nn.Linear(2, 2)
    projection = torch.nn.Sequential(linear) if scope == "child" else linear
    x = torch.ones(1, 2)
    with torch.no_grad():
        cache.prepare(x)
        expected = projection(x)
        cache.project("a", projection, x, lambda: projection(x))
        calls = []

        def pre_hook(module, args):
            if module is linear:
                calls.append(module)
                return (args[0] + 1,)

        def post_hook(module, args, output):
            if module is linear:
                calls.append(module)
                return output + 1

        if kind == "pre":
            expected = linear(x + 1)
            register = (
                torch.nn.modules.module.register_module_forward_pre_hook
                if scope == "global"
                else linear.register_forward_pre_hook
            )
            hook = register(pre_hook)
        else:
            expected = expected + 1
            register = (
                torch.nn.modules.module.register_module_forward_hook
                if scope == "global"
                else linear.register_forward_hook
            )
            hook = register(post_hook)
        try:
            for _ in range(2):
                assert torch.equal(cache.project("a", projection, x, lambda: projection(x)), expected)
            assert len(calls) == 2
        finally:
            hook.remove()
        assert torch.equal(cache.project("a", projection, x, lambda: projection(x)), projection(x))


@pytest.mark.parametrize("autocast_at_prepare", [False, True])
def test_numerical_settings_follow_projection_context(mocker, autocast_at_prepare):
    cache = ExactProjectionCache()
    linear = torch.nn.Linear(4, 4)
    x = torch.randn(2, 4)
    compute = mocker.Mock(side_effect=lambda: linear(x))
    with torch.no_grad():
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast_at_prepare):
            cache.prepare(x)
        for enabled in (False, True, False, True):
            with torch.autocast("cpu", dtype=torch.bfloat16, enabled=enabled):
                actual = cache.project("a", linear, x, compute)
                expected = linear(x)
                assert actual.dtype == expected.dtype
                assert torch.equal(actual, expected)
    assert compute.call_count == 2


@pytest.mark.parametrize(
    "setting",
    ["allow_tf32", "allow_bf16_reduced_precision_reduction", "allow_fp16_reduced_precision_reduction"],
)
def test_matmul_settings_invalidate_prepared_projection(monkeypatch, mocker, setting):
    cache = ExactProjectionCache()
    linear = torch.nn.Linear(2, 2)
    x = torch.ones(1, 2)
    compute = mocker.Mock(side_effect=lambda: linear(x))
    with torch.no_grad():
        cache.prepare(x)
        cache.project("a", linear, x, compute)
        monkeypatch.setattr(torch.backends.cuda.matmul, setting, not getattr(torch.backends.cuda.matmul, setting))
        assert torch.equal(cache.project("a", linear, x, compute), linear(x))
    assert compute.call_count == 2


def test_disabled_and_grad_paths_keep_original_computation(monkeypatch, mocker):
    import vllm_omni.diffusion.cache.exact_projection_cache as module

    monkeypatch.setattr(module, "tensor_digest", mocker.Mock(side_effect=AssertionError("must not hash")))
    linear = torch.nn.Linear(2, 2)
    x = torch.ones(1, 2, requires_grad=True)
    for cache in (ExactProjectionCache(max_bytes=0), ExactProjectionCache()):
        cache.prepare(x)
        cache.project("a", linear, x, lambda: linear(x)).sum().backward()
        assert x.grad is not None
    with torch.no_grad():
        cache = ExactProjectionCache(max_bytes=0)
        cache.prepare(x)
        assert torch.equal(cache.project("a", linear, x, lambda: linear(x)), linear(x))


def test_tp_rank_miss_forces_every_rank_to_recompute(monkeypatch, mocker):
    import vllm.distributed

    cache = ExactProjectionCache()
    linear = torch.nn.Linear(2, 2)
    x = torch.ones(1, 2)
    compute = mocker.Mock(side_effect=lambda: linear(x))
    with torch.no_grad():
        cache.prepare(x)
        cache.project("a", linear, x, compute)
        votes = mocker.Mock(side_effect=[torch.tensor([1]), torch.tensor([2])])
        monkeypatch.setattr(vllm.distributed, "get_tensor_model_parallel_world_size", lambda: 2)
        monkeypatch.setattr(vllm.distributed, "get_tp_group", lambda: SimpleNamespace(all_reduce=votes))
        cache.project("a", linear, x, compute)
        assert compute.call_count == 2
        cache.project("a", linear, x, compute)
        assert compute.call_count == 2
        assert votes.call_count == 2


def test_compiled_execution_bypasses_runtime_and_precomputed_results(monkeypatch, mocker):
    import vllm_omni.diffusion.cache.exact_projection_cache as module

    cache = ExactProjectionCache()
    linear = torch.nn.Linear(2, 2)
    x = torch.ones(1, 2)
    compute = mocker.Mock(side_effect=lambda: linear(x))
    with torch.no_grad():
        cache.prepare(x)
        cache.project("modulation", linear, x, compute)
        monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
        monkeypatch.setattr(module, "tensor_digest", mocker.Mock(side_effect=AssertionError("must not hash")))
        seeded = mocker.patch.object(cache, "_lookup_precomputed", side_effect=AssertionError("must not seed"))
        # An input prepared before compilation must also bypass the warm entry.
        assert torch.equal(cache.project("modulation", linear, x, compute), linear(x))
        cache.prepare(x)
        assert torch.equal(cache.project("modulation", linear, x, compute), linear(x))
        assert compute.call_count == 3
        seeded.assert_not_called()


@pytest.mark.parametrize("world_size", [1, 2])
def test_precomputed_extension_is_used_only_at_tp1(monkeypatch, mocker, world_size):
    import vllm.distributed

    cache = ExactProjectionCache()
    linear = torch.nn.Linear(2, 2)
    x = torch.ones(1, 2)
    monkeypatch.setattr(vllm.distributed, "get_tensor_model_parallel_world_size", lambda: world_size)
    votes = mocker.Mock(side_effect=[torch.tensor([0]), torch.tensor([world_size])])
    monkeypatch.setattr(vllm.distributed, "get_tp_group", lambda: SimpleNamespace(all_reduce=votes))
    with torch.no_grad():
        expected = linear(x)
        seeded = mocker.patch.object(cache, "_lookup_precomputed", return_value=expected)
        compute = mocker.Mock(side_effect=lambda: linear(x))
        cache.prepare(x)
        for _ in range(2):
            assert torch.equal(cache.project("conditioning.scale_shift", linear, x, compute), expected)
        assert seeded.call_count == (1 if world_size == 1 else 0)
        assert compute.call_count == (0 if world_size == 1 else 1)
        assert cache.hits == 1


def test_inference_weights_without_versions_use_safe_runtime_path(mocker):
    with torch.inference_mode():
        linear = torch.nn.Linear(2, 2)
        x = torch.ones(1, 2)
        cache = ExactProjectionCache()
        compute = mocker.Mock(side_effect=lambda: linear(x))
        cache.prepare(x)
        cache.project("a", linear, x, compute)
        linear.weight.add_(1)
        assert torch.equal(cache.project("a", linear, x, compute), linear(x))
        assert compute.call_count == 2 and not cache._entries
