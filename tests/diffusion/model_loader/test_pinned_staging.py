# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from vllm_omni.diffusion import envs as diffusion_envs
from vllm_omni.diffusion.model_loader import pinned_staging
from vllm_omni.diffusion.model_loader.pinned_staging import (
    PinnedStagingState,
    pinned_staging_weights_iterator,
    release_pinned_staging_cache,
)
from vllm_omni.diffusion.models.interface import (
    consumes_borrowed_weight_tensors,
    consumes_borrowed_weights_synchronously,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _cpu_slab(nbytes: int) -> torch.Tensor:
    return torch.empty(nbytes, dtype=torch.uint8)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("0", False), ("1", True), ("true", True), ("YES", True), ("off", False)],
)
def test_pinned_staging_environment_flag(monkeypatch, raw, expected):
    monkeypatch.setenv("VLLM_OMNI_ENABLE_PINNED_WEIGHT_STAGING", raw)
    assert diffusion_envs.VLLM_OMNI_ENABLE_PINNED_WEIGHT_STAGING is expected


def test_host_cache_cleanup_is_best_effort(monkeypatch):
    monkeypatch.delattr(pinned_staging.torch.accelerator, "empty_host_cache", raising=False)
    fallback_calls: list[None] = []
    monkeypatch.setattr(
        pinned_staging.torch._C,
        "_host_emptyCache",
        lambda: fallback_calls.append(None),
        raising=False,
    )
    release_pinned_staging_cache()
    assert fallback_calls == [None]

    def _fail_cleanup():
        raise RuntimeError("injected cleanup failure")

    monkeypatch.setattr(
        pinned_staging.torch.accelerator,
        "empty_host_cache",
        _fail_cleanup,
        raising=False,
    )
    monkeypatch.delattr(pinned_staging.torch._C, "_host_emptyCache", raising=False)
    release_pinned_staging_cache()


def test_staging_preserves_order_values_and_reuses_fixed_storage(monkeypatch):
    allocations: list[int] = []

    def _allocate(nbytes: int) -> torch.Tensor:
        allocations.append(nbytes)
        return _cpu_slab(nbytes)

    monkeypatch.setattr(pinned_staging, "_alloc_pinned", _allocate)
    source = iter(
        [
            ("first", torch.arange(8, dtype=torch.float32)),
            ("second", torch.arange(8, dtype=torch.float32) + 10),
        ]
    )
    state = PinnedStagingState()
    staged = pinned_staging_weights_iterator(
        source,
        capacity_bytes=64,
        min_bytes=1,
        state=state,
    )

    first_name, first = next(staged)
    first_ptr = first.data_ptr()
    assert first_name == "first"
    assert first.shape == (8,)
    assert first.dtype is torch.float32
    assert first.stride() == (1,)
    assert torch.equal(first, torch.arange(8, dtype=torch.float32))

    second_name, second = next(staged)
    assert second_name == "second"
    assert second.data_ptr() == first_ptr
    assert torch.equal(second, torch.arange(8, dtype=torch.float32) + 10)
    with pytest.raises(StopIteration):
        next(staged)
    assert allocations == [64]
    assert state.allocated


def test_staging_preserves_noncanonical_singleton_strides(monkeypatch):
    monkeypatch.setattr(pinned_staging, "_alloc_pinned", _cpu_slab)
    source = torch.empty_strided((1, 65536), (123456, 1), dtype=torch.float32)
    source.copy_(torch.arange(65536, dtype=torch.float32).view(1, -1))

    output = list(
        pinned_staging_weights_iterator(
            iter([("weight", source)]),
            capacity_bytes=1 << 20,
            min_bytes=1,
        )
    )

    staged = output[0][1]
    assert staged.shape == source.shape
    assert staged.stride() == source.stride()
    assert torch.equal(staged, source)


def test_unsupported_tensors_pass_through_by_identity(monkeypatch):
    monkeypatch.setattr(pinned_staging, "_alloc_pinned", _cpu_slab)
    small = torch.ones(1)
    oversized = torch.ones(32)
    non_contiguous = torch.arange(8).view(2, 4).t()
    requires_grad = torch.ones(4, requires_grad=True)

    output = list(
        pinned_staging_weights_iterator(
            iter(
                [
                    ("small", small),
                    ("oversized", oversized),
                    ("non_contiguous", non_contiguous),
                    ("requires_grad", requires_grad),
                ]
            ),
            capacity_bytes=64,
            min_bytes=8,
        )
    )

    assert [name for name, _ in output] == [
        "small",
        "oversized",
        "non_contiguous",
        "requires_grad",
    ]
    assert output[0][1] is small
    assert output[1][1] is oversized
    assert output[2][1] is non_contiguous
    assert output[3][1] is requires_grad


def test_empty_and_small_only_streams_do_not_allocate(monkeypatch):
    def _unexpected_allocation(_nbytes: int) -> torch.Tensor:
        raise AssertionError("no slab should be allocated")

    monkeypatch.setattr(pinned_staging, "_alloc_pinned", _unexpected_allocation)
    state = PinnedStagingState()
    assert (
        list(
            pinned_staging_weights_iterator(
                iter(()),
                capacity_bytes=64,
                min_bytes=8,
                state=state,
            )
        )
        == []
    )
    small = torch.ones(1)
    assert list(
        pinned_staging_weights_iterator(
            iter([("small", small)]),
            capacity_bytes=64,
            min_bytes=8,
            state=state,
        )
    ) == [("small", small)]
    assert not state.allocated


def test_allocation_failure_falls_back_without_losing_current_item(monkeypatch):
    def _fail(_nbytes: int) -> torch.Tensor:
        raise RuntimeError("memlock exhausted")

    monkeypatch.setattr(pinned_staging, "_alloc_pinned", _fail)
    first = torch.arange(8, dtype=torch.float32)
    second = torch.arange(8, dtype=torch.float32) + 1
    state = PinnedStagingState()
    output = list(
        pinned_staging_weights_iterator(
            iter([("first", first), ("second", second)]),
            capacity_bytes=64,
            min_bytes=1,
            state=state,
        )
    )

    assert output[0] == ("first", first)
    assert output[1] == ("second", second)
    assert output[0][1] is first
    assert output[1][1] is second
    assert not state.allocated


def test_staging_copy_error_propagates(monkeypatch):
    monkeypatch.setattr(pinned_staging, "_alloc_pinned", lambda _nbytes: torch.empty(1, dtype=torch.uint8))
    staged = pinned_staging_weights_iterator(
        iter([("weight", torch.arange(8, dtype=torch.float32))]),
        capacity_bytes=64,
        min_bytes=1,
    )

    with pytest.raises(RuntimeError):
        next(staged)


def test_upstream_error_propagates(monkeypatch):
    monkeypatch.setattr(pinned_staging, "_alloc_pinned", _cpu_slab)

    def _source():
        yield "weight", torch.ones(1)
        raise ValueError("checkpoint corrupt")

    staged = pinned_staging_weights_iterator(_source(), capacity_bytes=64, min_bytes=8)
    assert next(staged)[0] == "weight"
    with pytest.raises(ValueError, match="checkpoint corrupt"):
        next(staged)


@pytest.mark.parametrize(
    ("capacity_bytes", "min_bytes", "message"),
    [(0, 1, "capacity_bytes"), (1, -1, "min_bytes")],
)
def test_invalid_limits_rejected(capacity_bytes, min_bytes, message):
    staged = pinned_staging_weights_iterator(
        iter(()),
        capacity_bytes=capacity_bytes,
        min_bytes=min_bytes,
    )
    with pytest.raises(ValueError, match=message):
        next(staged)


def test_borrowed_weight_marker_does_not_leak_to_override():
    class SynchronousConsumer:
        @consumes_borrowed_weight_tensors
        def load_weights(self, weights):
            return list(weights)

    class RetainingOverride(SynchronousConsumer):
        def load_weights(self, weights):
            self.retained = list(weights)

    assert consumes_borrowed_weights_synchronously(SynchronousConsumer())
    assert not consumes_borrowed_weights_synchronously(RetainingOverride())
