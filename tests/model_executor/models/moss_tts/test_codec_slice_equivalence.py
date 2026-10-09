# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU equivalence tests for the B==1 slice fast path in RingKVCache.complete.

The slice fast path (``slot0`` is not ``None`` and ``B == 1``) takes contiguous
views of the cache instead of gather+writeback.  These tests verify the two
paths produce bit-identical ``keys``, ``values``, ``positions``, full cache
tensor, and ``end_offset`` across multiple slots, frame counts (including ring
wraparound), and multi-step accumulation.
"""

import pytest
import torch

from vllm_omni.model_executor.models.moss_tts.audio_tokenizer_v2 import (
    RingKVCache,
    StreamingExecutionContext,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_cache(capacity: int, device=torch.device("cpu"), dtype=torch.float32) -> RingKVCache:
    return RingKVCache(
        batch_size=4,
        num_heads=2,
        dim_per_head=3,
        capacity=capacity,
        respect_exec_mask=True,
        device=device,
        dtype=dtype,
    )


def _random_kv(
    batch_size: int, num_heads: int, num_frames: int, dim_per_head: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator(device="cpu").manual_seed(seed)
    return (
        torch.randn(batch_size, num_heads, num_frames, dim_per_head, generator=g),
        torch.randn(batch_size, num_heads, num_frames, dim_per_head, generator=g),
    )


def _assert_cache_state_equal(
    cache_slice: RingKVCache,
    cache_gather: RingKVCache,
    slot: int,
) -> None:
    """Assert keys, values, positions, cache, and end_offset match."""
    torch.testing.assert_close(cache_slice.cache, cache_gather.cache)
    torch.testing.assert_close(cache_slice.end_offset, cache_gather.end_offset)


@pytest.mark.parametrize("capacity", [8])
@pytest.mark.parametrize("slot", [0, 1, 2, 3])
@pytest.mark.parametrize("num_frames", [1, 3, 5, 15])
def test_slice_path_matches_gather_path_single_step(capacity, slot, num_frames):
    """Single-step B==1: slice view path vs gather path produce identical state."""
    batch_size, num_heads, dim_per_head = 1, 2, 3
    k, v = _random_kv(batch_size, num_heads, num_frames, dim_per_head, seed=42)

    cache_slice = _make_cache(capacity)
    cache_gather = _make_cache(capacity)

    state_slot_ids = torch.tensor([slot], dtype=torch.long)
    valid_rows = torch.tensor([True], dtype=torch.bool)

    ctx_slice = StreamingExecutionContext(state_slot_ids=state_slot_ids, valid_rows=valid_rows, slot0=slot)
    ctx_gather = StreamingExecutionContext(state_slot_ids=state_slot_ids, valid_rows=valid_rows, slot0=None)

    ctx_slice.validate(batch_size=batch_size, state_capacity=capacity, device=torch.device("cpu"))
    ctx_gather.validate(batch_size=batch_size, state_capacity=capacity, device=torch.device("cpu"))

    result_slice = cache_slice.complete(k, v, execution_context=ctx_slice)
    result_gather = cache_gather.complete(k, v, execution_context=ctx_gather)

    torch.testing.assert_close(result_slice.keys, result_gather.keys)
    torch.testing.assert_close(result_slice.values, result_gather.values)
    torch.testing.assert_close(result_slice.positions, result_gather.positions)
    _assert_cache_state_equal(cache_slice, cache_gather, slot)


@pytest.mark.parametrize("capacity", [8])
@pytest.mark.parametrize("slot", [0, 3])
@pytest.mark.parametrize("num_frames", [3, 5, 15])
def test_slice_path_matches_gather_path_multi_step(capacity, slot, num_frames):
    """Multi-step accumulation: slice path vs gather path stay identical over 6 steps."""
    batch_size, num_heads, dim_per_head = 1, 2, 3

    cache_slice = _make_cache(capacity)
    cache_gather = _make_cache(capacity)

    state_slot_ids = torch.tensor([slot], dtype=torch.long)
    valid_rows = torch.tensor([True], dtype=torch.bool)

    ctx_slice = StreamingExecutionContext(state_slot_ids=state_slot_ids, valid_rows=valid_rows, slot0=slot)
    ctx_gather = StreamingExecutionContext(state_slot_ids=state_slot_ids, valid_rows=valid_rows, slot0=None)

    for step in range(6):
        k, v = _random_kv(batch_size, num_heads, num_frames, dim_per_head, seed=step * 100 + 7)
        result_slice = cache_slice.complete(k, v, execution_context=ctx_slice)
        result_gather = cache_gather.complete(k, v, execution_context=ctx_gather)

        torch.testing.assert_close(result_slice.keys, result_gather.keys)
        torch.testing.assert_close(result_slice.values, result_gather.values)
        torch.testing.assert_close(result_slice.positions, result_gather.positions)
        torch.testing.assert_close(cache_slice.cache, cache_gather.cache)
        torch.testing.assert_close(cache_slice.end_offset, cache_gather.end_offset)


def test_slot0_validation_rejects_mismatched_batch_size():
    """slot0=3 with batch_size=2 must raise ValueError."""
    state_slot_ids = torch.tensor([0, 1], dtype=torch.long)
    valid_rows = torch.tensor([True, True], dtype=torch.bool)
    ctx = StreamingExecutionContext(state_slot_ids=state_slot_ids, valid_rows=valid_rows, slot0=3)
    with pytest.raises(ValueError, match="slot0 requires batch_size == 1"):
        ctx.validate(batch_size=2, state_capacity=8, device=torch.device("cpu"))


def test_slot0_validation_rejects_out_of_range_slot():
    """slot0=8 with state_capacity=8 must raise ValueError."""
    state_slot_ids = torch.tensor([0], dtype=torch.long)
    valid_rows = torch.tensor([True], dtype=torch.bool)
    ctx = StreamingExecutionContext(state_slot_ids=state_slot_ids, valid_rows=valid_rows, slot0=8)
    with pytest.raises(ValueError, match="slot0 must be in"):
        ctx.validate(batch_size=1, state_capacity=8, device=torch.device("cpu"))


def test_slot0_validation_accepts_valid_slot():
    """slot0=0 with batch_size=1 and state_capacity=8 must pass."""
    state_slot_ids = torch.tensor([0], dtype=torch.long)
    valid_rows = torch.tensor([True], dtype=torch.bool)
    ctx = StreamingExecutionContext(state_slot_ids=state_slot_ids, valid_rows=valid_rows, slot0=0)
    ctx.validate(batch_size=1, state_capacity=8, device=torch.device("cpu"))


def test_slot0_none_passes_validation_for_any_batch_size():
    """slot0=None must pass validation regardless of batch_size."""
    state_slot_ids = torch.tensor([0, 1], dtype=torch.long)
    valid_rows = torch.tensor([True, True], dtype=torch.bool)
    ctx = StreamingExecutionContext(state_slot_ids=state_slot_ids, valid_rows=valid_rows, slot0=None)
    ctx.validate(batch_size=2, state_capacity=8, device=torch.device("cpu"))


def test_slot0_validation_rejects_mismatched_slot():
    """slot0 must equal state_slot_ids[0] when both are present.

    An in-range but mismatched slot pair (e.g. state_slot_ids=[1], slot0=3)
    silently splits a request's state across two sessions: RingKVCache.complete
    writes slot 3 while attention reads slot 1.  The validation must reject
    this at the API boundary.
    """
    state_slot_ids = torch.tensor([1], dtype=torch.long)
    valid_rows = torch.tensor([True], dtype=torch.bool)
    ctx = StreamingExecutionContext(state_slot_ids=state_slot_ids, valid_rows=valid_rows, slot0=3)
    with pytest.raises(ValueError, match="slot0 .* must match state_slot_ids"):
        ctx.validate(batch_size=1, state_capacity=8, device=torch.device("cpu"))


def test_slot0_validation_accepts_consistent_slot():
    """slot0 == state_slot_ids[0] must pass validation."""
    for slot in [0, 1, 2, 3]:
        state_slot_ids = torch.tensor([slot], dtype=torch.long)
        valid_rows = torch.tensor([True], dtype=torch.bool)
        ctx = StreamingExecutionContext(state_slot_ids=state_slot_ids, valid_rows=valid_rows, slot0=slot)
        ctx.validate(batch_size=1, state_capacity=8, device=torch.device("cpu"))
