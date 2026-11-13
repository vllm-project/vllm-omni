# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CPU tests for the GroupNorm split-reduction heuristic.

``_split_for`` decides how many CTAs cooperate on each ``(batch, group)`` pair
and how wide a spatial slice each one gets. It is pure integer arithmetic, but
it is also what the kernels rely on for their central safety property: every
cooperating program must get a non-empty slice, because the partial combine
weights each partial by its own ``n`` and takes the first one as valid.

The correctness tests for the operators themselves need CUDA, so they only ever
observe whatever split the device under test happens to produce -- one point on
this curve, and never the ``waves``/SM combinations that a different GPU would
hit. These run in the ``core_model and cpu`` lane on every PR instead, and pin
the arithmetic across the whole range.
"""

import pytest

from vllm_omni.model_executor.models.common.ops._group_norm_reduction import (
    SPLIT_ALIGN,
    _split_for,
)

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.cpu,
]

# (spatial_size, groups, num_sms, waves) -> (split, chunk)
_CASES = [
    # Disabled by the escape hatch, whatever the shape.
    ((1024 * 1024, 32, 80, 0), (1, 1024 * 1024)),
    ((16384, 32, 132, 0), (1, 16384)),
    # Too little spatial work to be worth another CTA.
    ((4096, 32, 80, 2), (1, 4096)),
    ((256, 32, 132, 2), (1, 256)),
    # The HunyuanImage3 decode ladder at batch 1 on an A10G (80 SMs).
    ((1024 * 1024, 32, 80, 2), (5, 212992)),
    ((512 * 512, 32, 80, 2), (5, 53248)),
    ((256 * 256, 32, 80, 2), (4, 16384)),
    ((128 * 128, 32, 80, 2), (4, 4096)),
    # A ragged tail: 12480 = 3 full 4096 chunks plus 192.
    ((12480, 32, 80, 2), (4, 4096)),
    # Enough groups to already fill the device -- no split needed.
    ((512 * 512, 8 * 32, 80, 2), (1, 512 * 512)),
]


@pytest.mark.parametrize("args, expected", _CASES)
def test_split_for_is_pinned(args, expected):
    assert _split_for(*args) == expected


@pytest.mark.parametrize("waves", [0, -1, 1, 2, 4, 8, 16])
@pytest.mark.parametrize("num_sms", [1, 16, 58, 80, 108, 132, 256])
@pytest.mark.parametrize("spatial_size", [1, 255, 256, 4095, 4096, 4097, 12480, 16384, 65536, 1024 * 1024])
@pytest.mark.parametrize("groups", [1, 32, 128, 2048])
def test_split_for_invariants(spatial_size, groups, num_sms, waves):
    """Every cooperating program gets a non-empty, aligned slice.

    This is the property the partial combine depends on: it takes ``n_i`` at
    face value and never guards against a zero-length partial. Sweeping the
    grid is cheap here and covers SM counts and ``waves`` values no single GPU
    would exercise.
    """
    split, chunk = _split_for(spatial_size, groups, num_sms, waves)

    assert split >= 1
    assert chunk >= 1
    # The slices must tile the axis: cover it, without a whole spare chunk.
    assert split * chunk >= spatial_size
    assert (split - 1) * chunk < spatial_size, "a program would get an empty slice"

    if split == 1:
        assert chunk == spatial_size
    else:
        # Split chunks are BLOCK_SIZE-aligned so that every program except the
        # last issues only full, aligned loads.
        assert chunk % SPLIT_ALIGN == 0
        assert chunk >= SPLIT_ALIGN


def test_waves_zero_disables_the_split_everywhere():
    """The documented escape hatch, across the shapes where a split would apply."""
    for spatial_size in (12480, 16384, 65536, 1024 * 1024):
        for num_sms in (58, 80, 132):
            assert _split_for(spatial_size, 32, num_sms, 0) == (1, spatial_size)


def test_split_grows_with_device_width():
    """More SMs means more cooperating CTAs, never fewer."""
    splits = [_split_for(1024 * 1024, 32, sms, 2)[0] for sms in (16, 58, 80, 108, 132, 256)]
    assert splits == sorted(splits), splits
    assert splits[0] == 1, "a 16-SM device has no room to split at 32 groups"
    assert splits[-1] > splits[0]


def test_split_shrinks_as_groups_already_fill_the_device():
    """Batching raises the group count, which is parallelism the split need not add."""
    splits = [_split_for(512 * 512, 32 * b, 80, 2)[0] for b in (1, 2, 4, 8, 16)]
    assert splits == sorted(splits, reverse=True), splits
    assert splits[-1] == 1
