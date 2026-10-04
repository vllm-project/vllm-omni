# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU tests for the text-encoder TP group layout.

``_text_encoder_group_ranks`` decides the process groups behind
``--text-encoder-tp-size``.  It must tile the DiT world into contiguous
groups so that every rank holds group membership (``GroupCoordinator``
asserts on ranks outside all groups), while keeping the first group exactly
``[0, tp_size)`` so the "first tp_size ranks encode" contract is unchanged.
"""

import pytest

from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import (
    _text_encoder_group_ranks,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.mark.parametrize(
    ("dit_world", "tp_size", "expected"),
    [
        # tp_size == world: a single group covering every rank, identical to
        # the previous init_world_group([0..tp)) behavior.
        (4, 4, [[0, 1, 2, 3]]),
        (8, 8, [[0, 1, 2, 3, 4, 5, 6, 7]]),
        # tp_size < world: tiled groups, every rank is a member of exactly one.
        (16, 8, [[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15]]),
        (16, 4, [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]),
        (16, 2, [[2 * i, 2 * i + 1] for i in range(8)]),
        (4, 2, [[0, 1], [2, 3]]),
        (8, 4, [[0, 1, 2, 3], [4, 5, 6, 7]]),
    ],
)
def test_text_encoder_group_ranks_tile_dit_world(dit_world, tp_size, expected):
    group_ranks = _text_encoder_group_ranks(dit_world, tp_size)

    assert group_ranks == expected
    # Every DiT rank is a member of exactly one group (GroupCoordinator assert).
    assert sorted(rank for group in group_ranks for rank in group) == list(range(dit_world))
    # The encoding group contract is unchanged: the first tp_size ranks encode.
    assert group_ranks[0] == list(range(tp_size))


@pytest.mark.parametrize(("dit_world", "tp_size"), [(16, 3), (16, 6), (8, 3), (4, 3), (2, 3)])
def test_text_encoder_group_ranks_rejects_non_divisible(dit_world, tp_size):
    with pytest.raises(ValueError, match="must divide the DiT group size"):
        _text_encoder_group_ranks(dit_world, tp_size)


def test_text_encoder_group_ranks_rejects_zero_tp():
    with pytest.raises(ValueError, match=">= 1"):
        _text_encoder_group_ranks(16, 0)
