# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""L1 unit tests for BOOGU's SP shard-layout arithmetic.

These cover the property the model relies on to build attention masks without
communicating: every rank can derive *every* rank's segment bounds from the
global sequence length alone.
"""

import pytest
import torch

from vllm_omni.diffusion.models.boogu_image.boogu_image_transformer import (
    BooguImageTransformer2DModel,
)
from vllm_omni.diffusion.models.boogu_image.sp_layout import ShardLayout

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


class TestShardLayout:
    def test_padding_and_local_extent(self):
        layout = ShardLayout(original_seq_len=5, world_size=2, rank=0)
        assert layout.padded_seq_len == 6
        assert layout.local_seq_len == 3
        assert layout.padding_size == 1
        assert layout.bounds(0) == (0, 3)
        assert layout.bounds(1) == (3, 6)

    def test_valid_lengths_clip_to_each_rank(self):
        layout = ShardLayout(original_seq_len=5, world_size=2, rank=0)
        # Sample 0 has 5 valid tokens (3 on rank 0, 2 on rank 1); sample 1 has 2.
        assert layout.valid_lengths([5, 2], rank=0) == [3, 2]
        assert layout.valid_lengths([5, 2], rank=1) == [2, 0]

    def test_segment_lengths_split_across_the_boundary(self):
        layout = ShardLayout(original_seq_len=5, world_size=2, rank=0)
        # Two reference images of 2 and 3 tokens; the second straddles rank 0/1.
        assert layout.segment_lengths([[2, 3]], rank=0) == [[2, 1]]
        assert layout.segment_lengths([[2, 3]], rank=1) == [[0, 2]]

    def test_unsharded_fallback_is_identity(self):
        layout = ShardLayout(original_seq_len=7, world_size=1, rank=0)
        assert layout.local_seq_len == 7
        assert layout.padding_size == 0
        assert layout.valid_lengths([7, 3], rank=0) == [7, 3]

    @pytest.mark.parametrize("world_size", [1, 2, 4])
    def test_per_rank_lengths_sum_to_the_global_length(self, world_size):
        layout = ShardLayout(original_seq_len=13, world_size=world_size, rank=0)
        for global_length in (0, 1, 7, 13):
            total = sum(layout.valid_lengths([global_length], rank=r)[0] for r in range(world_size))
            assert total == global_length


class TestRankConcatMask:
    """The mask must describe the sequence Ulysses forms after its all-to-all.

    That sequence is rank0's local shard followed by rank1's, etc. -- not the
    natural global ordering -- because each rank packs [context|ref|noise]
    locally before attention.
    """

    def test_mask_is_concatenation_of_per_rank_segments(self):
        mask = BooguImageTransformer2DModel._rank_concat_mask_or_none(
            [[2], [1]],
            capacity=3,
            like=torch.zeros(1, 1),
        )
        # rank 0 contributes [T, T, F], rank 1 contributes [T, F, F]
        assert mask.tolist() == [[True, True, False, True, False, False]]

    def test_all_valid_mask_is_dropped(self):
        assert (
            BooguImageTransformer2DModel._rank_concat_mask_or_none(
                [[3], [3]],
                capacity=3,
                like=torch.zeros(1, 1),
            )
            is None
        )

    def test_required_keeps_an_all_valid_mask(self):
        mask = BooguImageTransformer2DModel._rank_concat_mask_or_none(
            [[3], [3]],
            capacity=3,
            like=torch.zeros(1, 1),
            required=True,
        )
        assert mask is not None
        assert bool(mask.all())

    def test_batched_samples_keep_independent_lengths(self):
        mask = BooguImageTransformer2DModel._rank_concat_mask_or_none(
            [[2, 0], [2, 1]],
            capacity=2,
            like=torch.zeros(1, 1),
        )
        assert mask.tolist() == [
            [True, True, True, True],
            [False, False, True, False],
        ]

    def test_single_rank_mask_matches_a_plain_prefix_mask(self):
        """With SP off the rank-concatenated mask degenerates to the old one."""
        lengths = [3, 1]
        mask = BooguImageTransformer2DModel._rank_concat_mask_or_none(
            [lengths],
            capacity=4,
            like=torch.zeros(1, 1),
        )
        expected = torch.zeros(2, 4, dtype=torch.bool)
        for i, length in enumerate(lengths):
            expected[i, :length] = True
        assert torch.equal(mask, expected)


class TestRankConcatEquivalence:
    """The built mask must equal an all-gather of per-rank local masks.

    BOOGU used to hand rank-local masks to the attention layer, which
    all-gathered them into the global sequence. The model now builds that
    global mask directly -- saving one collective per attention call -- so the
    two constructions must agree exactly. This is asserted structurally
    because the end-to-end SP1-vs-SP2 comparison is numerically insensitive to
    masking (a run with masks entirely disabled still matches to ~5e-7).
    """

    @staticmethod
    def _all_gather_of_local_masks(per_rank_lengths, capacity):
        """What the deleted `dist.all_gather` path used to produce."""
        batch_size = len(per_rank_lengths[0])
        local_masks = []
        for lengths in per_rank_lengths:
            local = torch.zeros(batch_size, capacity, dtype=torch.bool)
            for i, length in enumerate(lengths):
                local[i, :length] = True
            local_masks.append(local)
        return torch.cat(local_masks, dim=1)

    @pytest.mark.parametrize(
        ("global_len", "world_size", "sample_lengths"),
        [
            (5, 2, [5, 2]),
            (8, 2, [8, 1]),
            (13, 4, [13, 7, 0]),
            (16, 4, [16]),
            (7, 1, [7, 3]),
        ],
    )
    def test_matches_all_gather_of_local_masks(self, global_len, world_size, sample_lengths):
        layout = ShardLayout(original_seq_len=global_len, world_size=world_size, rank=0)
        per_rank = [layout.valid_lengths(sample_lengths, rank=r) for r in range(world_size)]

        built = BooguImageTransformer2DModel._rank_concat_mask_or_none(
            per_rank,
            layout.local_seq_len,
            like=torch.zeros(1, 1),
            required=True,
        )
        assert torch.equal(built, self._all_gather_of_local_masks(per_rank, layout.local_seq_len))

    def test_valid_token_count_is_preserved_across_ranks(self):
        layout = ShardLayout(original_seq_len=13, world_size=4, rank=0)
        per_rank = [layout.valid_lengths([13, 5], rank=r) for r in range(4)]
        mask = BooguImageTransformer2DModel._rank_concat_mask_or_none(
            per_rank,
            layout.local_seq_len,
            like=torch.zeros(1, 1),
            required=True,
        )
        # Sharding must neither drop nor invent valid positions.
        assert mask[0].sum().item() == 13
        assert mask[1].sum().item() == 5
