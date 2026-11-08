# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Unit tests for the head-split Ulysses overlap admission checks.

The pipeline only reproduces the composed result when each head group's head
count is a multiple of the Ulysses world size; everything else must fall back
to the plain path instead of silently changing the collective geometry.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.parallel.ulysses import UlyssesParallelAttention

pytestmark = [pytest.mark.diffusion, pytest.mark.cpu, pytest.mark.core_model]


def _strategy(world_size: int, use_sync: bool = False) -> UlyssesParallelAttention:
    sp_group = SimpleNamespace(ulysses_world_size=world_size, ulysses_group=None, ulysses_rank=0)
    return UlyssesParallelAttention(
        sp_group=sp_group,
        scatter_idx=2,
        gather_idx=1,
        use_sync=use_sync,
    )


def _qkv(heads: int, kv_heads: int | None = None, seq: int = 4, dim: int = 8):
    query = torch.zeros(1, seq, heads, dim)
    key = torch.zeros(1, seq, kv_heads if kv_heads is not None else heads, dim)
    return query, key, key.clone()


def _plan(world_size: int, groups: int, *, heads: int = 56, kv_heads: int | None = None, metadata=None, use_sync=False):
    query, key, value = _qkv(heads, kv_heads)
    return _strategy(world_size, use_sync).plan_head_overlap(query, key, value, metadata, groups)


class TestPlanHeadOverlap:
    def test_h3_shape_accepts_the_seven_group_split(self):
        # 56 heads over 8 ranks: only 7 groups x 8 heads keeps every group a
        # multiple of the world size, which is exactly the H3 A3 config.
        assert _plan(8, 7) == 7

    def test_rejects_splits_that_do_not_divide_by_the_world_size(self):
        # 56/2 = 28 and 56/4 = 14 are not multiples of 8.
        assert _plan(8, 2) == 0
        assert _plan(8, 4) == 0
        assert _plan(8, 8) == 0
        assert _plan(8, 14) == 0

    def test_accepts_a_two_group_split_at_world_size_four(self):
        assert _plan(4, 2) == 2

    def test_off_or_degenerate_group_counts(self):
        assert _plan(8, 0) == 0
        assert _plan(8, 1) == 0
        assert _plan(8, 1, heads=56) == 0
        assert _plan(8, 56) == 0

    def test_rejects_a_group_count_larger_than_the_head_count(self):
        assert _plan(8, 112) == 0

    def test_rejects_gqa(self):
        # Grouped KV heads would have to be re-grouped per slice.
        assert _plan(8, 7, heads=56, kv_heads=8) == 0

    def test_rejects_a_per_call_host_barrier(self):
        # use_sync serializes every all-to-all on the host, erasing the overlap.
        assert _plan(8, 7, use_sync=True) == 0

    def test_rejects_joint_tensors(self):
        metadata = AttentionMetadata(joint_query=torch.zeros(1, 2, 56, 8))
        assert _plan(8, 7, metadata=metadata) == 0

    def test_rejects_the_vsa_gate(self):
        metadata = AttentionMetadata(extra={"gate_compress": torch.zeros(1, 4, 56, 8)})
        assert _plan(8, 7, metadata=metadata) == 0

    def test_accepts_plain_metadata(self):
        assert _plan(8, 7, metadata=AttentionMetadata()) == 7


def test_rejects_the_fused_permute_all_to_all():
    """The permute-free kernel reshapes the whole head axis; splitting it is untested."""
    strategy = _strategy(8)
    strategy._ulysses_a2a_permute = True
    query, key, value = _qkv(56)

    assert strategy.plan_head_overlap(query, key, value, None, 7) == 0
