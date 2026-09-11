# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.models.ltx2.ltx2_sequence_parallel import (
    LTX2VideoToAudioParallelAttention,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _run_video_to_audio_parity(rank: int, world_size: int, master_port: int) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{master_port}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=60),
    )
    try:
        # Non-contiguous SP subgroups match TP-fastest rank ordering. Different
        # seeds across groups catch collectives that accidentally use WORLD.
        group_count = world_size // 2
        ulysses_group = dist.group.WORLD
        if group_count > 1:
            for offset in range(group_count):
                ranks = list(range(offset, world_size, group_count))
                group = dist.new_group(ranks)
                if rank in ranks:
                    ulysses_group = group
        ulysses_rank = dist.get_rank(ulysses_group)
        sp_group = SimpleNamespace(
            ring_world_size=1,
            ulysses_group=ulysses_group,
            ulysses_world_size=2,
            ulysses_rank=ulysses_rank,
        )
        strategy = LTX2VideoToAudioParallelAttention(sp_group)

        for batch_size, masked in [(1, False), (2, True)]:
            torch.manual_seed(17 + rank % group_count)
            query = torch.randn(batch_size, 3, 4, 8)
            global_key = torch.randn(batch_size, 6, 4, 8)
            global_value = torch.randn(batch_size, 6, 4, 8)
            key = global_key.chunk(2, dim=1)[ulysses_rank].contiguous()
            value = global_value.chunk(2, dim=1)[ulysses_rank].contiguous()
            mask = None
            if masked:
                mask = torch.ones(batch_size, 1, 1, 6, dtype=torch.bool)
                mask[0, ..., -1] = False
                mask[1, ..., -2:] = False
            metadata = AttentionMetadata(attn_mask=mask) if masked else None

            local_query, local_key, local_value, local_metadata, ctx = strategy.pre_attention(
                query, key, value, metadata
            )
            assert local_metadata is metadata
            local_output = F.scaled_dot_product_attention(
                local_query.transpose(1, 2),
                local_key.transpose(1, 2),
                local_value.transpose(1, 2),
                attn_mask=local_metadata.attn_mask if local_metadata is not None else None,
            ).transpose(1, 2)
            actual = strategy.post_attention(local_output, ctx)
            expected = F.scaled_dot_product_attention(
                query.transpose(1, 2),
                global_key.transpose(1, 2),
                global_value.transpose(1, 2),
                attn_mask=mask,
            ).transpose(1, 2)

            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [2, 4], ids=["world", "subgroups"])
def test_ltx_video_to_audio_sp_matches_replicated_attention(unused_tcp_port, world_size):
    torch.multiprocessing.spawn(
        _run_video_to_audio_parity,
        args=(world_size, unused_tcp_port),
        nprocs=world_size,
    )


def test_ltx_video_to_audio_sp_rejects_ring_parallelism():
    sp_group = SimpleNamespace(ring_world_size=2)

    with pytest.raises(NotImplementedError, match="ring_degree must be 1"):
        LTX2VideoToAudioParallelAttention(sp_group)
