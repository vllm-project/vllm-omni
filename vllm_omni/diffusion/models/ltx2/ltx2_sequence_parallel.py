# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.parallel.base import (
    ParallelAttentionContext,
)
from vllm_omni.diffusion.distributed.comm import SeqAllToAll4D
from vllm_omni.diffusion.distributed.group_coordinator import (
    SequenceParallelGroupCoordinator,
)


@dataclass(frozen=True, slots=True)
class _VideoToAudioContext(ParallelAttentionContext):
    ulysses_pg: dist.ProcessGroup
    use_sync: bool


class LTX2VideoToAudioParallelAttention:
    """LTX video-to-audio SP with replicated audio queries.

    Video K/V arrive sequence-sharded, while the much shorter audio query is
    replicated. Redistribute only K/V across sequence/head dimensions, slice
    audio query heads locally, then gather output heads. This uses three
    collectives instead of standard Ulysses' four and keeps audio replicated.
    """

    def __init__(
        self,
        sp_group: SequenceParallelGroupCoordinator,
        *,
        use_sync: bool = False,
    ) -> None:
        if sp_group.ring_world_size != 1:
            raise NotImplementedError(
                "LTX video-to-audio sequence parallelism currently supports pure Ulysses only (ring_degree must be 1)."
            )
        self._sp_group = sp_group
        self._ulysses_pg = sp_group.ulysses_group
        self._use_sync = use_sync

    @property
    def enabled(self) -> bool:
        return True

    @property
    def name(self) -> str:
        return "ltx2_video_to_audio"

    def pre_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None,
    ):
        if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
            raise ValueError(
                "LTX video-to-audio SP expects Q/K/V in (B, S, H, D) layout, "
                f"got {query.shape=}, {key.shape=}, {value.shape=}."
            )
        if key.shape != value.shape:
            raise ValueError(
                "LTX video-to-audio SP requires matching K/V shapes, "
                f"got key={tuple(key.shape)} and value={tuple(value.shape)}."
            )

        world_size = self._sp_group.ulysses_world_size
        rank = self._sp_group.ulysses_rank
        query_heads = int(query.shape[2])
        if query_heads % world_size != 0:
            raise ValueError(
                "LTX video-to-audio SP requires query heads divisible by "
                f"ulysses_degree, got heads={query_heads}, ulysses_degree={world_size}."
            )
        if int(key.shape[2]) % world_size != 0:
            raise ValueError(
                "LTX video-to-audio SP requires K/V heads divisible by "
                f"ulysses_degree, got heads={key.shape[2]}, ulysses_degree={world_size}."
            )

        heads_per_rank = query_heads // world_size
        head_start = rank * heads_per_rank
        query = query[:, :, head_start : head_start + heads_per_rank, :].contiguous()
        key = SeqAllToAll4D.apply(self._ulysses_pg, key, 2, 1, self._use_sync)
        value = SeqAllToAll4D.apply(self._ulysses_pg, value, 2, 1, self._use_sync)

        ctx = _VideoToAudioContext(
            name=self.name,
            ulysses_pg=self._ulysses_pg,
            use_sync=self._use_sync,
        )
        return query, key, value, attn_metadata, ctx

    def post_attention(
        self,
        attn_output: torch.Tensor,
        ctx: ParallelAttentionContext | None,
    ) -> torch.Tensor:
        if not isinstance(ctx, _VideoToAudioContext):
            raise TypeError(f"Unexpected LTX video-to-audio context: {type(ctx)!r}.")

        # Gather along heads by moving H to the leading dimension required by
        # all_gather_into_tensor, then restore the backend's (B, S, H, D) layout.
        local_heads_first = attn_output.permute(2, 0, 1, 3).contiguous()
        world_size = dist.get_world_size(ctx.ulysses_pg)
        gathered = torch.empty(
            (local_heads_first.shape[0] * world_size, *local_heads_first.shape[1:]),
            dtype=attn_output.dtype,
            device=attn_output.device,
        )
        dist.all_gather_into_tensor(gathered, local_heads_first, group=ctx.ulysses_pg)
        if ctx.use_sync:
            from vllm_omni.platforms import current_omni_platform

            current_omni_platform.synchronize()
        return gathered.permute(1, 2, 0, 3).contiguous()
