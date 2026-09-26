# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Ulysses for window attention: canonical sequence shards and uneven head shards."""

from __future__ import annotations

import math

import torch
import torch.distributed as dist

from .na_ops import LocalWindowContext, SeedVR2WindowRuntime
from .window_geometry import DEFAULT_WINDOW, DEFAULT_WINDOW_METHODS
from .window_sp import WindowLayout, WindowLayoutKey


def balanced_sizes(size: int, parts: int) -> tuple[int, ...]:
    return tuple(size // parts + (rank < size % parts) for rank in range(parts))


class SeedVR2UlyssesRuntime(SeedVR2WindowRuntime):
    """Keep MLP rows sharded; exchange sequence for heads only around attention."""

    def __init__(
        self,
        token_grid: tuple[int, int, int],
        *,
        text_len: int,
        heads: int,
        group: dist.ProcessGroup,
        world_size: int,
        rank: int,
        window: tuple[int, int, int] = DEFAULT_WINDOW,
        methods: tuple[str, ...] = DEFAULT_WINDOW_METHODS,
        num_layers: int = 32,
    ) -> None:
        if not 1 <= world_size <= heads or dist.get_world_size(group) != world_size:
            raise ValueError("Ulysses requires a matching group with at least one attention head per rank")
        # Every head shard sees every window. The inherited planner therefore
        # builds single-rank window metadata, independent of sequence ownership.
        super().__init__(token_grid, text_len=text_len, window=window, methods=methods, num_layers=num_layers)
        self.group, self.world_size, self.rank = group, world_size, rank
        self.token_sizes = balanced_sizes(math.prod(token_grid), world_size)
        self.head_sizes = balanced_sizes(heads, world_size)
        self._orders: dict[tuple[WindowLayoutKey, torch.device], tuple[torch.Tensor, torch.Tensor]] = {}
        self.stats.update(ulysses_exchanges=0, ulysses_head_gathers=0)

    def orders(self, ctx: LocalWindowContext) -> tuple[torch.Tensor, torch.Tensor]:
        key = (ctx.layout_key, ctx.joint_order.device)
        if key not in self._orders:
            layout = self.manager.layout_for_key(ctx.layout_key)
            order = layout.window_token_ids.to(ctx.joint_order.device)
            self._orders[key] = order, torch.argsort(order)
        return self._orders[key]

    def local_rows_for(self, canonical_rows: torch.Tensor, layout: WindowLayout) -> torch.Tensor:
        return canonical_rows.narrow(0, sum(self.token_sizes[: self.rank]), self.token_sizes[self.rank])

    def ensure_layout(self, hidden: torch.Tensor, current: WindowLayoutKey, required: WindowLayoutKey) -> torch.Tensor:
        return hidden

    def to_heads(self, video: torch.Tensor, ctx: LocalWindowContext) -> torch.Tensor:
        """[local tokens, 3, all heads, D] -> [window-packed tokens, 3, local heads, D]."""
        local_tokens, qkv, _, dim = video.shape
        local_heads = self.head_sizes[self.rank]
        send = torch.cat([part.reshape(-1) for part in video.split(self.head_sizes, dim=2)])
        recv = video.new_empty(sum(self.token_sizes) * qkv * local_heads * dim)
        dist.all_to_all_single(
            recv,
            send,
            output_split_sizes=[tokens * qkv * local_heads * dim for tokens in self.token_sizes],
            input_split_sizes=[local_tokens * qkv * heads * dim for heads in self.head_sizes],
            group=self.group,
        )
        self.stats["ulysses_exchanges"] += 1
        return recv.view(-1, qkv, local_heads, dim).index_select(0, self.orders(ctx)[0])

    def text_heads(self, text: torch.Tensor) -> torch.Tensor:
        return text.narrow(2, sum(self.head_sizes[: self.rank]), self.head_sizes[self.rank])

    def from_heads(self, video: torch.Tensor, ctx: LocalWindowContext) -> torch.Tensor:
        """[window-packed tokens, local heads, D] -> [local tokens, all heads, D]."""
        canonical = video.index_select(0, self.orders(ctx)[1])
        dim = video.shape[-1]
        local_heads, local_tokens = self.head_sizes[self.rank], self.token_sizes[self.rank]
        receive_sizes = [local_tokens * heads * dim for heads in self.head_sizes]
        received = video.new_empty(sum(receive_sizes))
        dist.all_to_all_single(
            received,
            canonical.reshape(-1),
            output_split_sizes=receive_sizes,
            input_split_sizes=[tokens * local_heads * dim for tokens in self.token_sizes],
            group=self.group,
        )
        self.stats["ulysses_exchanges"] += 1
        return torch.cat(
            [
                part.view(local_tokens, heads, dim)
                for part, heads in zip(received.split(receive_sizes), self.head_sizes, strict=True)
            ],
            dim=1,
        )

    def gather_axis(self, local: torch.Tensor, sizes: tuple[int, ...], axis: int) -> torch.Tensor:
        shape = list(local.shape)
        shape[axis] = max(sizes)
        padded = local.new_zeros(shape)
        padded.narrow(axis, 0, sizes[self.rank]).copy_(local)
        gathered = [torch.empty_like(padded) for _ in sizes]
        dist.all_gather(gathered, padded, group=self.group)
        return torch.cat([part.narrow(axis, 0, size) for part, size in zip(gathered, sizes, strict=True)], dim=axis)

    def reduce_text(self, local_window_sum: torch.Tensor, global_windows: int) -> torch.Tensor:
        heads = self.head_sizes[self.rank]
        mean = (local_window_sum.float() / global_windows).to(local_window_sum.dtype)
        self.stats["ulysses_head_gathers"] += 1
        return self.gather_axis(mean.view(self.text_len, heads, -1), self.head_sizes, 1).flatten(1)

    def to_canonical_rows(self, local_rows: torch.Tensor, layout: WindowLayout) -> torch.Tensor:
        return self.gather_axis(local_rows, self.token_sizes, 0)
