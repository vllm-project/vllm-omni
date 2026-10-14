# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Copyright (c) 2026 SandAI. All Rights Reserved.

"""Block-window attention planning for the MAGI-2 refiner.

The refiner attends locally on a block grid of the video latent.  Video tokens
are permuted block-major (raster order inside a block) and followed by the
text tokens.  A video query block attends every key block within the window
radius plus all text tokens; text queries attend every token.

``build_reference_window_ranges`` reproduces the token order and the q/k range
layout of SandAI's Apache-2.0 MAGI-2 refiner data proxy.
``build_window_attention_plan`` expresses the same key sets as a paged-KV plan
for varlen FlashAttention kernels, and ``torch_window_attention_reference``
evaluates a plan exactly in PyTorch for small inputs.
``Magi2WindowAttentionKernel`` runs a plan behind the shared diffusion
``Attention`` layer.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn as nn

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata

from .configuration_magi2 import MAGI2_REFINER_CONFIG, Magi2RefinerWindowConfig
from .parallel import Magi2ParallelGroup, get_magi2_ulysses_group

PAGE_SIZE = 16
TORCH_REFERENCE_MAX_TOKENS = 65_536


@dataclass(frozen=True, eq=False)
class Magi2WindowRanges:
    """Reference window-attention ranges over the block-major token sequence.

    ``q_ranges``/``k_ranges`` are int32 ``[rows, 2]`` half-open ranges: first
    the dense rows (every video block against the text tokens, then the text
    queries in chunks against the video and against the text tokens; none
    without text), then the window rows ordered by neighbour offset (t, h, w
    outer to inner) and query block.  ``token_order[i]`` is the raster token at
    block-major position ``i``; ``token_restore_order`` is its inverse.
    """

    grid: tuple[int, int, int]
    text_len: int
    block_grid: tuple[int, int, int]
    block_ranges: torch.Tensor
    token_order: torch.Tensor
    token_restore_order: torch.Tensor
    q_ranges: torch.Tensor
    k_ranges: torch.Tensor
    num_dense_rows: int
    max_seqlen_q: int
    max_seqlen_k: int

    @property
    def num_video_tokens(self) -> int:
        return math.prod(self.grid)

    @property
    def num_tokens(self) -> int:
        return self.num_video_tokens + self.text_len


@dataclass(frozen=True, eq=False)
class Magi2WindowAttentionPlan:
    """Paged-KV plan for the refiner's window attention.

    Keys and values are stored as zero-padded pages of ``page_size`` tokens of
    the block-major sequence (see ``pad_to_pages``).  Video query block ``b``
    is ``[cu_seqlens_q[b], cu_seqlens_q[b + 1])``; its keys are the first
    ``seqused_k[b]`` tokens of the pages ``page_table[b]``: the pages of the
    in-window key blocks in window order, then the text pages.  Entries past a
    row's last page are zero.  The text queries ``[num_video_tokens,
    num_tokens)`` form one row over every page (``text_page_table``,
    ``text_seqused_k``); there is no text row without text.
    """

    grid: tuple[int, int, int]
    text_len: int
    block_grid: tuple[int, int, int]
    page_size: int
    cu_seqlens_q: torch.Tensor
    page_table: torch.Tensor
    seqused_k: torch.Tensor
    max_seqlen_q: int
    max_seqlen_k: int
    text_cu_seqlens_q: torch.Tensor
    text_page_table: torch.Tensor
    text_seqused_k: torch.Tensor

    @property
    def num_video_tokens(self) -> int:
        return math.prod(self.grid)

    @property
    def num_tokens(self) -> int:
        return self.num_video_tokens + self.text_len

    @property
    def num_blocks(self) -> int:
        return math.prod(self.block_grid)

    @property
    def num_pages(self) -> int:
        return -(-self.num_tokens // self.page_size)

    @property
    def num_padded_tokens(self) -> int:
        return self.num_pages * self.page_size

    def to(self, device: torch.device | str) -> Magi2WindowAttentionPlan:
        tensors = {
            f.name: getattr(self, f.name).to(device)
            for f in dataclasses.fields(self)
            if isinstance(getattr(self, f.name), torch.Tensor)
        }
        return dataclasses.replace(self, **tensors)

    def pad_to_pages(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``x`` (``[num_tokens, ...]``) as zero-padded ``[num_pages, page_size, ...]`` pages."""

        if x.shape[0] != self.num_tokens:
            raise ValueError(f"expected {self.num_tokens} tokens, got {x.shape[0]}")
        padded = x.new_zeros((self.num_padded_tokens, *x.shape[1:]))
        padded[: self.num_tokens].copy_(x)
        return padded.view(self.num_pages, self.page_size, *x.shape[1:])


def _validate_grid(grid: Sequence[int], text_len: int) -> tuple[tuple[int, int, int], int]:
    grid = tuple(int(n) for n in grid)
    if len(grid) != 3 or any(n <= 0 for n in grid):
        raise ValueError(f"MAGI-2 refiner latent grid must be three positive (T, H, W) values, got {grid}")
    text_len = int(text_len)
    if text_len < 0:
        raise ValueError(f"text_len must be non-negative, got {text_len}")
    if math.prod(grid) + text_len >= 2**31:
        raise ValueError("MAGI-2 refiner sequence does not fit int32 token indices")
    return grid, text_len


def _axis_block_sizes(n: int, block: int) -> torch.Tensor:
    num_blocks = -(-n // block)
    sizes = torch.full((num_blocks,), block, dtype=torch.int64)
    sizes[-1] = n - block * (num_blocks - 1)
    return sizes


def _block_layout(
    grid: tuple[int, int, int], window: Magi2RefinerWindowConfig
) -> tuple[tuple[int, int, int], torch.Tensor, torch.Tensor]:
    """Return the block grid and the int64 start and size of every block in block-major order."""

    t_sizes, h_sizes, w_sizes = (_axis_block_sizes(n, b) for n, b in zip(grid, window.block_size))
    sizes = (t_sizes[:, None, None] * h_sizes[None, :, None] * w_sizes[None, None, :]).flatten()
    starts = torch.cumsum(sizes, 0) - sizes
    return (t_sizes.numel(), h_sizes.numel(), w_sizes.numel()), starts, sizes


def block_major_order(
    grid: Sequence[int],
    window: Magi2RefinerWindowConfig = MAGI2_REFINER_CONFIG.window,
) -> torch.Tensor:
    """Return the int64 raster index of the video token at each block-major position.

    Blocks are visited in (t, h, w) grid order and tokens in raster order
    inside a block; partial edge blocks keep only their real tokens.
    """

    grid, _ = _validate_grid(grid, 0)
    padded = tuple(-(-n // b) * b for n, b in zip(grid, window.block_size))
    index = torch.full(padded, -1, dtype=torch.int64)
    index[: grid[0], : grid[1], : grid[2]] = torch.arange(math.prod(grid)).view(grid)
    bt, bh, bw = window.block_size
    index = index.view(padded[0] // bt, bt, padded[1] // bh, bh, padded[2] // bw, bw)
    index = index.permute(0, 2, 4, 1, 3, 5).flatten()
    return index[index >= 0]


def _window_offsets(window: Magi2RefinerWindowConfig) -> torch.Tensor:
    """Return the ``[num_offsets, 3]`` neighbour offsets, t outer to w inner."""

    axes = (torch.arange(-r, r + 1) for r in window.radius)
    return torch.cartesian_prod(*axes).view(-1, 3)


def _block_neighbours(
    block_grid: tuple[int, int, int], window: Magi2RefinerWindowConfig
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``[num_blocks, num_offsets]`` key block ids and their in-grid mask."""

    nt, nh, nw = block_grid
    block = torch.arange(nt * nh * nw)
    coords = torch.stack((block // (nh * nw), (block // nw) % nh, block % nw), dim=1)
    neighbour = coords[:, None, :] + _window_offsets(window)[None, :, :]
    limits = torch.tensor(block_grid)
    valid = ((neighbour >= 0) & (neighbour < limits)).all(dim=-1)
    ids = (neighbour[..., 0] * nh + neighbour[..., 1]) * nw + neighbour[..., 2]
    return torch.where(valid, ids, 0), valid


def build_reference_window_ranges(
    grid: Sequence[int],
    text_len: int,
    window: Magi2RefinerWindowConfig = MAGI2_REFINER_CONFIG.window,
) -> Magi2WindowRanges:
    """Build the reference token order and q/k ranges for a ``(T, H, W)`` latent grid."""

    grid, text_len = _validate_grid(grid, text_len)
    num_video = math.prod(grid)
    num_tokens = num_video + text_len
    block_grid, starts, sizes = _block_layout(grid, window)
    block_ranges = torch.stack((starts, starts + sizes), dim=1)

    order = torch.cat((block_major_order(grid, window), torch.arange(num_video, num_tokens)))
    restore = torch.empty_like(order)
    restore[order] = torch.arange(num_tokens)

    neighbours, valid = _block_neighbours(block_grid, window)
    # Window rows: for each offset (t, h, w outer to inner), every query block with an in-grid neighbour.
    query_blocks = torch.arange(neighbours.shape[0])[:, None].expand_as(neighbours)
    window_q = block_ranges[query_blocks.t()[valid.t()]]
    window_k = block_ranges[neighbours.t()[valid.t()]]

    q_parts, k_parts = [], []
    if text_len > 0:
        chunk = window.dense_query_chunk
        text_starts = torch.arange(num_video, num_tokens, chunk)
        text_q = torch.stack((text_starts, torch.clamp(text_starts + chunk, max=num_tokens)), dim=1)
        video_k = torch.tensor([[0, num_video]])
        text_k = torch.tensor([[num_video, num_tokens]])
        q_parts = [block_ranges, text_q, text_q]
        k_parts = [
            text_k.expand(block_ranges.shape[0], 2),
            video_k.expand(text_q.shape[0], 2),
            text_k.expand(text_q.shape[0], 2),
        ]
    num_dense_rows = sum(part.shape[0] for part in q_parts)
    q_ranges = torch.cat((*q_parts, window_q)).to(torch.int32).contiguous()
    k_ranges = torch.cat((*k_parts, window_k)).to(torch.int32).contiguous()
    return Magi2WindowRanges(
        grid=grid,
        text_len=text_len,
        block_grid=block_grid,
        block_ranges=block_ranges.to(torch.int32).contiguous(),
        token_order=order.to(torch.int32).contiguous(),
        token_restore_order=restore.to(torch.int32).contiguous(),
        q_ranges=q_ranges,
        k_ranges=k_ranges,
        num_dense_rows=num_dense_rows,
        max_seqlen_q=int((q_ranges[:, 1] - q_ranges[:, 0]).max()),
        max_seqlen_k=num_tokens,
    )


def build_window_attention_plan(
    grid: Sequence[int],
    text_len: int,
    window: Magi2RefinerWindowConfig = MAGI2_REFINER_CONFIG.window,
    page_size: int = PAGE_SIZE,
) -> Magi2WindowAttentionPlan:
    """Build the paged-KV window-attention plan for a ``(T, H, W)`` latent grid.

    Every video block must cover whole pages, so H and W must be multiples of
    the block's h and w sizes and one block frame (``bh * bw`` tokens) a
    multiple of ``page_size``.
    """

    grid, text_len = _validate_grid(grid, text_len)
    _, bh, bw = window.block_size
    if page_size <= 0 or grid[1] % bh or grid[2] % bw or (bh * bw) % page_size:
        raise ValueError(
            f"MAGI-2 refiner paged window attention needs H % {bh} == 0, W % {bw} == 0 and "
            f"{bh * bw}-token block frames divisible by page_size={page_size}; got grid={grid}"
        )
    num_video = math.prod(grid)
    num_tokens = num_video + text_len
    block_grid, starts, sizes = _block_layout(grid, window)
    # Every block covers whole bh * bw frames, so every size and start is a page multiple.
    assert not (sizes % page_size).any()
    block_first_page = starts // page_size
    block_pages = sizes // page_size

    neighbours, valid = _block_neighbours(block_grid, window)
    counts = torch.where(valid, block_pages[neighbours], 0)
    video_pages = counts.sum(dim=1)
    text_pages = -(-text_len // page_size)
    page_table = torch.zeros(neighbours.shape[0], int(video_pages.max()) + text_pages, dtype=torch.int64)
    # Lay each row's neighbour blocks out page by page, in window order.
    rows, slots = torch.nonzero(valid, as_tuple=True)
    pages_per_slot = counts[rows, slots]
    slot_column = (torch.cumsum(counts, dim=1) - counts)[rows, slots]
    page_in_block = torch.arange(int(pages_per_slot.sum())) - torch.repeat_interleave(
        torch.cumsum(pages_per_slot, 0) - pages_per_slot, pages_per_slot
    )
    flat_index = torch.repeat_interleave(rows * page_table.shape[1] + slot_column, pages_per_slot) + page_in_block
    page_ids = torch.repeat_interleave(block_first_page[neighbours[rows, slots]], pages_per_slot) + page_in_block
    page_table.view(-1)[flat_index] = page_ids
    if text_pages:
        text_page_ids = num_video // page_size + torch.arange(text_pages)
        text_columns = video_pages[:, None] + torch.arange(text_pages)[None, :]
        page_table.scatter_(1, text_columns, text_page_ids.expand_as(text_columns))
    seqused_k = video_pages * page_size + text_len

    num_pages = -(-num_tokens // page_size)
    return Magi2WindowAttentionPlan(
        grid=grid,
        text_len=text_len,
        block_grid=block_grid,
        page_size=page_size,
        cu_seqlens_q=torch.cat((starts, torch.tensor([num_video]))).to(torch.int32),
        page_table=page_table.to(torch.int32).contiguous(),
        seqused_k=seqused_k.to(torch.int32),
        max_seqlen_q=int(sizes.max()),
        max_seqlen_k=int(seqused_k.max()),
        text_cu_seqlens_q=torch.tensor([0, text_len], dtype=torch.int32),
        text_page_table=torch.arange(num_pages, dtype=torch.int32)[None, :],
        text_seqused_k=torch.tensor([num_tokens], dtype=torch.int32),
    )


def torch_window_attention_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    plan: Magi2WindowAttentionPlan,
    *,
    softmax_scale: float | None = None,
    max_tokens: int = TORCH_REFERENCE_MAX_TOKENS,
) -> torch.Tensor:
    """Evaluate a window-attention plan with one fp32 softmax per query row.

    ``query`` is ``[N, Hq, D]`` and ``key``/``value`` are ``[N, Hkv, D]`` in
    block-major order; query head ``h`` reads key/value head
    ``h // (Hq // Hkv)``.  Keys are gathered through the plan's pages exactly
    as a paged kernel reads them.  The output has the query's dtype.
    """

    if query.ndim != 3 or key.ndim != 3 or key.shape != value.shape:
        raise ValueError("expected query [N, Hq, D] and key/value [N, Hkv, D]")
    num_tokens, num_heads_q, head_dim = query.shape
    num_heads_kv = key.shape[1]
    if num_tokens != plan.num_tokens or key.shape[0] != num_tokens or key.shape[2] != head_dim:
        raise ValueError(
            f"window attention inputs do not match the plan: query {tuple(query.shape)}, "
            f"key {tuple(key.shape)}, plan tokens {plan.num_tokens}"
        )
    if num_heads_q % num_heads_kv:
        raise ValueError("query heads must be a multiple of key/value heads")
    if num_tokens > max_tokens:
        raise ValueError(
            f"the PyTorch window-attention reference is limited to {max_tokens} tokens "
            f"(got {num_tokens}); use the paged FlashAttention kernels for production sizes"
        )
    scale = head_dim**-0.5 if softmax_scale is None else softmax_scale
    group = num_heads_q // num_heads_kv
    key_pages = plan.pad_to_pages(key.float())
    value_pages = plan.pad_to_pages(value.float())

    rows = [
        (int(plan.cu_seqlens_q[b]), int(plan.cu_seqlens_q[b + 1]), plan.page_table[b], int(plan.seqused_k[b]))
        for b in range(plan.num_blocks)
    ]
    if plan.text_len:
        rows.append((plan.num_video_tokens, plan.num_tokens, plan.text_page_table[0], int(plan.text_seqused_k[0])))
    out = torch.empty_like(query)
    for q_start, q_end, pages, used in rows:
        pages = pages[: -(-used // plan.page_size)].to(device=key_pages.device, dtype=torch.long)
        k = key_pages[pages].flatten(0, 1)[:used].repeat_interleave(group, dim=1)
        v = value_pages[pages].flatten(0, 1)[:used].repeat_interleave(group, dim=1)
        scores = torch.einsum("qhd,khd->hqk", query[q_start:q_end].float(), k) * scale
        out[q_start:q_end] = torch.einsum("hqk,khd->qhd", scores.softmax(dim=-1), v).to(out.dtype)
    return out


def window_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    plan: Magi2WindowAttentionPlan,
    *,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """Run a plan's window attention over its full token sequence.

    CPU tensors run ``torch_window_attention_reference``.  Device tensors need a
    paged FlashAttention kernel, which this function does not provide, so they
    are refused instead of being routed to the host-bound reference.
    """

    if query.device.type != "cpu":
        raise NotImplementedError(
            f"MAGI-2 refiner window attention has no paged kernel for {query.device.type} tensors"
        )
    return torch_window_attention_reference(query, key, value, plan, softmax_scale=softmax_scale)


def ulysses_window_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    plan: Magi2WindowAttentionPlan,
    split_sizes: Sequence[int],
    *,
    softmax_scale: float | None = None,
    group: Magi2ParallelGroup | None = None,
) -> torch.Tensor:
    """Run window attention for one Ulysses rank's contiguous token slice.

    ``split_sizes`` holds every rank's token count of the plan's sequence.  The
    plan is global, so a sliced sequence needs the Ulysses sequence/head
    exchange around the kernel; this function does not perform it and refuses
    more than one rank.
    """

    group = group or get_magi2_ulysses_group()
    split_sizes = [int(size) for size in split_sizes]
    if len(split_sizes) != group.world_size or sum(split_sizes) != plan.num_tokens:
        raise ValueError(
            f"split_sizes={split_sizes} do not partition the plan's {plan.num_tokens} tokens "
            f"across {group.world_size} Ulysses ranks"
        )
    if query.shape[0] != split_sizes[group.rank]:
        raise ValueError(f"Ulysses rank {group.rank} owns {split_sizes[group.rank]} tokens, got {query.shape[0]}")
    if group.world_size > 1:
        raise NotImplementedError("MAGI-2 refiner window attention does not implement the Ulysses exchange")
    return window_attention(query, key, value, plan, softmax_scale=softmax_scale)


class Magi2WindowAttentionKernel(nn.Module):
    """Refiner window attention behind the shared diffusion ``Attention`` contract.

    ``attn_metadata.extra`` carries the request's ``magi2_window_plan`` and the
    per-rank token counts ``magi2_split_sizes`` of its sequence.
    """

    def __init__(self, softmax_scale: float) -> None:
        super().__init__()
        self.softmax_scale = softmax_scale

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        if attn_metadata is None:
            raise ValueError("MAGI-2 refiner window attention requires attention metadata")
        plan = attn_metadata.extra.get("magi2_window_plan")
        split_sizes = attn_metadata.extra.get("magi2_split_sizes")
        if not isinstance(plan, Magi2WindowAttentionPlan):
            raise TypeError("magi2_window_plan must be a Magi2WindowAttentionPlan")
        if not isinstance(split_sizes, (list, tuple)):
            raise TypeError("magi2_split_sizes must be a list or tuple of per-rank token counts")
        return ulysses_window_attention(query, key, value, plan, split_sizes, softmax_scale=self.softmax_scale)
