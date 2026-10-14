# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Copyright (c) 2026 SandAI. All Rights Reserved.

"""Token packing for the MAGI-2 refiner transformer.

One refiner forward sees a single sample as one sequence: the video latent's
tokens in block-major order, followed by the text tokens in their own order.
The token layout, the 9-D coordinates and the Ulysses split are adapted from
SandAI's Apache-2.0 MAGI-2 refiner data proxy.  Video tokens are taken from
the latent with an exact rearrange (one token per latent position, patch size
1), and the proxy keeps no per-request state: ``unpack`` only needs the grid.
"""

from __future__ import annotations

import dataclasses
import math
from collections import OrderedDict
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass
from itertools import accumulate
from typing import Any, TypeVar

import torch

from .configuration_magi2 import MAGI2_REFINER_CONFIG, Magi2RefinerConfig, Magi2RefinerWindowConfig
from .parallel import balanced_split_sizes
from .preview_data_proxy import Modality
from .window_attention import (
    PAGE_SIZE,
    Magi2WindowAttentionPlan,
    block_major_order,
    build_window_attention_plan,
)


@dataclass(frozen=True, eq=False)
class Magi2RefinerInput:
    """One refiner forward's tokens, or one Ulysses rank's contiguous slice of them.

    The sequence that ``window_plan`` describes is the block-major video tokens
    followed by the text tokens.  ``video`` ``[Nv, Cv]`` and ``text``
    ``[Nt, Ct]`` are the fp32 features of the slice's video and text tokens;
    ``coords`` ``[n, 9]`` (fp32) and ``modality`` ``[n]`` (int64) describe its
    ``n`` tokens in order.  ``split_sizes`` holds every rank's token count and
    ``rank`` selects the slice; an unsliced input has one rank.
    """

    video: torch.Tensor
    text: torch.Tensor
    coords: torch.Tensor
    modality: torch.Tensor
    window_plan: Magi2WindowAttentionPlan
    split_sizes: tuple[int, ...]
    rank: int = 0

    @property
    def grid(self) -> tuple[int, int, int]:
        return self.window_plan.grid

    @property
    def video_split_sizes(self) -> list[int]:
        """Every rank's video token count, for gathering per-rank video outputs."""

        num_video = self.window_plan.num_video_tokens
        bounds = [0, *accumulate(self.split_sizes)]
        return [min(end, num_video) - min(start, num_video) for start, end in zip(bounds, bounds[1:])]

    def shard(self, rank: int, world_size: int) -> Magi2RefinerInput:
        """Return Ulysses rank ``rank``'s contiguous slice of an unsliced input.

        The sequence is split evenly; the first ``N % world_size`` ranks hold
        one extra token.
        """

        if len(self.split_sizes) != 1:
            raise ValueError("only an unsliced MAGI-2 refiner input can be sharded")
        if not 0 <= rank < world_size:
            raise ValueError(f"rank must be in [0, {world_size}), got {rank}")
        split_sizes = balanced_split_sizes(self.window_plan.num_tokens, world_size)
        start = sum(split_sizes[:rank])
        end = start + split_sizes[rank]
        num_video = self.window_plan.num_video_tokens
        text_start, text_end = max(start, num_video) - num_video, max(end, num_video) - num_video
        return dataclasses.replace(
            self,
            video=self.video[min(start, num_video) : min(end, num_video)],
            text=self.text[text_start:text_end],
            coords=self.coords[start:end],
            modality=self.modality[start:end],
            split_sizes=tuple(split_sizes),
            rank=rank,
        )

    def model_kwargs(self) -> dict[str, Any]:
        """Keyword arguments of ``Magi2RefinerTransformer.forward``."""

        return {
            "video": self.video,
            "text": self.text,
            "coords": self.coords,
            "modality": self.modality,
            "window_plan": self.window_plan,
            "split_sizes": list(self.split_sizes),
        }


_T = TypeVar("_T")

_CACHE_SIZE = 8


def _cached(cache: OrderedDict[Hashable, _T], key: Hashable, build: Callable[[], _T]) -> _T:
    """Return ``cache[key]``, building it on a miss and evicting the least recently used entry."""

    if key in cache:
        cache.move_to_end(key)
        return cache[key]
    # Cached tensors outlive the call, so they are built as normal tensors even
    # under inference mode and stay usable by grad-enabled callers.
    with torch.inference_mode(False):
        value = build()
    cache[key] = value
    if len(cache) > _CACHE_SIZE:
        cache.popitem(last=False)
    return value


def _build_block_order(
    grid: tuple[int, int, int], window: Magi2RefinerWindowConfig, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    order = block_major_order(grid, window)
    restore = torch.empty_like(order)
    restore[order] = torch.arange(order.numel())
    return order.to(device), restore.to(device)


def _build_window_plan(
    grid: tuple[int, int, int],
    text_len: int,
    window: Magi2RefinerWindowConfig,
    page_size: int,
    device: torch.device,
) -> Magi2WindowAttentionPlan:
    plan = build_window_attention_plan(grid, text_len, window, page_size=page_size)
    return plan if device.type == "cpu" else plan.to(device)


def _video_coords(grid: tuple[int, int, int], device: torch.device) -> torch.Tensor:
    """Raster-order ``(t, h, w, T, H, W, T, H, W)`` coordinates of the video tokens.

    The reference size equals the grid size (extrapolated spatial RoPE).
    """

    axes = (torch.arange(n, dtype=torch.float32, device=device) for n in grid)
    positions = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, 3)
    sizes = torch.tensor((*grid, *grid), dtype=torch.float32, device=device)
    return torch.cat((positions, sizes.expand(positions.shape[0], -1)), dim=-1)


def _text_coords(text_len: int, device: torch.device) -> torch.Tensor:
    """``(t, 0, 0, L, 1, 1, 1, 1, 1)`` coordinates with ``t = -L .. -1``."""

    t = torch.arange(text_len, dtype=torch.float32, device=device) - text_len
    positions = torch.stack((t, torch.zeros_like(t), torch.zeros_like(t)), dim=-1)
    sizes = torch.tensor((text_len, 1, 1, 1, 1, 1), dtype=torch.float32, device=device)
    return torch.cat((positions, sizes.expand(text_len, -1)), dim=-1)


class Magi2RefinerDataProxy:
    """Pack refiner forwards into block-major token sequences and unpack velocities.

    Block orders and window plans are cached on the instance, at most eight of
    each, and are freed with it.
    """

    def __init__(self, config: Magi2RefinerConfig = MAGI2_REFINER_CONFIG, *, page_size: int = PAGE_SIZE) -> None:
        self.config = config
        self.page_size = page_size
        self._block_orders: OrderedDict[Hashable, tuple[torch.Tensor, torch.Tensor]] = OrderedDict()
        self._window_plans: OrderedDict[Hashable, Magi2WindowAttentionPlan] = OrderedDict()

    def block_order(self, grid: Sequence[int], device: torch.device | str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
        """Return the block-major order of the raster video tokens and its inverse.

        ``order[i]`` is the raster index of the token at block-major position
        ``i``; ``restore[order] == arange``.  Both are int64.
        """

        key = (tuple(int(n) for n in grid), torch.device(device))
        return _cached(self._block_orders, key, lambda: _build_block_order(key[0], self.config.window, key[1]))

    def window_plan(
        self,
        grid: Sequence[int],
        text_len: int,
        device: torch.device | str = "cpu",
    ) -> Magi2WindowAttentionPlan:
        """Return the window-attention plan of a ``(T, H, W)`` grid and text length on ``device``.

        The plan is built on the first request and reused by every later
        call and by ``pack``.
        """

        key = (tuple(int(n) for n in grid), int(text_len), torch.device(device))
        return _cached(
            self._window_plans,
            key,
            lambda: _build_window_plan(key[0], key[1], self.config.window, self.page_size, key[2]),
        )

    def pack(self, latent: torch.Tensor, text: torch.Tensor) -> Magi2RefinerInput:
        """Pack a ``[1, Cv, T, H, W]`` latent and ``[L, Ct]`` (or ``[1, L, Ct]``) text features.

        ``L`` may be zero.  Features are converted to fp32 on the latent's
        device; the window plan is built once per grid, text length and device.
        """

        cfg = self.config
        if latent.ndim != 5 or latent.shape[0] != 1 or latent.shape[1] != cfg.video_in_channels:
            raise ValueError(
                f"MAGI-2 refiner packs one [1, {cfg.video_in_channels}, T, H, W] latent, got {tuple(latent.shape)}"
            )
        if text.ndim == 3 and text.shape[0] == 1:
            text = text[0]
        if text.ndim != 2 or text.shape[1] != cfg.text_in_channels:
            raise ValueError(
                f"MAGI-2 refiner text features must be [L, {cfg.text_in_channels}] or "
                f"[1, L, {cfg.text_in_channels}], got {tuple(text.shape)}"
            )
        device = latent.device
        grid = tuple(int(n) for n in latent.shape[2:])
        num_video, text_len = math.prod(grid), text.shape[0]
        window_plan = self.window_plan(grid, text_len, device)
        order, _ = self.block_order(grid, device)

        # '1 C T H W -> (T H W) C', then block-major.
        video = latent[0].to(torch.float32).flatten(1).t().index_select(0, order)
        coords = torch.cat((_video_coords(grid, device).index_select(0, order), _text_coords(text_len, device)))
        modality = torch.cat(
            (
                torch.full((num_video,), int(Modality.VIDEO), dtype=torch.int64, device=device),
                torch.full((text_len,), int(Modality.TEXT), dtype=torch.int64, device=device),
            )
        )
        return Magi2RefinerInput(
            video=video,
            text=text.to(device=device, dtype=torch.float32),
            coords=coords,
            modality=modality,
            window_plan=window_plan,
            split_sizes=(num_video + text_len,),
        )

    def unpack(self, velocity: torch.Tensor, grid: Sequence[int]) -> torch.Tensor:
        """Return block-major ``[V, Cv]`` video velocities as a ``[1, Cv, T, H, W]`` latent."""

        grid = tuple(int(n) for n in grid)
        expected = (math.prod(grid), self.config.video_in_channels)
        if tuple(velocity.shape) != expected:
            raise ValueError(f"expected a {list(expected)} block-major velocity, got {tuple(velocity.shape)}")
        _, restore = self.block_order(grid, velocity.device)
        # '(T H W) C -> 1 C T H W' after restoring raster order.
        raster = velocity.index_select(0, restore)
        return raster.t().reshape(1, expected[1], *grid).contiguous()


__all__ = [
    "Magi2RefinerDataProxy",
    "Magi2RefinerInput",
]
