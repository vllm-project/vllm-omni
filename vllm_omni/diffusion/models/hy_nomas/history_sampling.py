# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (C) 2026 Tencent. All rights reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Archived selection, adapted from Hy-Nomas 911cea6458 uniform_history_cache.

Video-frame budgets exclude text. Each represented prompt segment contributes
one text block, even when its first chunk is absent from the selected grid.
Archived post-RoPE rows are copied without recomputing their absolute positions.
"""

import math
from collections.abc import Iterator
from dataclasses import dataclass
from numbers import Real

import numpy as np
import torch

from .history_archive import HyNomasArchivedChunk

ARCHIVED_HISTORY_MODES = ("uniform", "recent_mix_uniform", "temporal_context")


def history_frame_indices(total_frames: int, budget: int, mode: str, *, gamma: float = 2.0) -> tuple[int, ...]:
    if type(total_frames) is not int or total_frames < 0 or type(budget) is not int or budget < 1:
        raise ValueError("History requires nonnegative total frames and a positive integer budget")
    if mode not in ARCHIVED_HISTORY_MODES:
        raise ValueError(f"Unknown archived history mode: {mode!r}")
    if mode == "temporal_context" and (
        isinstance(gamma, bool) or not isinstance(gamma, Real) or not math.isfinite(gamma) or gamma <= 0
    ):
        raise ValueError("Temporal context gamma must be finite and positive")
    count = min(total_frames, budget)
    if count == 0:
        return ()
    if count == 1:
        return (total_frames - 1,)
    if count == total_frames:
        return tuple(range(total_frames))
    if mode == "uniform":
        return tuple(i * (total_frames - 1) // (count - 1) for i in range(count))
    if mode == "recent_mix_uniform":
        recent = budget // 2
        end = total_frames - recent
        return history_frame_indices(end, budget - recent, "uniform") + tuple(range(end, total_frames))
    slack = total_frames - count
    rounded = np.rint(slack * (np.arange(count) / (count - 1)) ** gamma)
    offsets = (0, *(min(slack, max(0, int(value))) for value in rounded[1:-1]), slack)
    return tuple(total_frames - 1 - (index + offsets[index]) for index in reversed(range(count)))


@dataclass(frozen=True)
class SelectedFrame:
    position: int
    video: HyNomasArchivedChunk
    text: HyNomasArchivedChunk | None = None

    @property
    def text_tokens(self) -> int:
        return self.text.text_tokens if self.text is not None else 0

    @property
    def num_tokens(self) -> int:
        return self.video.tokens_per_frame + self.text_tokens

    def pieces(self, layer: int | None = None, value: bool = False) -> list[torch.Tensor]:
        """Borrow CPU slices; callers own only the bounded packed working set."""
        rows = self.video.frame_rows(self.position)
        if layer is None:
            pieces = [self.video.mask[:, rows]]
        else:
            pieces = [self.video.layers[layer][int(value)][rows]]
        if self.text is not None:
            start = self.text.frame_count * self.text.tokens_per_frame
            pieces.append(self.text.mask[:, start:] if layer is None else self.text.layers[layer][int(value)][start:])
        return pieces


def select_history(records: tuple[HyNomasArchivedChunk, ...], positions: tuple[int, ...]) -> list[SelectedFrame]:
    """Pack chronological video followed by one text block per prompt segment."""
    if not records or not positions or any(a >= b for a, b in zip(positions, positions[1:])):
        raise ValueError("Archive selection requires nonempty chronological frames")
    text = {record.segment: record for record in records if record.text_tokens}
    selected: list[SelectedFrame] = []
    index = 0
    for position in positions:
        while index < len(records) and position >= records[index].frame_start + records[index].frame_count:
            index += 1
        if index == len(records) or position < records[index].frame_start:
            raise ValueError("Selected frame is outside the archive")
        selected.append(SelectedFrame(position, records[index]))
    for index, frame in enumerate(selected):
        if index + 1 == len(selected) or selected[index + 1].video.segment != frame.video.segment:
            selected[index] = SelectedFrame(frame.position, frame.video, text.get(frame.video.segment))
    return selected


def selected_layers(selected: list[SelectedFrame], num_layers: int) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Materialize one CPU layer at a time, preserving the archived KV dtype."""
    for layer in range(num_layers):
        key = torch.cat([part for frame in selected for part in frame.pieces(layer)], 0)
        value = torch.cat([part for frame in selected for part in frame.pieces(layer, True)], 0)
        yield key, value


def selected_mask(selected: list[SelectedFrame]) -> torch.Tensor:
    return torch.cat([part for frame in selected for part in frame.pieces()], 1)
