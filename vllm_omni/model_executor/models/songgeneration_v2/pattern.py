# Licensed under the License Terms of SongGeneration (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/tencent-ailab/SongGeneration/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License. The License Terms of SongGeneration permit
# academic / research / educational use only and prohibit commercial use.
# ==============================================================================
#
# Copyright (C) 2025 Tencent. All rights reserved.
# Modifications Copyright contributors to the vLLM-Omni project.
#
# Vendored from tencent-ailab/SongGeneration (codeclm/modules/pattern.py),
# adapted for inference-only use in vLLM-Omni.
"""Delayed codebook pattern for SongGeneration v2 LeLM Stage 0.

Vendored from upstream ``codeclm/modules/pattern.py``.
Only ``DelayedPatternProvider`` is used; default delays=[0, 250, 250].
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch


@dataclass
class LayoutCoord:
    """A single (timestep, codebook) coordinate in the pattern layout."""

    t: int
    q: int


PatternLayout = list[list[LayoutCoord]]


class Pattern:
    """Pattern materialised for a fixed number of ``timesteps``.

    ``build_pattern_sequence`` maps dense codes ``[B, K, T]`` to the
    interleaved pattern sequence ``[B, K, S]``. ``revert_pattern_sequence``
    is the inverse used to undo delays after generation.
    """

    def __init__(self, layout: PatternLayout, code_depth: int, timesteps: int):
        assert len(layout) > 0
        self.code_depth = code_depth
        self.timesteps = timesteps
        self.layout = layout
        self._valid_layout = [seq for seq in layout if all(c.t < timesteps for c in seq)]
        # cache scatter-index builds; lengths are the same across calls
        self._build_pattern_sequence_scatter_indexes = lru_cache(100)(self._build_pattern_sequence_scatter_indexes)
        self._build_reverted_sequence_scatter_indexes = lru_cache(100)(self._build_reverted_sequence_scatter_indexes)

    @property
    def valid_layout(self) -> PatternLayout:
        return self._valid_layout

    def get_sequence_coords_with_timestep(self, t: int, q: int | None = None):
        coords = []
        for s, seq_codes in enumerate(self.layout):
            for code in seq_codes:
                if code.t == t and (q is None or code.q == q):
                    coords.append((s, code))
        return coords

    def get_steps_with_timestep(self, t: int, q: int | None = None) -> list[int]:
        return [step for step, _ in self.get_sequence_coords_with_timestep(t, q)]

    def get_first_step_with_timesteps(self, t: int, q: int | None = None) -> int | None:
        steps = self.get_steps_with_timestep(t, q)
        return steps[0] if steps else None

    def _build_pattern_sequence_scatter_indexes(
        self,
        timesteps: int,
        code_depth: int,
        keep_only_valid_steps: bool,
        device: torch.device | str = "cpu",
    ):
        assert code_depth == self.code_depth, f"codebook mismatch: {code_depth} != {self.code_depth}"
        assert timesteps <= self.timesteps
        ref_layout = self.valid_layout if keep_only_valid_steps else self.layout
        indexes = np.zeros((code_depth, len(ref_layout)), dtype=np.int64)
        mask = np.zeros((code_depth, len(ref_layout)), dtype=bool)
        indexes[:] = code_depth * timesteps  # sentinel -> special token slot
        for s, sequence_coords in enumerate(ref_layout):
            for coords in sequence_coords:
                if coords.t < timesteps:
                    indexes[coords.q, s] = coords.t + coords.q * timesteps
                    mask[coords.q, s] = 1
        return torch.from_numpy(indexes).to(device), torch.from_numpy(mask).to(device)

    def build_pattern_sequence(self, z: torch.Tensor, special_token: int, keep_only_valid_steps: bool = False):
        """``[B, K, T]`` dense codes -> ``[B, K, S]`` pattern sequence."""
        B, K, T = z.shape
        indexes, mask = self._build_pattern_sequence_scatter_indexes(
            T, K, keep_only_valid_steps=keep_only_valid_steps, device=str(z.device)
        )
        z = z.reshape(B, -1)
        z = torch.cat([z, torch.zeros_like(z[:, :1]) + special_token], dim=1)
        values = z[:, indexes.view(-1)].view(B, K, indexes.shape[-1])
        return values, indexes, mask

    def _build_reverted_sequence_scatter_indexes(
        self,
        sequence_steps: int,
        code_depth: int,
        keep_only_valid_steps: bool = False,
        is_model_output: bool = False,
        device: torch.device | str = "cpu",
    ):
        ref_layout = self.valid_layout if keep_only_valid_steps else self.layout
        timesteps = self.timesteps
        assert code_depth == self.code_depth
        assert sequence_steps <= len(ref_layout)
        if is_model_output:
            ref_layout = ref_layout[1:]
        indexes = np.zeros((code_depth, timesteps), dtype=np.int64)
        mask = np.zeros((code_depth, timesteps), dtype=bool)
        indexes[:] = code_depth * sequence_steps
        for s, sequence_codes in enumerate(ref_layout):
            if s < sequence_steps:
                for code in sequence_codes:
                    if code.t < timesteps:
                        indexes[code.q, code.t] = s + code.q * sequence_steps
                        mask[code.q, code.t] = 1
        return torch.from_numpy(indexes).to(device), torch.from_numpy(mask).to(device)

    def revert_pattern_sequence(self, s: torch.Tensor, special_token: int, keep_only_valid_steps: bool = False):
        """``[B, K, S]`` pattern sequence -> dense ``[B, K, T]``."""
        B, K, S = s.shape
        indexes, mask = self._build_reverted_sequence_scatter_indexes(
            S, K, keep_only_valid_steps, is_model_output=False, device=str(s.device)
        )
        s = s.view(B, -1)
        s = torch.cat([s, torch.zeros_like(s[:, :1]) + special_token], dim=1)
        values = s[:, indexes.view(-1)].view(B, K, indexes.shape[-1])
        return values, indexes, mask

    def revert_pattern_logits(self, logits: torch.Tensor, special_token: float, keep_only_valid_steps: bool = False):
        """``[B, card, K, S]`` pattern logits -> ``[B, card, K, T]`` dense."""
        B, card, K, S = logits.shape
        indexes, mask = self._build_reverted_sequence_scatter_indexes(
            S, K, keep_only_valid_steps, is_model_output=True, device=str(logits.device)
        )
        logits = logits.reshape(B, card, -1)
        logits = torch.cat([logits, torch.zeros_like(logits[:, :, :1]) + special_token], dim=-1)
        values = logits[:, :, indexes.view(-1)].view(B, card, K, indexes.shape[-1])
        return values, indexes, mask


class CodebooksPatternProvider(ABC):
    def __init__(self, code_depth: int):
        assert code_depth > 0
        self.code_depth = code_depth
        self.get_pattern = lru_cache(100)(self.get_pattern)  # type: ignore[assignment]

    @abstractmethod
    def get_pattern(self, timesteps: int) -> Pattern:
        raise NotImplementedError


class DelayedPatternProvider(CodebooksPatternProvider):
    """Delays codebook q by ``delays[q]`` timesteps.

    With ``delays=[0, 250, 250]`` and 3 streams, stream 0 is emitted from
    t=0 while streams 1 and 2 are delayed by 250 sequence steps each.
    """

    def __init__(
        self,
        code_depth: int,
        delays: list[int] | None = None,
        flatten_first: int = 0,
        empty_initial: int = 0,
    ):
        super().__init__(code_depth)
        if delays is None:
            delays = list(range(code_depth))
        self.delays = delays
        self.flatten_first = flatten_first
        self.empty_initial = empty_initial
        assert len(self.delays) == self.code_depth
        assert sorted(self.delays) == self.delays

    def get_pattern(self, timesteps: int) -> Pattern:
        out: PatternLayout = [[]]
        max_delay = max(self.delays)
        if self.empty_initial:
            out += [[] for _ in range(self.empty_initial)]
        if self.flatten_first:
            for t in range(min(timesteps, self.flatten_first)):
                for q in range(self.code_depth):
                    out.append([LayoutCoord(t, q)])
        for t in range(self.flatten_first, timesteps + max_delay):
            v = []
            for q, delay in enumerate(self.delays):
                t_for_q = t - delay
                if t_for_q >= self.flatten_first:
                    v.append(LayoutCoord(t_for_q, q))
            out.append(v)
        return Pattern(out, code_depth=self.code_depth, timesteps=timesteps)


__all__ = ["LayoutCoord", "Pattern", "CodebooksPatternProvider", "DelayedPatternProvider"]
