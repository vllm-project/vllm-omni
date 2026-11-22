# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Full KV archive and reusable, exact selected-history attention staging."""

from __future__ import annotations

from typing import Any

import torch

from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.dense_attention import CosmosSimDenseAttentionCache


class FrustumKVArchive:
    def __init__(
        self,
        owner: Any,
        text_kv: list[tuple[torch.Tensor, torch.Tensor]],
        text_length: int,
        tokens_per_frame: int,
        horizon: int,
        budget: int,
        chunk_size: int,
        attention: CosmosSimDenseAttentionCache,
        paged_state: Any = None,
        branch: str = "main",
    ) -> None:
        self.attention = attention
        self.tokens_per_frame = tokens_per_frame
        self.paged_state = paged_state
        self.branch = branch
        self.frames = 0
        self.chunk_start = 0
        self.selected: list[int] = []
        self.prepared: tuple[int, ...] | None = None
        attention.activate(owner, text_kv, text_length, text_length + (budget + chunk_size) * tokens_per_frame)
        self.archive = (
            []
            if paged_state is not None
            else [
                tuple(
                    torch.empty((1, horizon * tokens_per_frame, *t.shape[2:]), device=t.device, dtype=t.dtype)
                    for t in pair
                )
                for pair in text_kv
            ]
        )

    def select(self, start: int, ids: list[int]) -> None:
        if (
            start != self.frames
            or ids != sorted(set(ids))
            or any(type(i) is not int or not 0 <= i < start for i in ids)
        ):
            raise ValueError("Retrieval must select ordered unique past frames at the archive frontier")
        self.chunk_start = start
        self.selected = list(ids)
        self.prepared = None

    @torch.no_grad()
    def prepare(self, frame_start: int) -> None:
        # For framewise clean refresh, include earlier finalized frames of this
        # same chunk without re-scoring or replacing any selected older frames.
        ids = tuple(self.selected + list(range(self.chunk_start, frame_start)))
        if frame_start != self.frames or any(i >= self.frames for i in ids):
            raise ValueError("Retrieval cannot read uncommitted or future K/V")
        if ids == self.prepared:
            return
        count = self.tokens_per_frame
        positions = [f * count + i for f in ids for i in range(count)]
        offset = self.attention.text_length
        if self.paged_state is not None:
            self.paged_state.gather_history_into(self.branch, positions, self.attention.kv, offset)
        elif positions:
            indices = torch.tensor(positions, dtype=torch.long, device=self.archive[0][0].device)
            for pair, buffers in zip(self.archive, self.attention.kv, strict=True):
                for source, dest in zip(pair, buffers, strict=True):
                    torch.index_select(source[0], 0, indices, out=dest[0, offset : offset + len(positions)])
        self.attention.history_length = len(positions)
        self.prepared = ids

    @torch.no_grad()
    def commit(self, frames: int) -> None:
        begin = self.attention.text_length + self.attention.history_length
        count = frames * self.tokens_per_frame
        current = [(k[:, begin : begin + count], v[:, begin : begin + count]) for k, v in self.attention.kv]
        if self.paged_state is not None:
            self.paged_state.append_history(self.branch, current)
        else:
            start = self.frames * self.tokens_per_frame
            for pair, sources in zip(self.archive, current, strict=True):
                for dest, source in zip(pair, sources, strict=True):
                    dest[:, start : start + count].copy_(source)
        self.frames += frames
        self.prepared = None
