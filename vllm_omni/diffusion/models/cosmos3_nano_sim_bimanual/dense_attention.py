# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Stable joint-attention storage for Sim's full-history dense execution."""

import weakref

import torch


class CosmosSimDenseAttentionCache:
    """Keep text, committed history and current K/V in one contiguous buffer.

    Sim admits one dense session at a time. Buffers survive session reset so
    regional CUDA graphs can reuse their addresses; the next activation writes
    new conditioning and resets the visible history. Capacity follows the
    requested rollout, not the deployment's maximum history horizon.

    Sliding history uses the ordinary dense path. This cache never evicts or
    changes attention order: every view is exactly text + history + current.
    """

    def __init__(self):
        self.kv: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.text_length = 0
        self.history_length = 0
        self._owner = None
        self._text_source = None
        self._rope_buffers = {}
        self._rope_geometry = None

    def stage_rope(self, cos, sin, geometry):
        """Copy changing positions once, rather than into every regional graph.

        Stable base allocations remain inputs, not constants: every forward
        refreshes their values before any block runs on the same CUDA stream.
        Keep only the current geometry's chunk shapes across requests.
        """
        if geometry != self._rope_geometry:
            self._rope_buffers.clear()
            self._rope_geometry = geometry
        key = (cos.shape, cos.dtype, cos.device)
        if key not in self._rope_buffers:
            pair = (torch.empty_like(cos), torch.empty_like(sin))
            for buffer in pair:
                torch._dynamo.mark_static_address(buffer)
            self._rope_buffers[key] = pair
        buffers = self._rope_buffers[key]
        for buffer, source in zip(buffers, (cos, sin)):
            buffer.copy_(source)
        return buffers

    def fits(self, text_kv, capacity: int) -> bool:
        return len(self.kv) == len(text_kv) and all(
            buffer.shape[1] >= capacity
            and buffer.shape[2:] == source.shape[2:]
            and buffer.dtype == source.dtype
            and buffer.device == source.device
            for buffers, sources in zip(self.kv, text_kv)
            for buffer, source in zip(buffers, sources)
        )

    @torch.no_grad()
    def activate(self, owner, text_kv, text_length: int, capacity: int, history=None):
        history_length = 0 if history is None else history[0][0].shape[1]
        if not text_kv or text_length <= 0 or capacity < text_length + history_length:
            raise ValueError("Dense joint-attention capacity must contain text and retained history")
        if not self.fits(text_kv, capacity):
            self.kv = [
                tuple(
                    torch.empty((1, capacity, *source.shape[2:]), dtype=source.dtype, device=source.device)
                    for source in pair
                )
                for pair in text_kv
            ]
            for pair in self.kv:
                for buffer in pair:
                    torch._dynamo.mark_static_address(buffer)
            self._owner = None
        if (
            self._owner is not None
            and self._owner() is owner
            and self._text_source() is text_kv[0][0]
            and self.text_length == text_length
            and self.history_length == history_length
        ):
            return
        for layer, (buffers, text) in enumerate(zip(self.kv, text_kv)):
            for index, (buffer, source) in enumerate(zip(buffers, text)):
                buffer[:, :text_length].copy_(source[:, :text_length])
                if history_length:
                    buffer[:, text_length : text_length + history_length].copy_(history[layer][index])
        self.text_length = text_length
        self.history_length = history_length
        self._owner = weakref.ref(owner)
        self._text_source = weakref.ref(text_kv[0][0])

    def commit(self, tokens: int):
        """Publish current K/V already written by every layer, without copies."""
        end = self.text_length + self.history_length + tokens
        if tokens <= 0 or not self.kv or end > self.kv[0][0].shape[1]:
            raise ValueError("Dense joint-attention commit exceeds its reserved capacity")
        self.history_length += tokens
        return [(k[:, self.text_length : end], v[:, self.text_length : end]) for k, v in self.kv]
