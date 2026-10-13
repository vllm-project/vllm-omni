# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Request-local reference implementation of Solar's transactional raw KV."""

from __future__ import annotations

import torch


class SolarRawKVCache:
    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self.layers: dict[int, list[tuple[int, torch.Tensor, torch.Tensor]]] = {}
        self.pending: dict[int, tuple[int, torch.Tensor, torch.Tensor]] = {}
        self.chunk_index = 0

    def history(self, layer: int, chunk_index: int) -> list[tuple[int, torch.Tensor, torch.Tensor]]:
        if chunk_index != self.chunk_index or not 0 <= layer < self.num_layers:
            raise ValueError("Solar KV layer or chunk index differs from committed history")
        return self.layers.get(layer, [])

    def stage(self, layer: int, chunk_index: int, key: torch.Tensor, value: torch.Tensor) -> None:
        self.history(layer, chunk_index)
        if layer in self.pending or key.shape != value.shape or key.ndim != 4 or key.shape[0] != 1:
            raise ValueError("Invalid or duplicate Solar KV stage")
        # A view would retain the full packed QKV allocation, including prefix
        # and unused projections. Own just this chunk's raw video rows.
        self.pending[layer] = (chunk_index, key.detach().clone(), value.detach().clone())

    def commit_chunk(self, chunk_index: int) -> None:
        if chunk_index != self.chunk_index or set(self.pending) != set(range(self.num_layers)):
            raise RuntimeError("Solar KV commit requires every layer of the current chunk")
        for layer, entry in self.pending.items():
            self.layers[layer] = [*self.layers.get(layer, []), entry][-5:]
        self.pending.clear()
        self.chunk_index += 1

    def abort(self) -> None:
        self.pending.clear()

    def clear(self) -> None:
        self.layers.clear()
        self.pending.clear()
        self.chunk_index = 0
