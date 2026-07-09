# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Concrete memory objects (RFC #4480).

Both classes back their storage with plain (monolithic) buffers. Attention KV
is deliberately absent: for DreamZero it is owned by the AR-Diffusion engine's
paged pool (PR #4534), and the RFC's ``PagedKV`` arrives in Phase 1 as a
handle wrapping that engine state rather than a dense buffer here. The RFC's
``FixedState`` and ``RetrievalStore`` land with their first consumers, but the
names are reserved.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from typing import TypeVar

import numpy as np
import torch

from vllm_omni.experimental.world_models.memory.base import MemoryObject

# The live ``{"is_init", "k", "v"}`` dict an encode-once cross-attention layer
# populates on the first forward and reads thereafter.
CrossKVCache = dict[str, bool | torch.Tensor | None]

# The frame/chunk type a ``LatentBuffer`` holds (e.g. ``np.ndarray`` pixel
# frames, ``torch.Tensor`` latent chunks).
ItemT = TypeVar("ItemT")


class EncodeOnceKV(MemoryObject[CrossKVCache, CrossKVCache]):
    """Encode-once cross-attention KV.

    Wraps an ``{"is_init", "k", "v"}`` dict that a model's cross-attention
    layers populate once (on the first forward) and read thereafter. ``view()``
    returns the live dict so the model mutates it in place. DreamZero's
    cross-attn KV moved into the AR-Diffusion engine pool (PR #4534); the
    in-tree consumer of this class is the Cosmos3 UND text K/V port.
    """

    def __init__(self) -> None:
        self._cache: CrossKVCache | None = None

    def allocate(self) -> None:
        self._cache = {"is_init": False, "k": None, "v": None}

    def commit(self, payload: CrossKVCache | None = None) -> None:
        if payload is not None:
            self._cache = payload

    def view(self, *, include_staged: bool = True) -> CrossKVCache:
        if self._cache is None:
            raise RuntimeError("EncodeOnceKV is not allocated; call allocate() first.")
        return self._cache

    def reset(self) -> None:
        self._cache = None

    @property
    def nbytes(self) -> int:
        if self._cache is None:
            return 0
        total = 0
        for key in ("k", "v"):
            tensor = self._cache.get(key)
            if isinstance(tensor, torch.Tensor):
                total += tensor.numel() * tensor.element_size()
        return total

    @property
    def resident(self) -> bool:
        return self._cache is not None


class LatentBuffer(MemoryObject[ItemT, list[ItemT]]):
    """Append / ring buffer of latent or pixel frames.

    A bounded ``deque`` (``maxlen`` set at ``allocate()`` time). Compaction
    (FramePack-style) is not yet implemented. Model-specific stacking logic
    stays in the caller; this object only stores and views the frames.
    """

    def __init__(self) -> None:
        self._buf: deque[ItemT] | None = None

    def allocate(self, *, maxlen: int | None = None) -> None:
        self._buf = deque(maxlen=maxlen)

    def append(self, payload: ItemT) -> None:
        if self._buf is None:
            raise RuntimeError("LatentBuffer is not allocated; call allocate() first.")
        self._buf.append(payload)

    def extend(self, payloads: Iterable[ItemT]) -> None:
        if self._buf is None:
            raise RuntimeError("LatentBuffer is not allocated; call allocate() first.")
        self._buf.extend(payloads)

    def commit(self, payload: ItemT | None = None) -> None:
        if payload is not None:
            self.append(payload)

    def view(self, *, include_staged: bool = True) -> list[ItemT]:
        if self._buf is None:
            raise RuntimeError("LatentBuffer is not allocated; call allocate() first.")
        return list(self._buf)

    def __len__(self) -> int:
        return 0 if self._buf is None else len(self._buf)

    def reset(self) -> None:
        self._buf = None

    @property
    def nbytes(self) -> int:
        if self._buf is None:
            return 0
        total = 0
        for item in self._buf:
            if isinstance(item, torch.Tensor):
                total += item.numel() * item.element_size()
            elif isinstance(item, np.ndarray):
                total += int(item.nbytes)
        return total

    @property
    def resident(self) -> bool:
        return self._buf is not None
