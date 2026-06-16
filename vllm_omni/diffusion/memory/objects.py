# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Concrete memory objects (RFC #4480).

All three classes back their storage with plain (monolithic) buffers. The RFC
names ``PagedKV`` for paged block-table backing; here it is a single contiguous
tensor and the paged backing is not yet implemented. ``FixedState`` and
``RetrievalStore`` from the RFC are not implemented here (DreamZero needs
neither), but the names are reserved.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from typing import Any

import numpy as np
import torch

from vllm_omni.diffusion.memory.base import MemoryObject


class PagedKV(MemoryObject):
    """Self-attention KV for one layer and one CFG branch.

    RFC name ``PagedKV``; for now it is backed by a single contiguous tensor of
    shape ``(2, B, seq, H, D)`` -- the exact shape the bespoke ``DreamZeroState``
    stores. The diffusion loop hands grown KV in via ``commit()`` and reads it
    back via ``view()``; growth and windowing happen in the model, not here.
    """

    def __init__(self) -> None:
        self._buf: torch.Tensor | None = None
        self._staged: Any = None

    def allocate(  # type: ignore[override]  # explicit shape params; base is **spec
        self,
        *,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device | str,
        num_heads: int,
        head_dim: int,
        **_: Any,
    ) -> None:
        self._buf = torch.zeros(2, batch_size, 0, num_heads, head_dim, dtype=dtype, device=device)

    def commit(self, payload: torch.Tensor | None = None) -> None:
        if payload is None:
            raise ValueError("PagedKV.commit requires a tensor payload.")
        # Match DreamZeroState.update_kv_cache: store a clone so the caller's
        # tensor (e.g. the model's torch.stack result) cannot alias and later
        # mutate the cache.
        self._buf = payload.clone()

    def view(self, *, include_staged: bool = True) -> torch.Tensor:
        if self._buf is None:
            raise RuntimeError("PagedKV is not allocated; call allocate() first.")
        return self._buf

    def reset(self) -> None:
        self._buf = None
        self._staged = None

    @property
    def nbytes(self) -> int:
        if self._buf is None:
            return 0
        return self._buf.numel() * self._buf.element_size()

    @property
    def resident(self) -> bool:
        return self._buf is not None


class EncodeOnceKV(MemoryObject):
    """Encode-once cross-attention KV.

    Wraps the ``{"is_init", "k", "v"}`` dict that DreamZero's cross-attention
    layers populate once (on the first forward) and read thereafter. ``view()``
    returns the live dict so the model mutates it in place, exactly as today.
    """

    def __init__(self) -> None:
        self._cache: dict[str, bool | torch.Tensor | None] | None = None

    def allocate(self, **_: Any) -> None:
        self._cache = {"is_init": False, "k": None, "v": None}

    def commit(self, payload: dict[str, bool | torch.Tensor | None] | None = None) -> None:
        if payload is not None:
            self._cache = payload

    def view(self, *, include_staged: bool = True) -> dict[str, bool | torch.Tensor | None]:
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


class LatentBuffer(MemoryObject):
    """Append / ring buffer of latent or pixel frames.

    A bounded ``deque`` (``maxlen`` set at ``allocate()`` time). Compaction
    (FramePack-style) is not yet implemented. Model-specific stacking logic
    stays in the caller; this object only stores and views the frames.
    """

    def __init__(self) -> None:
        self._buf: deque[Any] | None = None

    def allocate(self, *, maxlen: int | None = None, **_: Any) -> None:
        self._buf = deque(maxlen=maxlen)

    def append(self, payload: Any) -> None:
        if self._buf is None:
            raise RuntimeError("LatentBuffer is not allocated; call allocate() first.")
        self._buf.append(payload)

    def extend(self, payloads: Iterable[Any]) -> None:
        if self._buf is None:
            raise RuntimeError("LatentBuffer is not allocated; call allocate() first.")
        self._buf.extend(payloads)

    def commit(self, payload: Any = None) -> None:
        if payload is not None:
            self.append(payload)

    def view(self, *, include_staged: bool = True) -> list[Any]:
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
