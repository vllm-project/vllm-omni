# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Concrete memory objects (RFC #4480).

``LatentBuffer`` backs its storage with a plain (monolithic) buffer. Attention
KV is deliberately absent: for DreamZero it is owned by the AR-Diffusion engine's
paged pool (PR #4534), and the RFC's ``PagedKV`` arrives in Phase 1 as a
handle wrapping that engine state rather than a dense buffer here. The RFC's
``FixedState`` and ``RetrievalStore`` land with their first consumers, but the
names are reserved.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from typing import TypeVar

from vllm_omni.experimental.world_models.session_state.accounting import (
    SeenStorages,
    device_bytes,
)
from vllm_omni.experimental.world_models.session_state.base import StateObject

# The frame/chunk type a ``LatentBuffer`` holds (e.g. ``np.ndarray`` pixel
# frames, ``torch.Tensor`` latent chunks).
ItemT = TypeVar("ItemT")


class LatentBuffer(StateObject[ItemT, list[ItemT]]):
    """Append / ring buffer of latent or pixel frames.

    A bounded ``deque`` (``maxlen`` set at ``allocate()`` time). The object
    only stores and views the frames; model-specific stacking stays in the
    caller.
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

    def nbytes_by_device(self, seen: SeenStorages) -> dict[str, int]:
        if self._buf is None:
            return {}
        return device_bytes(list(self._buf), seen)

    @property
    def resident(self) -> bool:
        return self._buf is not None
