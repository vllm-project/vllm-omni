# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""A bounded, immutable cross-request snapshot of the first video block's K/V.

The snapshot lives on CPU. A hit copies it into newly allocated request-owned
pages, so no later video write can mutate another request's history. Text
conditioning K/V never enters this store.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from threading import RLock

import torch


@dataclass(frozen=True, slots=True)
class FirstBlockKVKey:
    input_sha256: str
    model_identity: int
    model_revision: int
    branch: str
    token_count: int
    num_layers: int
    local_heads: int
    head_size: int
    dtype: torch.dtype
    device: torch.device


@dataclass(frozen=True, slots=True)
class FirstBlockKVSnapshot:
    key: FirstBlockKVKey
    layers: tuple[tuple[torch.Tensor, torch.Tensor], ...]
    nbytes: int


def fingerprint_first_block_inputs(
    cache_input: torch.Tensor,
    prompt_embeds: torch.Tensor,
    camera: torch.Tensor,
) -> str:
    """Hash values, dtype and shape, never an allocation address or shape alone."""
    digest = hashlib.sha256(b"lingbot-first-block-self-kv-v1\0")
    for name, tensor in (
        ("cache_input", cache_input),
        ("prompt_embeds", prompt_embeds),
        ("camera", camera),
    ):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a tensor")
        digest.update(name.encode("ascii") + b"\0")
        digest.update(str(tensor.dtype).encode("ascii") + b"\0")
        digest.update(repr(tuple(tensor.shape)).encode("ascii") + b"\0")
        # An inference tensor need not expose a usable _version. A value hash
        # also catches changes to a live tensor without relying on its address.
        contents = tensor.detach().contiguous().view(torch.uint8).cpu().numpy()
        digest.update(contents.tobytes())
    return digest.hexdigest()


class FirstBlockKVStore:
    """One complete snapshot per model worker, with a strict CPU byte limit."""

    def __init__(self, *, max_bytes: int) -> None:
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        self.max_bytes = max_bytes
        self._lock = RLock()
        self._snapshot: FirstBlockKVSnapshot | None = None
        self._generation = 0
        self.hits = 0
        self.misses = 0
        self.publications = 0
        self.restores = 0

    def lookup(self, key: FirstBlockKVKey) -> FirstBlockKVSnapshot | None:
        with self._lock:
            snapshot = self._snapshot
            if snapshot is not None and snapshot.key == key:
                self.hits += 1
                return snapshot
            self.misses += 1
            return None

    def record_restore(self, key: FirstBlockKVKey) -> None:
        with self._lock:
            self.restores += 1

    def begin_build(self, key: FirstBlockKVKey) -> int:
        with self._lock:
            # A single-entry cache must release the old snapshot before a new
            # multi-GiB snapshot is copied to CPU, rather than temporarily
            # retaining two complete entries during publication.
            if self._snapshot is not None and self._snapshot.key != key:
                self._snapshot = None
            self._generation += 1
            return self._generation

    def publish(
        self,
        generation: int,
        key: FirstBlockKVKey,
        layers: tuple[tuple[torch.Tensor, torch.Tensor], ...],
    ) -> bool:
        if len(layers) != key.num_layers:
            raise ValueError("first-block snapshot must contain every layer")
        shape = (key.token_count, key.local_heads, key.head_size)
        nbytes = 0
        for k, v in layers:
            for tensor in (k, v):
                if tensor.device.type != "cpu" or tensor.dtype != key.dtype or tuple(tensor.shape) != shape:
                    raise ValueError("first-block snapshot has an incompatible K/V representation")
                nbytes += tensor.numel() * tensor.element_size()
        if nbytes > self.max_bytes:
            return False
        with self._lock:
            if generation != self._generation:
                return False
            self._snapshot = FirstBlockKVSnapshot(key, layers, nbytes)
            self.publications += 1
            return True

    def invalidate(self) -> None:
        """Retire the snapshot and any in-flight publication after weight changes."""
        with self._lock:
            self._generation += 1
            self._snapshot = None
