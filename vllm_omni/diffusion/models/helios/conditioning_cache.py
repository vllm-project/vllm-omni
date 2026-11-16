# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Helios adapter for request-owned, non-paged conditioning K/V.

No entry is shared across requests. Versionless inference tensors are compared
against owned snapshots; object identity or an allocator address is insufficient.
This conservative comparison can synchronize CUDA. Callers must not mutate
conditioning or model weights concurrently with an executing forward.
"""

from __future__ import annotations

import weakref
from collections.abc import Callable, Hashable
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from threading import RLock

import torch

from vllm_omni.diffusion.attention.conditioning import (
    ConditioningKVIdentity,
    ConditioningKVRepresentation,
    ConditioningKVScope,
    TensorConditioningKVStorage,
)

_active_cache: ContextVar[HeliosConditioningCache | None] = ContextVar("helios_conditioning_cache", default=None)


def active_conditioning_cache() -> HeliosConditioningCache | None:
    return _active_cache.get()


def request_conditioning_cache(forward):
    """Own a cache for the entire ordinary request, including exceptional exit."""

    @wraps(forward)
    def wrapped(*args, **kwargs):
        with HeliosConditioningCache() as cache, cache.activate():
            return forward(*args, **kwargs)

    return wrapped


def tensor_version(tensor: torch.Tensor) -> int | None:
    try:
        return tensor._version
    except RuntimeError:
        return None


@dataclass
class _Entry:
    source: weakref.ReferenceType
    version: int | None
    source_metadata: tuple
    snapshot: torch.Tensor | None
    projected: torch.Tensor
    identity: ConditioningKVIdentity
    scope: ConditioningKVScope


class HeliosConditioningCache:
    """Model-specific preparation; validity/publication/release use the common scope."""

    def __init__(self, request_id: str = "helios") -> None:
        self.request_id = request_id
        self._entries: dict[str, _Entry] = {}
        self._lock = RLock()
        self._closed = False
        self._revision = 0

    @contextmanager
    def activate(self):
        token = _active_cache.set(self)
        try:
            yield self
        finally:
            _active_cache.reset(token)

    def prepare(
        self,
        source: torch.Tensor,
        *,
        branch: str,
        projection_revision: Hashable,
        build: Callable[[torch.Tensor], tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]],
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        if branch not in ("positive", "negative"):
            raise ValueError("Helios conditioning branch must be positive or negative")
        with self._lock:
            if self._closed:
                raise RuntimeError("Helios conditioning cache is closed")
            version = tensor_version(source)
            metadata = (source.shape, source.stride(), source.dtype, source.device)
            entry = self._entries.get(branch)
            if entry is not None:
                # lookup also waits for the publication event before reading the
                # snapshot/projected embeddings produced on a different stream.
                layers = entry.scope.lookup(entry.identity)
                same = (
                    entry.source() is source
                    and entry.source_metadata == metadata
                    and entry.identity.projection_revision == projection_revision
                )
                if same and version is not None:
                    same = entry.version == version
                elif same:
                    same = entry.snapshot is not None and torch.equal(source, entry.snapshot)
                if same and layers is not None:
                    if entry.projected.is_cuda:
                        entry.projected.record_stream(torch.cuda.current_stream(entry.projected.device))
                    return entry.projected, [(layer["k"], layer["v"]) for layer in layers]
                # Retire the old value before rebuilding; failure cannot expose it.
                entry.scope.close()
                del self._entries[branch]
            self._revision += 1
            snapshot = source.detach().clone() if version is None else None
            projected, key_values = build(source)
            if not key_values:
                raise ValueError("Helios conditioning requires at least one layer")
            k = key_values[0][0]
            batch, length, heads, dim = k.shape
            storage = TensorConditioningKVStorage(device=k.device, dtype=k.dtype, num_layers=len(key_values))
            scope = ConditioningKVScope(storage, self.request_id, len(key_values))
            identity = ConditioningKVIdentity(
                cache_name="helios_text",
                kv_branch=branch,
                condition_revision=self._revision,
                projection_revision=projection_revision,
                representation=ConditioningKVRepresentation(
                    dtype=k.dtype,
                    device=k.device,
                    layout="BSND",
                    kv_length=length,
                    local_heads=heads,
                    head_size=dim,
                    batch_size=batch,
                ),
            )
            try:
                scope.publish(scope.begin_build(identity), key_values)
                layers = scope.lookup(identity)
                assert layers is not None
            except BaseException:
                scope.close()
                raise
            self._entries[branch] = _Entry(weakref.ref(source), version, metadata, snapshot, projected, identity, scope)
            return projected, [(layer["k"], layer["v"]) for layer in layers]

    def invalidate(self) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("Helios conditioning cache is closed")
            for entry in self._entries.values():
                entry.scope.close()
            self._entries.clear()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            for entry in self._entries.values():
                entry.scope.close()
            self._entries.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
