# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Weakref-guarded LRU cache for cross-attention key/value reuse.

``HeliosTransformer3DModel`` projects the text-encoder output into
cross-attention K/V once per request and reuses the projection across the
~50 denoise steps of that request.  The cache is keyed by a storage
fingerprint of the source tensor (``data_ptr`` / shape / stride / dtype /
device / ``_version``).  That fingerprint is **not** unique across requests:
once the source tensor is freed, PyTorch's caching allocator may hand its
address to an unrelated tensor of the same shape, producing a false cache
hit and cross-prompt contamination (request B silently reuses request A's
projected K/V).

``SourceTensorLRUCache`` stores a :mod:`weakref` to the source tensor next
to the cached value and verifies the source is still alive on every lookup.
An address reused by a *different* tensor still matches the fingerprint,
but the original source's weakref is dead by then, so the lookup becomes a
miss instead of a silent wrong hit.  Within a single request the source
tensor is held alive across all denoise steps, so legitimate intra-request
reuse still hits.
"""

from __future__ import annotations

import weakref
from collections import OrderedDict

import torch


class SourceTensorLRUCache:
    """Bounded LRU cache keyed by a source-tensor fingerprint + liveness.

    The key is derived from the source tensor's storage fingerprint
    (``data_ptr`` / shape / stride / dtype / device / ``_version``); a
    weakref to the source is stored alongside the value and checked on
    lookup so that an allocator address reused after the source is freed
    never yields a false hit.  Stale entries (dead weakref) are evicted
    lazily on lookup and proactively on insert.
    """

    def __init__(self, max_size: int = 2) -> None:
        self._cache: OrderedDict[tuple, tuple[weakref.ReferenceType, object]] = OrderedDict()
        self._max_size = max_size

    @staticmethod
    def _key(tensor: torch.Tensor) -> tuple:
        try:
            version = tensor._version
        except RuntimeError:
            version = None
        return (
            tensor.data_ptr(),
            tuple(tensor.shape),
            tuple(tensor.stride()),
            tensor.dtype,
            tensor.device.type,
            tensor.device.index,
            version,
        )

    def get(self, source: torch.Tensor):
        """Return the cached value for ``source`` or ``None`` on miss.

        A miss is also returned when the source tensor that originally
        populated a matching entry has been freed (its weakref is dead),
        even if the fingerprint now matches a new tensor reusing the
        address; the stale entry is evicted in that case.
        """
        key = self._key(source)
        entry = self._cache.get(key)
        if entry is None:
            return None
        src_ref, value = entry
        if src_ref() is None:
            # Source freed: the address may have been reused by an unrelated
            # tensor, so the fingerprint match is a coincidence -> miss.
            self._cache.pop(key, None)
            return None
        self._cache.move_to_end(key)
        return value

    def put(self, source: torch.Tensor, value) -> None:
        key = self._key(source)
        # Proactively drop entries whose source has been freed so cached
        # values do not linger until the LRU bound evicts them.
        if self._cache:
            for stale in [k for k, v in self._cache.items() if v[0]() is None]:
                del self._cache[stale]
        self._cache[key] = (weakref.ref(source), value)
        self._cache.move_to_end(key)
        while len(self._cache) > self._max_size:
            self._cache.popitem(last=False)

    def clear(self) -> None:
        self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)
