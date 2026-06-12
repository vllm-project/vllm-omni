# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded LRU cache for Higgs Audio v3 reference-audio codec encoding.

Voice-clone requests that re-send the same ref_audio are otherwise re-encoded
through the DAC encoder on every call. The cache keys by a stable content
hash (size + sample rate + head/mid/tail sentinel bytes) so the same audio
served via different URLs or repeated base64 payloads also hits warm.
Tensors are cloned on get/put so callers and the cache cannot alias storage.
"""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from collections.abc import Callable

import numpy as np
import torch

__all__ = ["cached_encode_reference_audio", "cache_stats", "clear_cache"]

_DEFAULT_MAX_ENTRIES = 128
_SENTINEL_BYTES = 4096

_cache: OrderedDict[str, torch.Tensor] = OrderedDict()
_cache_lock = threading.Lock()
_hits = 0
_misses = 0


def _content_hash(wav: np.ndarray, sr: int) -> str:
    """Stable content hash. Head/mid/tail sentinel keeps cost O(1)."""
    raw = wav.tobytes()
    n = len(raw)
    h = hashlib.sha256()
    h.update(n.to_bytes(8, "little"))
    h.update(int(sr).to_bytes(4, "little"))
    if n == 0:
        return h.hexdigest()
    if n <= _SENTINEL_BYTES * 3:
        h.update(raw)
        return h.hexdigest()
    h.update(raw[:_SENTINEL_BYTES])
    mid = (n - _SENTINEL_BYTES) // 2
    h.update(raw[mid : mid + _SENTINEL_BYTES])
    h.update(raw[-_SENTINEL_BYTES:])
    return h.hexdigest()


def cached_encode_reference_audio(
    wav: np.ndarray,
    sr: int,
    encode_fn: Callable[[np.ndarray, int], torch.Tensor],
    max_entries: int = _DEFAULT_MAX_ENTRIES,
) -> torch.Tensor:
    """LRU-cached wrapper around ``encode_reference_audio``.

    ``encode_fn`` is the actual codec encoder; passed in so this module
    stays decoupled from the tokenizer import (avoids circular imports
    during stage-input-processor discovery).

    Returns a CPU tensor clone — never the cache's interior storage.
    """
    global _hits, _misses
    key = _content_hash(wav, sr)
    with _cache_lock:
        cached = _cache.get(key)
        if cached is not None:
            _hits += 1
            _cache.move_to_end(key)
            return cached.clone()

    # Miss: encode outside the lock so concurrent misses on different
    # refs don't serialize on this single mutex.
    codes = encode_fn(wav, sr)
    codes_cpu = codes.detach().to("cpu").contiguous()
    with _cache_lock:
        if key in _cache:
            # Concurrent miss winner already stored it; refresh LRU position
            # and return that copy to avoid divergent tensors per request.
            _cache.move_to_end(key)
            _misses += 1  # we still paid the encode cost
            return _cache[key].clone()
        _cache[key] = codes_cpu
        if len(_cache) > max_entries:
            _cache.popitem(last=False)
        _misses += 1
    return codes_cpu.clone()


def cache_stats() -> dict:
    """Return (hits, misses, size, hit_rate) — for /metrics-style reporting."""
    with _cache_lock:
        total = _hits + _misses
        return {
            "hits": _hits,
            "misses": _misses,
            "size": len(_cache),
            "hit_rate": (_hits / total) if total > 0 else 0.0,
        }


def clear_cache() -> None:
    """Drop all cached entries. Used by tests."""
    global _hits, _misses
    with _cache_lock:
        _cache.clear()
        _hits = 0
        _misses = 0
