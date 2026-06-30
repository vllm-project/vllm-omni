"""Tests for embedding_cache.cache."""

import os
import threading
import time

import torch

from vllm_omni.embedding_cache.cache import EmbeddingCache

# ── helpers ────────────────────────────────────────────────────────────────


def _t(*shape) -> torch.Tensor:
    return torch.randn(*shape)


def _nbytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


# ── basic get/put ───────────────────────────────────────────────────────────


def test_put_then_get():
    c = EmbeddingCache(ttl_seconds=60, max_bytes=1 << 30)
    t = _t(4, 64)
    c.put("k1", t, _nbytes(t))
    result = c.get("k1")
    assert result is not None
    assert result.shape == t.shape
    c.close()


def test_get_missing_returns_none():
    c = EmbeddingCache(ttl_seconds=60, max_bytes=1 << 30)
    assert c.get("missing") is None
    c.close()


# ── TTL eviction ────────────────────────────────────────────────────────────


def test_ttl_eviction():
    c = EmbeddingCache(ttl_seconds=1, max_bytes=1 << 30)
    t = _t(4, 64)
    c.put("k1", t, _nbytes(t))
    assert c.get("k1") is not None
    time.sleep(1.1)
    assert c.get("k1") is None  # expired
    c.close()


# ── LRU eviction ────────────────────────────────────────────────────────────


def test_lru_eviction():
    elem_bytes = _nbytes(_t(1, 64))
    c = EmbeddingCache(ttl_seconds=60, max_bytes=elem_bytes * 2)
    t1, t2, t3 = _t(1, 64), _t(1, 64), _t(1, 64)
    c.put("k1", t1, elem_bytes)
    c.put("k2", t2, elem_bytes)
    # Access k1 so k2 is LRU
    c.get("k1")
    # Inserting k3 should evict k2
    c.put("k3", t3, elem_bytes)
    assert c.get("k1") is not None
    assert c.get("k2") is None  # evicted
    assert c.get("k3") is not None
    c.close()


# ── stats ───────────────────────────────────────────────────────────────────


def test_stats():
    c = EmbeddingCache(ttl_seconds=60, max_bytes=1 << 30)
    t = _t(4, 64)
    c.put("k1", t, _nbytes(t))
    c.get("k1")  # hit
    c.get("missing")  # miss
    s = c.stats()
    assert s["hits"] == 1
    assert s["misses"] == 1
    assert s["entries"] == 1
    c.close()


# ── clear ───────────────────────────────────────────────────────────────────


def test_clear():
    c = EmbeddingCache(ttl_seconds=60, max_bytes=1 << 30)
    t = _t(4, 64)
    c.put("k1", t, _nbytes(t))
    c.clear()
    assert c.get("k1") is None
    assert c.stats()["entries"] == 0
    c.close()


# ── thread safety ───────────────────────────────────────────────────────────


def test_concurrent_put_get():
    c = EmbeddingCache(ttl_seconds=60, max_bytes=1 << 30)
    errors = []

    def worker(idx: int):
        try:
            key = f"k{idx}"
            t = _t(4, 64)
            c.put(key, t, _nbytes(t))
            for _ in range(20):
                c.get(key)
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    assert not errors, errors
    c.close()


# ── singleton ───────────────────────────────────────────────────────────────


def test_get_embedding_cache_disabled():
    # Without env var the singleton is None
    os.environ.pop("VLLM_OMNI_EMBEDDING_CACHE", None)
    # Reload to reset singleton state
    import importlib

    import vllm_omni.embedding_cache.cache as cache_mod

    importlib.reload(cache_mod)
    assert cache_mod.get_embedding_cache() is None


def test_get_embedding_cache_enabled(monkeypatch):
    monkeypatch.setenv("VLLM_OMNI_EMBEDDING_CACHE", "1")
    import importlib

    import vllm_omni.embedding_cache.cache as cache_mod

    importlib.reload(cache_mod)
    inst = cache_mod.get_embedding_cache()
    assert inst is not None
    assert isinstance(inst, cache_mod.EmbeddingCache)
    inst.close()
