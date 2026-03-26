import threading

import pytest

from vllm_omni.utils.voice_cache import VoiceEmbeddingCache

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def cache():
    return VoiceEmbeddingCache(max_entries=4)


class TestVoiceEmbeddingCache:
    def test_miss_returns_none(self, cache: VoiceEmbeddingCache):
        assert cache.get("nonexistent") is None
        assert cache.stats()["misses"] == 1

    def test_put_and_hit(self, cache: VoiceEmbeddingCache):
        cache.put("abc", {"val": 42})
        result = cache.get("abc")
        assert result is not None
        assert result["val"] == 42
        assert cache.stats()["hits"] == 1

    def test_lru_eviction(self, cache: VoiceEmbeddingCache):
        for i in range(5):
            cache.put(f"key{i}", {"i": i})
        # key0 should have been evicted (oldest, max_entries=4)
        assert cache.get("key0") is None
        # key1..key4 should still be present
        for i in range(1, 5):
            assert cache.get(f"key{i}") is not None
        assert cache.stats()["entries"] == 4

    def test_lru_access_promotes(self, cache: VoiceEmbeddingCache):
        cache.put("a", {"v": 1})
        cache.put("b", {"v": 2})
        cache.put("c", {"v": 3})
        cache.put("d", {"v": 4})
        # Access "a" to promote it to MRU
        cache.get("a")
        # Insert "e" -- should evict "b" (now the oldest), not "a"
        cache.put("e", {"v": 5})
        assert cache.get("a") is not None
        assert cache.get("b") is None

    def test_put_overwrites(self, cache: VoiceEmbeddingCache):
        cache.put("k", {"old": True})
        cache.put("k", {"new": True})
        result = cache.get("k")
        assert result is not None
        assert "new" in result
        assert "old" not in result
        assert cache.stats()["entries"] == 1

    def test_make_cache_key_includes_mode(self):
        k1 = VoiceEmbeddingCache.make_cache_key("alice", xvec_only=True)
        k2 = VoiceEmbeddingCache.make_cache_key("alice", xvec_only=False)
        assert k1 != k2
        assert "xvec" in k1
        assert "icl" in k2

    def test_make_cache_key_deterministic(self):
        k1 = VoiceEmbeddingCache.make_cache_key("bob", xvec_only=True)
        k2 = VoiceEmbeddingCache.make_cache_key("bob", xvec_only=True)
        assert k1 == k2

    def test_make_cache_key_created_at_isolation(self):
        """Different created_at timestamps must produce different keys (stale-cache protection)."""
        k1 = VoiceEmbeddingCache.make_cache_key("alice", xvec_only=False, created_at=1000.0)
        k2 = VoiceEmbeddingCache.make_cache_key("alice", xvec_only=False, created_at=2000.0)
        assert k1 != k2

    def test_stale_cache_protection(self, cache: VoiceEmbeddingCache):
        """Re-upload (new created_at) must produce a cache miss, not a stale hit."""
        key_old = VoiceEmbeddingCache.make_cache_key("alice", xvec_only=False, created_at=1000.0)
        key_new = VoiceEmbeddingCache.make_cache_key("alice", xvec_only=False, created_at=2000.0)
        cache.put(key_old, {"ref_spk_embedding": "old_emb"})
        # Re-upload produces a new created_at → different key → cold miss
        assert cache.get(key_new) is None
        # Old key still in cache (not yet evicted)
        assert cache.get(key_old) is not None

    def test_cache_mode_isolation(self, cache: VoiceEmbeddingCache):
        """xvec entry must NOT be served for an icl request (same voice)."""
        key_xvec = VoiceEmbeddingCache.make_cache_key("alice", xvec_only=True)
        key_icl = VoiceEmbeddingCache.make_cache_key("alice", xvec_only=False)
        cache.put(key_xvec, {"ref_code": None, "ref_spk_embedding": "emb"})
        # icl request should miss — different key
        assert cache.get(key_icl) is None
        # xvec request should hit
        assert cache.get(key_xvec) is not None

    def test_stats_counters(self, cache: VoiceEmbeddingCache):
        cache.put("x", {"v": 1})
        cache.get("x")  # hit
        cache.get("x")  # hit
        cache.get("y")  # miss
        s = cache.stats()
        assert s["hits"] == 2
        assert s["misses"] == 1
        assert s["entries"] == 1
        assert s["max_entries"] == 4

    def test_thread_safety(self):
        cache = VoiceEmbeddingCache(max_entries=32)
        errors = []

        def worker(thread_id: int):
            try:
                for i in range(50):
                    key = f"t{thread_id}_k{i}"
                    cache.put(key, {"tid": thread_id, "i": i})
                    cache.get(key)
                    cache.get(f"t{(thread_id + 1) % 10}_k{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread safety errors: {errors}"
        s = cache.stats()
        assert s["entries"] <= 32
