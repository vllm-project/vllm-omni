import threading

import pytest
import torch

from vllm_omni.utils.speaker_cache import SpeakerEmbeddingCache, get_speaker_cache

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def cache():
    return SpeakerEmbeddingCache(max_bytes=10 * 1024**2)


def _k(model: str, name: str, created_at: int = 0) -> tuple[str, str, int]:
    return (model, name, created_at)


class TestSpeakerEmbeddingCacheBehavior:
    def test_miss_returns_none(self, cache):
        assert cache.get(_k("voxcpm2", "nonexistent")) is None

    def test_put_and_hit(self, cache):
        cache.put(_k("voxcpm2", "alice"), {"val": 42})
        assert cache.get(_k("voxcpm2", "alice"))["val"] == 42

    def test_lru_access_promotes(self):
        c = SpeakerEmbeddingCache(max_bytes=4 * 4096)
        for k in ("a", "b", "c", "d"):
            c.put(_k("m", k), {"emb": torch.zeros(1024, dtype=torch.float32)})
        c.get(_k("m", "a"))
        c.put(_k("m", "e"), {"emb": torch.zeros(1024, dtype=torch.float32)})
        assert c.get(_k("m", "a")) is not None
        assert c.get(_k("m", "b")) is None

    def test_put_overwrites(self, cache):
        cache.put(_k("m", "k"), {"old": True})
        cache.put(_k("m", "k"), {"new": True})
        assert "new" in cache.get(_k("m", "k"))
        assert cache.stats()["entries"] == 1

    def test_make_cache_key_namespaces_model_type(self):
        k1 = SpeakerEmbeddingCache.make_cache_key("alice", model_type="voxcpm2")
        k2 = SpeakerEmbeddingCache.make_cache_key("alice", model_type="fish_speech")
        assert k1 != k2
        assert k1 == ("voxcpm2", "alice", 0)
        assert k2 == ("fish_speech", "alice", 0)

    def test_make_cache_key_created_at_isolation(self):
        k_old = SpeakerEmbeddingCache.make_cache_key("alice", model_type="voxcpm2", created_at=1712000000)
        k_new = SpeakerEmbeddingCache.make_cache_key("alice", model_type="voxcpm2", created_at=1712000042)
        assert k_old != k_new

    def test_make_cache_key_requires_fields(self):
        with pytest.raises(ValueError):
            SpeakerEmbeddingCache.make_cache_key("", model_type="voxcpm2")
        with pytest.raises(ValueError):
            SpeakerEmbeddingCache.make_cache_key("alice", model_type="")

    def test_clear_all(self, cache):
        cache.put(_k("m", "a"), {"v": 1})
        cache.put(_k("m", "b"), {"v": 2})
        assert cache.clear() == 2
        assert cache.stats()["entries"] == 0

    def test_clear_matches_speaker_across_model_types(self, cache):
        cache.put(_k("voxcpm2", "alice", 1), {"v": 1})
        cache.put(_k("fish_speech", "alice", 2), {"v": 2})
        cache.put(_k("cosyvoice3", "bob", 3), {"v": 3})
        assert cache.clear("alice") == 2
        assert cache.get(_k("voxcpm2", "alice", 1)) is None
        assert cache.get(_k("fish_speech", "alice", 2)) is None
        assert cache.get(_k("cosyvoice3", "bob", 3)) is not None

    def test_stale_cache_on_reupload(self, cache):
        cache.put(_k("voxcpm2", "alice", 1712000000), {"emb": torch.zeros(4), "gen": "old"})
        assert cache.get(_k("voxcpm2", "alice", 1712000042)) is None

    def test_memory_bytes(self, cache):
        assert cache.memory_bytes() == 0
        t = torch.zeros(1024, dtype=torch.float32)  # 4096 bytes
        cache.put(_k("m", "k"), {"emb": t})
        assert cache.memory_bytes() == 4096

    def test_memory_bytes_ignores_non_tensors(self, cache):
        cache.put(_k("m", "k"), {"flag": True, "name": "test", "nothing": None})
        assert cache.memory_bytes() == 0

    def test_byte_budget_evicts(self):
        c = SpeakerEmbeddingCache(max_bytes=8192)
        c.put(_k("m", "a"), {"emb": torch.zeros(1024, dtype=torch.float32)})
        c.put(_k("m", "b"), {"emb": torch.zeros(1024, dtype=torch.float32)})
        c.put(_k("m", "c"), {"emb": torch.zeros(1024, dtype=torch.float32)})
        assert c.get(_k("m", "a")) is None
        assert c.get(_k("m", "b")) is not None
        assert c.get(_k("m", "c")) is not None
        assert c.memory_bytes() <= 8192

    def test_oversize_entry_skipped(self):
        c = SpeakerEmbeddingCache(max_bytes=1024)
        c.put(_k("m", "huge"), {"emb": torch.zeros(2048, dtype=torch.float32)})
        assert c.get(_k("m", "huge")) is None
        assert c.stats()["entries"] == 0

    def test_stats(self, cache):
        cache.put(_k("m", "x"), {"v": 1})
        cache.get(_k("m", "x"))
        cache.get(_k("m", "y"))
        s = cache.stats()
        assert s["hits"] == 1
        assert s["misses"] >= 1
        assert s["entries"] == 1

    def test_thread_safety(self):
        cache = SpeakerEmbeddingCache()
        errors = []

        def worker(tid):
            try:
                for i in range(50):
                    cache.put(_k("m", f"t{tid}_v{i}"), {"tid": tid})
                    cache.get(_k("m", f"t{tid}_v{i}"))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        assert cache.stats()["entries"] == 500

    def test_empty_speaker_name_raises_error(self, cache):
        with pytest.raises(ValueError, match="speaker_name cannot be an empty string"):
            cache.clear("")

    def test_cpu_storage_verification(self, cache):
        tensor = torch.randn(10, 128)
        cache.put(_k("m", "alice"), {"emb": tensor})
        cached = cache.get(_k("m", "alice"))
        assert cached["emb"].device.type == "cpu"


class TestSingleton:
    def test_singleton_identity(self, fresh_speaker_cache):
        a = get_speaker_cache()
        b = get_speaker_cache()
        assert a is b
