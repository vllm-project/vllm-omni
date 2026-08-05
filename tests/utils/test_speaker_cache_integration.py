"""Integration tests for the process-wide speaker cache across serving + models."""

import pytest
import torch

from vllm_omni.utils.speaker_cache import SpeakerEmbeddingCache, get_speaker_cache

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class TestSpeakerCacheIntegration:
    def test_delete_propagates_across_model_types(self):
        cache = SpeakerEmbeddingCache()
        voxcpm_key = SpeakerEmbeddingCache.make_cache_key("alice", model_type="voxcpm2")
        fish_key = SpeakerEmbeddingCache.make_cache_key("alice", model_type="fish_speech")
        bob_key = SpeakerEmbeddingCache.make_cache_key("bob", model_type="voxcpm2")

        cache.put(voxcpm_key, {"emb": torch.zeros(4)})
        cache.put(fish_key, {"emb": torch.zeros(4)})
        cache.put(bob_key, {"emb": torch.zeros(4)})

        removed = cache.clear("alice")

        assert removed == 2
        assert cache.get(voxcpm_key) is None
        assert cache.get(fish_key) is None
        assert cache.get(bob_key) is not None

    def test_singleton_shared_across_call_sites(self, fresh_speaker_cache):
        cache_a = get_speaker_cache()
        cache_b = get_speaker_cache()
        assert cache_a is cache_b
        key = SpeakerEmbeddingCache.make_cache_key("carol", model_type="voxcpm2")
        cache_a.put(key, {"tag": "from_a"})
        got = cache_b.get(key)
        assert got is not None
        assert got["tag"] == "from_a"

    def test_shutdown_clears_all_entries(self):
        cache = SpeakerEmbeddingCache()
        for i in range(3):
            k = SpeakerEmbeddingCache.make_cache_key(f"voice{i}", model_type="voxcpm2")
            cache.put(k, {"emb": torch.zeros(2)})
        assert cache.stats()["entries"] == 3
        cache.clear()
        assert cache.stats()["entries"] == 0

    def test_stale_cache_protection_delete_then_reupload(self):
        cache = SpeakerEmbeddingCache()
        old_key = SpeakerEmbeddingCache.make_cache_key("alice", model_type="voxcpm2", created_at=1712000000)
        cache.put(old_key, {"emb": torch.ones(4) * 3.14, "gen": "old"})

        new_key = SpeakerEmbeddingCache.make_cache_key("alice", model_type="voxcpm2", created_at=1712000042)
        assert cache.get(new_key) is None
        assert cache.get(old_key) is not None

        cache.clear("alice")
        assert cache.get(old_key) is None
        assert cache.get(new_key) is None
