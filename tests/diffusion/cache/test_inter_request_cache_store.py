# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for the inter-request DiT cache store (CPU-only).

Covers the defect classes found in review:
- cache-key construction: prompt forms, seedless requests, image conditioning,
  model-digest namespacing
- semantic search: hybrid/text-only groups, threshold and dimension filters
- eviction + memory accounting (plain and LMCache-shell modes with a fake engine)
- LMCache recovery of step latents from a shell
- persistence round-trip (save_to_disk / load_from_disk)
"""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.cache.inter_request.cache_store import (
    CacheKey,
    DiTCacheStore,
    StepLatentData,
    build_cache_key_from_request,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def make_key(prompt="a cat", seed=42, **kw):
    defaults = dict(
        negative_prompt="",
        height=480,
        width=832,
        num_inference_steps=40,
        guidance_scale=4.0,
        true_cfg_scale=1.0,
        sigmas=None,
        max_sequence_length=None,
        num_images_per_prompt=1,
        num_frames=81,
    )
    defaults.update(kw)
    return CacheKey(prompt=prompt, seed=seed, **defaults)


def make_req(
    prompt="a cat",
    seed=42,
    generator=None,
    height=480,
    width=832,
    steps=40,
    frames=81,
):
    return SimpleNamespace(
        prompt=prompt,
        sampling_params=SimpleNamespace(
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=4.0,
            guidance_scale_provided=True,
            true_cfg_scale=1.0,
            seed=seed,
            generator=generator,
            sigmas=None,
            max_sequence_length=None,
            num_outputs_per_prompt=1,
            num_frames=frames,
        ),
    )


class FakePipeline:
    pass


class TestBuildCacheKey:
    def test_str_prompt_with_seed(self):
        key = build_cache_key_from_request(make_req(), FakePipeline())
        assert key is not None
        assert key.seed == 42

    def test_dict_prompt_with_negative(self):
        key = build_cache_key_from_request(
            make_req(prompt={"prompt": "a cat", "negative_prompt": "blurry"}),
            FakePipeline(),
        )
        assert key is not None
        assert key.negative_prompt == "blurry"

    def test_seedless_request_returns_none(self):
        # Unseeded requests must not be cached: identical prompts would all
        # hit the first request's output and become deterministic.
        assert build_cache_key_from_request(make_req(seed=None), FakePipeline()) is None

    def test_seedless_with_generator_resolves_seed(self):
        gen = torch.Generator().manual_seed(1234)
        key = build_cache_key_from_request(make_req(seed=None, generator=gen), FakePipeline())
        assert key is not None
        assert key.seed == 1234

    def test_image_conditioned_prompt_returns_none(self):
        # edit / i2v requests carry multi_modal_data; the conditioning image is
        # not part of the key so reuse must be refused.
        req = make_req(prompt={"prompt": "edit this", "multi_modal_data": {"image": "x.png"}})
        assert build_cache_key_from_request(req, FakePipeline()) is None

    def test_prompt_embeds_returns_none(self):
        req = make_req(prompt={"prompt": "x", "prompt_embeds": torch.zeros(1, 8)})
        assert build_cache_key_from_request(req, FakePipeline()) is None

    def test_non_text_prompt_returns_none(self):
        # OmniTokensPrompt / structured prompt objects have no text identity.
        req = make_req(prompt={"prompt_token_ids": [1, 2, 3]})
        assert build_cache_key_from_request(req, FakePipeline()) is None
        req = make_req(prompt=SimpleNamespace(kind="custom"))
        assert build_cache_key_from_request(req, FakePipeline()) is None

    def test_model_digest_namespaces_keys(self):
        k1 = build_cache_key_from_request(make_req(), FakePipeline(), model_digest="aaa")
        k2 = build_cache_key_from_request(make_req(), FakePipeline(), model_digest="bbb")
        assert k1.to_hash() != k2.to_hash()
        # Same digest → stable hash
        k3 = build_cache_key_from_request(make_req(), FakePipeline(), model_digest="aaa")
        assert k1.to_hash() == k3.to_hash()


class TestEvictionAndAccounting:
    def test_plain_eviction_drops_oldest_entirely(self):
        store = DiTCacheStore(max_entries=2, max_memory_gb=1.0)
        store.put(make_key(prompt="a"), torch.zeros(1))
        store.put(make_key(prompt="b"), torch.zeros(1))
        store.put(make_key(prompt="c"), torch.zeros(1))
        assert store.size == 2
        # "a" was evicted entirely
        assert store.get(make_key(prompt="a")) is None
        assert store.get(make_key(prompt="c")) is not None

    def test_memory_budget_evicts(self):
        big = torch.zeros(1024, 1024)  # 4 MB
        store = DiTCacheStore(max_entries=10, max_memory_gb=0.00002)  # ~84 bytes
        store.put(make_key(prompt="a"), big.clone())
        store.put(make_key(prompt="b"), big.clone())
        # budget exceeded: first entry dropped
        assert store.get(make_key(prompt="a")) is None
        assert store.stats()["memory_mb"] >= 0

    def test_lmcache_mode_keeps_shell(self):
        # Zero CPU budget in LMCache mode: heavy tensors drop on every put
        # (they are tiered to disk), but the embedding shell stays searchable.
        store = DiTCacheStore(max_entries=10, max_memory_gb=0.0, lmcache_engine=object())
        store.put(make_key(prompt="a"), torch.zeros(2, 2))
        entry = next(iter(store._store.values()))
        assert entry.latents is None
        assert entry.cache_key is not None
        # shell remains present (searchable)
        assert store.size == 1


class FakeLMCache:
    """Minimal in-memory stand-in for ECCacheEngine."""

    def __init__(self):
        self.data = {}

    def put(self, key, tensor):
        self.data[key] = tensor.detach().clone()
        return True

    def get(self, key, device="cpu"):
        return self.data.get(key)


class TestLMCacheStepRecovery:
    def test_get_step_latents_rebuilds_from_meta(self):
        engine = FakeLMCache()
        store = DiTCacheStore(max_entries=10, max_memory_gb=1.0, lmcache_engine=engine)
        steps = [StepLatentData(step_index=i, timestep=float(40 - i), latent=torch.full((2, 2), i)) for i in range(3)]
        key = make_key(prompt="vid")
        store.put(key, torch.zeros(2, 2), step_latents=steps)
        store.close()
        generation = store._store[key.to_hash()].step_storage_key
        assert f"{generation}:steps_meta" in engine.data
        assert f"{generation}:step_0000" in engine.data

        # Simulate eviction to shell
        entry = store._store[key.to_hash()]
        entry.step_latents = None

        recovered = store.get_step_latents(key)
        assert recovered is not None
        assert [s.step_index for s in recovered] == [0, 1, 2]
        assert torch.equal(recovered[1].latent, torch.full((2, 2), 1.0))

    def test_shell_recovery_of_final_latent_requires_disk(self):
        # Final latents are torch.save-only; without final_disk_dir a shell
        # with a missing file must NOT recover from LMCache (no :final key).
        engine = FakeLMCache()
        store = DiTCacheStore(max_entries=10, max_memory_gb=1.0, lmcache_engine=engine)
        key = make_key(prompt="x")
        store.put(key, torch.full((2, 2), 7.0))
        entry = store._store[key.to_hash()]
        entry.latents = None
        assert store.get(key) is None  # no final_direct dir → miss, no crash


class TestSemanticSearch:
    def _store_with_entries(self):
        store = DiTCacheStore(max_entries=10, max_memory_gb=1.0)

        def emb(v):
            t = torch.zeros(8)
            t[0] = v
            return t / t.norm()

        store.put(make_key(prompt="cat on sofa"), torch.zeros(2), clip_embedding=emb(1.0))
        store.put(make_key(prompt="dog on sofa"), torch.zeros(2), clip_embedding=emb(-1.0))
        return store, emb

    def test_best_match_and_threshold(self):
        store, emb = self._store_with_entries()
        latents, steps, sim, cached_prompt, _ = store.semantic_search(emb(0.9), threshold=0.5)
        assert latents is not None
        assert cached_prompt == "cat on sofa"
        assert sim > 0.9

    def test_below_threshold_misses(self):
        store, emb = self._store_with_entries()
        # orthogonal query
        q = torch.zeros(8)
        q[1] = 1.0
        latents, _, sim, _, _ = store.semantic_search(q, threshold=0.99)
        assert latents is None

    def test_dimension_filter_excludes_mismatched_entries(self):
        store, emb = self._store_with_entries()
        # require a height that no entry has
        latents, _, _, _, _ = store.semantic_search(emb(1.0), threshold=0.1, required_height=9999)
        assert latents is None

    def test_hybrid_group_when_image_embedding_present(self):
        store, emb = self._store_with_entries()
        dog_hash = next(kh for kh, e in store._store.items() if e.cache_key.prompt == "dog on sofa")
        img_emb = torch.zeros(8)
        img_emb[0] = 1.0
        img_emb = img_emb / img_emb.norm()
        assert store.update_image_embedding(dog_hash, img_emb)

        # Query aligned with the dog's *image* embedding and anti-aligned with
        # its text embedding: hybrid score = t2t * sigmoid_penalty(t2i) < t2t.
        latents, _, sim, prompt, mtype = store.semantic_search(emb(0.9), threshold=0.1)
        # the cat entry (text-aligned, no image) wins with its full t2t score
        assert prompt == "cat on sofa"
        assert mtype == "text-text"


class TestPersistenceRoundTrip:
    def test_save_load_roundtrip(self, tmp_path):
        store = DiTCacheStore(max_entries=10, max_memory_gb=1.0)
        key = make_key(prompt="persist me")
        # Embeddings are L2-normalized at encode time in the production path;
        # store one the same way here.
        emb = torch.zeros(8) + 1e-3
        emb = emb / emb.norm()
        store.put(key, torch.full((2, 2), 3.0), clip_embedding=emb)
        n = store.save_to_disk(tmp_path)
        assert n == 1

        store2 = DiTCacheStore(max_entries=10, max_memory_gb=1.0)
        loaded = store2.load_from_disk(tmp_path)
        assert loaded == 1
        out = store2.get(key)
        assert out is not None
        assert torch.equal(out, torch.full((2, 2), 3.0))
        # restored embeddings remain searchable
        latents, _, _, prompt, _ = store2.semantic_search(emb, threshold=0.5)
        assert prompt == "persist me"


class TestMatrixGrowthAlignment:
    def test_churn_reuses_rows(self):
        store = DiTCacheStore(max_entries=1, max_memory_gb=1.0)
        store._ROW_CHUNK = 2
        text = torch.tensor([1.0, 0.0])
        latent = torch.zeros(2)
        for i in range(16):
            # Alternate overwriting the same key and evicting it for a new key.
            key = make_key(prompt=f"p{i // 2}")
            store.put(key, latent, clip_embedding=text)
            assert store._next_row == 1
            assert store._capacity == 2
            assert store._key_rows == {key.to_hash(): 0}
            assert store._img_matrix[0].count_nonzero().item() == 0
            assert store.semantic_search(text, threshold=0.9)[3] == key.prompt
            assert store.update_image_embedding(key.to_hash(), text)
        store.clear()
        store.put(make_key(prompt="fresh"), latent, clip_embedding=text)
        assert store.semantic_search(text, threshold=0.9)[3] == "fresh"

    def test_row_keys_aligned_past_chunk_growth(self):
        """Rows must stay aligned with _row_keys after capacity growth (P1).

        _ROW_CHUNK is 512; the 513rd insertion grows the matrices and pads
        _row_keys. Keys must be assigned by matrix row, not appended.
        """
        store = DiTCacheStore(max_entries=10_000, max_memory_gb=1.0)

        def basis(i, dim=600):
            t = torch.zeros(dim)
            t[i] = 1.0
            return t

        keys = []
        for i in range(530):
            k = make_key(prompt=f"p{i}")
            store.put(k, torch.zeros(2, 2), clip_embedding=basis(i))
            keys.append(k)
        assert store._next_row == 530
        assert len(store._row_keys) == store._capacity
        # Every key sits at its matrix row.
        for row, k in enumerate(keys):
            assert store._row_keys[row] == k.to_hash()
            assert store._key_rows[k.to_hash()] == row
        # The 513rd entry (first after growth) is searchable and matches itself.
        q = basis(512)
        latents, _, sim, prompt, _ = store.semantic_search(q, threshold=0.9)
        assert prompt == "p512"
        assert sim > 0.99


class TestKeyIdentity:
    def _req(self, **over):
        return make_req(**over)

    def test_lora_digest_separates_keys(self):
        from types import SimpleNamespace

        lora = SimpleNamespace(lora_name="style", lora_int_id=7)
        k_plain = build_cache_key_from_request(self._req(), FakePipeline())
        k_lora = build_cache_key_from_request(
            self._req(),
            FakePipeline(),
        )
        # same request → same key
        assert k_plain.to_hash() == k_lora.to_hash()
        req = make_req()
        req.sampling_params.lora_request = lora
        req.sampling_params.lora_scale = 0.8
        k_with = build_cache_key_from_request(req, FakePipeline())
        assert k_with.lora_digest == "style:7:0.8"
        assert k_with.to_hash() != k_plain.to_hash()

    def test_output_type_separates_keys(self):
        req_img = make_req()
        req_img.sampling_params.output_type = "default"
        req_lat = make_req()
        req_lat.sampling_params.output_type = "latent"
        k_img = build_cache_key_from_request(req_img, FakePipeline())
        k_lat = build_cache_key_from_request(req_lat, FakePipeline())
        assert k_img.output_type == "default"
        assert k_lat.output_type == "latent"
        assert k_img.to_hash() != k_lat.to_hash()


class TestSemanticScheduleAndLoraFilters:
    def _store(self):
        store = DiTCacheStore(max_entries=10, max_memory_gb=1.0)

        def emb(v):
            t = torch.zeros(8)
            t[0] = v
            return t / t.norm()

        return store, emb

    def test_different_step_count_rejected(self):
        store, emb = self._store()
        store.put(make_key(prompt="a", num_inference_steps=40), torch.zeros(2), clip_embedding=emb(1.0))
        # 20-step request must NOT match a 40-step trajectory anymore
        lat, _, _, _, _ = store.semantic_search(emb(1.0), threshold=0.5, required_num_inference_steps=20)
        assert lat is None
        # same step count still matches
        lat, _, _, prompt, _ = store.semantic_search(emb(1.0), threshold=0.5, required_num_inference_steps=40)
        assert prompt == "a"

    def test_sigmas_mismatch_rejected(self):
        store, emb = self._store()
        store.put(
            make_key(prompt="a", sigmas=tuple([0.9, 0.5, 0.1])),
            torch.zeros(2),
            clip_embedding=emb(1.0),
        )
        lat, _, _, _, _ = store.semantic_search(
            emb(1.0),
            threshold=0.5,
            required_num_inference_steps=40,
            required_sigmas=(0.9, 0.5, 0.2),
        )
        assert lat is None
        lat, _, _, prompt, _ = store.semantic_search(
            emb(1.0),
            threshold=0.5,
            required_num_inference_steps=40,
            required_sigmas=(0.9, 0.5, 0.1),
        )
        assert prompt == "a"

    def test_lora_digest_mismatch_rejected(self):
        store, emb = self._store()
        store.put(
            make_key(prompt="a", lora_digest="loraA:1:0.5"),
            torch.zeros(2),
            clip_embedding=emb(1.0),
        )
        lat, _, _, _, _ = store.semantic_search(emb(1.0), threshold=0.5, required_lora_digest="loraB:2:0.5")
        assert lat is None
        lat, _, _, prompt, _ = store.semantic_search(emb(1.0), threshold=0.5, required_lora_digest="loraA:1:0.5")
        assert prompt == "a"


class TestTieredReadNonRetention:
    def test_get_demotes_recovered_final(self, tmp_path):
        engine = FakeLMCache()
        store = DiTCacheStore(
            max_entries=10,
            max_memory_gb=1.0,
            lmcache_engine=engine,
            final_disk_dir=str(tmp_path),
        )
        key = make_key(prompt="x")
        store.put(key, torch.full((2, 2), 7.0))
        store.close()  # Drain tier writes before simulating eviction.

        entry = store._store[key.to_hash()]
        entry.latents = None  # simulate eviction to shell

        out = store.get(key)
        assert out is not None and torch.equal(out, torch.full((2, 2), 7.0))
        # must NOT stay resident in tiered mode
        assert entry.latents is None

    def test_get_step_latents_demotes_recovered_steps(self):
        engine = FakeLMCache()
        store = DiTCacheStore(max_entries=10, max_memory_gb=1.0, lmcache_engine=engine)
        steps = [StepLatentData(step_index=0, timestep=40.0, latent=torch.ones(2, 2))]
        key = make_key(prompt="vid")
        store.put(key, torch.zeros(2, 2), step_latents=steps)
        store.close()
        entry = store._store[key.to_hash()]
        entry.step_latents = None

        recovered = store.get_step_latents(key)
        assert recovered is not None and len(recovered) == 1
        assert entry.step_latents is None  # demoted after recovery


class TestBoundedSimilarityStats:
    def test_stats_bounded(self):
        from vllm_omni.diffusion.cache.inter_request.cache_store import _SIM_STATS_MAX_SAMPLES

        store = DiTCacheStore(max_entries=10, max_memory_gb=1.0)
        store._all_sims.extend([0.5] * (_SIM_STATS_MAX_SAMPLES + 5000))
        assert len(store._all_sims) == _SIM_STATS_MAX_SAMPLES
        stats = store.get_similarity_stats()
        assert stats["final_sim"]["total_comparisons"] == _SIM_STATS_MAX_SAMPLES


class TestTierWriteConcurrency:
    def test_blocked_write_allows_reads_and_skips_saturated_queue(self, tmp_path):
        from concurrent.futures import ThreadPoolExecutor
        from threading import Event

        started, resume = Event(), Event()

        class SlowCache(FakeLMCache):
            def put(self, key, tensor):
                started.set()
                assert resume.wait(10), "tier writer was never released"
                return super().put(key, tensor)

        store = DiTCacheStore(max_memory_gb=0, lmcache_engine=SlowCache(), final_disk_dir=str(tmp_path))
        keys = [make_key(prompt=f"generation-{i}") for i in range(3)]
        steps = [StepLatentData(0, 1.0, torch.ones(1))]
        try:
            store.put(keys[0], torch.zeros(1), steps)
            assert started.wait(3)
            with ThreadPoolExecutor(max_workers=1) as pool:
                try:
                    assert pool.submit(store.get, keys[0]).result(timeout=2) is None
                    for key in keys[1:]:
                        pool.submit(store.put, key, torch.ones(1), steps).result(timeout=2)
                finally:
                    resume.set()
        finally:
            resume.set()
            store.close()
        assert store.get(keys[0]).item() == 0
        assert store.get(keys[1]).item() == 1
        assert store.get(keys[2]) is None

    def test_failed_overwrite_does_not_expose_old_or_partial_tier(self, tmp_path):
        class RejectNewMeta(FakeLMCache):
            fail = False

            def put(self, key, tensor):
                if self.fail and key.endswith(":steps_meta"):
                    return False
                return super().put(key, tensor)

        engine = RejectNewMeta()
        store = DiTCacheStore(max_memory_gb=0, lmcache_engine=engine, final_disk_dir=str(tmp_path))
        key = make_key()
        store.put(key, torch.ones(1), [StepLatentData(0, 1, torch.ones(1))])
        store._tier_writer.submit(lambda: None).result(timeout=3)
        assert store.get(key).item() == 1
        engine.fail = True
        store.put(key, torch.zeros(1), [StepLatentData(0, 1, torch.zeros(1))])
        store.close()
        assert store.get(key) is None
        assert store.get_step_latents(key) is None

    def test_clear_during_write_and_replace_same_key(self, tmp_path, monkeypatch):
        import threading

        from vllm_omni.diffusion.cache.inter_request import cache_store

        entered, resume = threading.Event(), threading.Event()
        save = cache_store._atomic_tensor_save

        def slow_save(tensor, path):
            entered.set()
            assert resume.wait(5)
            save(tensor, path)

        monkeypatch.setattr(cache_store, "_atomic_tensor_save", slow_save)
        store = DiTCacheStore(max_memory_gb=0, lmcache_engine=FakeLMCache(), final_disk_dir=str(tmp_path))
        key = make_key()
        try:
            store.put(key, torch.ones(1))
            assert entered.wait(3)
            store.clear()
            store.put(key, torch.full((1,), 2))
            assert store.get(key) is None
        finally:
            resume.set()
            store.close()
        assert store.get(key).item() == 2

    def test_corrupt_disk_file_is_a_miss(self, tmp_path):
        store = DiTCacheStore(max_memory_gb=0, lmcache_engine=FakeLMCache(), final_disk_dir=str(tmp_path))
        key = make_key()
        store.put(key, torch.ones(1))
        store.close()
        (tmp_path / f"{key.to_hash()}.pt").write_bytes(b"truncated tensor")
        assert store.get(key) is None

    def test_interrupted_save_preserves_complete_snapshot(self, tmp_path, monkeypatch):
        store = DiTCacheStore()
        key = make_key()
        store.put(key, torch.ones(1))
        assert store.save_to_disk(tmp_path) == 1
        store.put(key, torch.zeros(1))

        def broken_save(value, path):
            path.write_bytes(b"partial")
            raise OSError("simulated write interruption")

        monkeypatch.setattr(torch, "save", broken_save)
        assert store.save_to_disk(tmp_path) == 0
        reopened = DiTCacheStore()
        assert reopened.load_from_disk(tmp_path) == 1
        assert reopened.get(key).item() == 1
        assert len(list(tmp_path.iterdir())) == 1

    def test_disk_only_step_recovery_releases_borrowed_buffers(self):
        from unittest.mock import Mock

        class DiskOnly(FakeLMCache):
            def __init__(self):
                super().__init__()
                self._storage_manager = SimpleNamespace(get=self.read_disk)
                self.releases = []

            def _make_cache_key(self, key):
                return key

            def get(self, key, device="cpu"):
                return None

            def read_disk(self, key, location):
                assert location == "LocalDiskBackend"
                buffer = self.data[key].clone()
                release = Mock(side_effect=buffer.zero_)
                self.releases.append(release)
                return SimpleNamespace(tensor=buffer, ref_count_down=release)

        engine = DiskOnly()
        store = DiTCacheStore(max_memory_gb=0, lmcache_engine=engine)
        key = make_key()
        store.put(key, torch.ones(1), [StepLatentData(0, 9.0, torch.full((1,), 7))])
        store.close()
        steps = store.get_step_latents(key)
        assert steps and steps[0].timestep == 9.0 and steps[0].latent.item() == 7
        assert len(engine.releases) == 2
        for release in engine.releases:
            release.assert_called_once_with()
