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
        # steps went through the fake engine
        assert f"{key.to_hash()}:steps_meta" in engine.data
        assert f"{key.to_hash()}:step_0000" in engine.data

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
