# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for the inter-request cache backend polymorphic hooks (CPU-only).

Exercises InterRequestCacheBackend.short_circuit_requests /
post_forward_store / merge_hit_outputs with fake requests and pipelines:
- exact hit short-circuits forward
- seedless / image-conditioned requests bypass the cache
- semantic-hit resume path (with a stubbed CLIP) sets resume_from_step
"""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.cache.inter_request.backend import InterRequestCacheBackend
from vllm_omni.diffusion.data import DiffusionCacheConfig, DiffusionOutput

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class FakePipeline:
    def __init__(self):
        self._diffuse_step_hooks = []

    def register_diffuse_step_hook(self, hook):
        self._diffuse_step_hooks.append(hook)


def make_req(prompt="a cat", seed=42, **over):
    sp = dict(
        height=480,
        width=832,
        num_inference_steps=20,
        guidance_scale=4.0,
        guidance_scale_provided=True,
        true_cfg_scale=1.0,
        seed=seed,
        generator=None,
        sigmas=None,
        max_sequence_length=None,
        num_outputs_per_prompt=1,
        num_frames=1,
        resume_from_step=0,
        resume_latents=None,
    )
    sp.update(over)
    return SimpleNamespace(prompt=prompt, sampling_params=SimpleNamespace(**sp))


@pytest.fixture
def backend():
    cfg = DiffusionCacheConfig()  # no LMCache dir, no CLIP → exact hits only
    b = InterRequestCacheBackend(cfg)
    b.enable(FakePipeline())
    return b


def store_one(backend, req, tensor=None):
    out = DiffusionOutput(output=tensor if tensor is not None else torch.full((2, 2), 5.0))
    backend.post_forward_store([req], [out], target_device=None, runner=None)
    return out


class TestShortCircuit:
    def test_exact_hit_skips_forward(self, backend):
        req = make_req()
        store_one(backend, req)

        hits, remaining = backend.short_circuit_requests([make_req()], target_device=None)
        assert len(hits) == 1
        assert remaining == []
        idx, hit_out = hits[0]
        assert idx == 0
        assert torch.equal(hit_out.output, torch.full((2, 2), 5.0))

    def test_miss_returns_request(self, backend):
        store_one(backend, make_req(prompt="first"))
        hits, remaining = backend.short_circuit_requests([make_req(prompt="different")], target_device=None)
        assert hits == []
        assert len(remaining) == 1

    def test_seedless_request_bypasses_cache(self, backend):
        req = make_req(seed=None)
        # store is a no-op for seedless (key build refuses)
        store_one(backend, req, torch.full((2, 2), 9.0))
        hits, remaining = backend.short_circuit_requests([make_req(seed=None)], target_device=None)
        assert hits == []
        assert len(remaining) == 1

    def test_image_conditioned_request_bypasses_cache(self, backend):
        req = make_req(prompt={"prompt": "edit", "multi_modal_data": {"image": "a.png"}})
        store_one(backend, req, torch.full((2, 2), 1.0))
        hits, remaining = backend.short_circuit_requests([req], target_device=None)
        assert hits == []
        assert len(remaining) == 1


class TestPostForwardStore:
    def test_store_then_exact_hit(self, backend):
        store_one(backend, make_req())
        hits, remaining = backend.short_circuit_requests([make_req()], target_device=None)
        assert len(hits) == 1 and remaining == []

    def test_post_forward_noop_when_disabled(self, backend):
        backend.enabled = False
        out = DiffusionOutput(output=torch.zeros(1))
        result = backend.post_forward_store([make_req()], [out], None, None)
        assert result is not None  # outputs returned unchanged


class TestMergeHitOutputs:
    def test_merge_preserves_positions(self, backend):
        # Original batch had 3 requests: idx0/idx2 computed, idx1 was a hit.
        # merge restores the original batch length and positions.
        computed = [
            DiffusionOutput(output=torch.tensor([1.0])),
            DiffusionOutput(output=torch.tensor([2.0])),
        ]
        hits = [(1, DiffusionOutput(output=torch.tensor([99.0])))]  # hit at index 1
        merged = backend.merge_hit_outputs(computed, hits)
        assert len(merged) == 3
        assert torch.equal(merged[0].output, torch.tensor([1.0]))
        assert torch.equal(merged[1].output, torch.tensor([99.0]))
        assert torch.equal(merged[2].output, torch.tensor([2.0]))

    def test_no_hits_passthrough(self, backend):
        computed = [DiffusionOutput(output=torch.tensor([1.0]))]
        assert backend.merge_hit_outputs(computed, []) is computed


class TestResumePath:
    def test_resume_uses_step_latents(self, backend):
        from vllm_omni.diffusion.cache.inter_request.cache_store import (
            StepLatentData,
            build_cache_key_from_request,
        )

        req = make_req()
        steps = [StepLatentData(step_index=i, timestep=float(20 - i), latent=torch.full((2, 2), i)) for i in range(6)]
        key = build_cache_key_from_request(req, backend._pipeline, model_digest=backend._model_digest)
        backend._cache_store.put(key, torch.full((2, 2), 5.0), step_latents=steps)

        req2 = make_req()
        req2.sampling_params.resume_from_step = 3
        hits, remaining = backend.short_circuit_requests([req2], target_device=None)
        # resume requests stay in the batch with resume_latents populated
        assert hits == []
        assert len(remaining) == 1
        assert remaining[0].sampling_params.resume_latents is not None
        assert torch.equal(remaining[0].sampling_params.resume_latents, torch.full((2, 2), 2.0))
