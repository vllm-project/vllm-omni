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
    def test_slices_batched_recordings_by_request(self, backend):
        first = make_req(prompt="first")
        second = make_req(prompt="second", num_outputs_per_prompt=2)
        latents = torch.tensor([[1.0], [2.0], [3.0]])
        backend.before_diffuse()
        backend.recorder.record(0, 10.0, latents)
        outputs = [DiffusionOutput(output=latents[:1]), DiffusionOutput(output=latents[1:])]
        backend.post_forward_store([first, second], outputs, None, None)

        assert torch.equal(backend.lookup_step_latents(first)[0].latent, latents[:1])
        assert torch.equal(backend.lookup_step_latents(second)[0].latent, latents[1:])
        hits, remaining = backend.short_circuit_requests([first, second], target_device=None)
        assert [index for index, _ in hits] == [0, 1]
        assert not remaining

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
    @pytest.mark.parametrize(
        ("steps", "similarity", "expected"),
        [(1, 1.0, 0), (5, 1.0, 2), (8, 0.8, 4), (20, 0.74, 0), (20, 0.75, 5), (20, 1.0, 10)],
    )
    def test_skip_bounds(self, backend, steps, similarity, expected):
        assert backend.compute_skip_steps(similarity, steps) == expected

    def test_skip_leaves_work_at_full_ratio_and_threshold(self, backend):
        backend._clip_max_skip_ratio = 1.0
        backend._clip_threshold = 1.0
        assert backend.compute_skip_steps(1.0, 5) == 4
        backend._cache_store._max_stored_steps = 2
        assert backend.compute_skip_steps(1.0, 5) == 2

    def test_resume_uses_step_latents(self, backend):
        from vllm_omni.diffusion.cache.inter_request.cache_store import (
            StepLatentData,
            build_cache_key_from_request,
        )

        req = make_req()
        steps = [StepLatentData(step_index=i, timestep=float(20 - i), latent=torch.full((1, 2), i)) for i in range(6)]
        key = build_cache_key_from_request(req, backend._pipeline, model_digest=backend._model_digest)
        backend._cache_store.put(key, torch.full((2, 2), 5.0), step_latents=steps)

        req2 = make_req()
        req2.sampling_params.resume_from_step = 3
        hits, remaining = backend.short_circuit_requests([req2], target_device=None)
        # resume requests stay in the batch with resume_latents populated
        assert hits == []
        assert len(remaining) == 1
        assert remaining[0].sampling_params.resume_latents is not None
        assert torch.equal(remaining[0].sampling_params.resume_latents, torch.full((1, 2), 2.0))

    def test_resume_rejects_a_batch_recorded_under_one_request(self, backend):
        from vllm_omni.diffusion.cache.inter_request.cache_store import StepLatentData

        req = make_req(resume_from_step=1)
        backend.store(req, torch.ones(1, 2), step_latents=[StepLatentData(0, 20.0, torch.ones(2, 2))])
        hits, remaining = backend.short_circuit_requests([req], target_device=None)
        assert hits == [] and remaining == [req]
        assert req.sampling_params.resume_from_step == 0
        assert req.sampling_params.resume_latents is None


class TestEnableGuards:
    def test_pipeline_without_step_hook_bypasses_cache(self):
        b = InterRequestCacheBackend(DiffusionCacheConfig())
        b.enable(SimpleNamespace())
        try:
            assert not b.is_enabled()
            reqs = [make_req()]
            assert b.short_circuit_requests(reqs, target_device=None) == ([], reqs)
        finally:
            b.shutdown()

    @pytest.fixture
    def tiled_pipeline(self):
        from vllm_omni.diffusion.data import DiffusionParallelConfig, OmniDiffusionConfig

        pipeline = FakePipeline()
        pipeline.od_config = OmniDiffusionConfig(
            parallel_config=DiffusionParallelConfig(tensor_parallel_size=2, vae_patch_parallel_size=2),
        )
        return pipeline

    def test_tiled_vae_disables_backend(self, tiled_pipeline, monkeypatch):
        from unittest.mock import Mock

        from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner

        b = InterRequestCacheBackend(DiffusionCacheConfig())
        b.enable(tiled_pipeline)
        assert b.enabled is False
        assert b.recorder is None
        assert tiled_pipeline._diffuse_step_hooks == []
        for is_dummy in (True, False):
            b.before_diffuse(is_dummy=is_dummy)
            b.after_diffuse(is_dummy=is_dummy)
        hits, remaining = b.short_circuit_requests([make_req()], target_device=None)
        assert hits == [] and len(remaining) == 1
        shutdown = Mock(wraps=b.shutdown)
        monkeypatch.setattr(b, "shutdown", shutdown)
        DiffusionModelRunner.shutdown(SimpleNamespace(cache_backend=b))
        shutdown.assert_called_once_with()

    def test_tiled_vae_preserves_composite_refresh(self, tiled_pipeline, monkeypatch):
        from unittest.mock import Mock

        from vllm_omni.diffusion.cache.composite_backend import CompositeCacheBackend
        from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner

        b = CompositeCacheBackend(DiffusionCacheConfig())
        dit = b.cache_dit_backend
        monkeypatch.setattr(dit, "enable", lambda pipeline: setattr(dit, "enabled", True))
        refresh = Mock()
        monkeypatch.setattr(dit, "refresh", refresh)
        b.enable(tiled_pipeline)
        assert b.recorder is None
        assert b.is_enabled()
        runner = SimpleNamespace(cache_backend=b, pipeline=tiled_pipeline)
        DiffusionModelRunner._refresh_cache_for_requests(runner, [make_req()], od_config=tiled_pipeline.od_config)
        refresh.assert_called_once_with(tiled_pipeline, 20, True)
        b.shutdown()


class TestPostForwardStoreGuards:
    def test_empty_tensor_placeholder_not_stored(self, backend):
        req = make_req()
        out = DiffusionOutput(output=torch.zeros(0))  # non-owner rank placeholder
        backend.post_forward_store([req], [out], target_device=None, runner=None)
        hits, remaining = backend.short_circuit_requests([make_req()], target_device=None)
        assert hits == [] and len(remaining) == 1

    def test_media_output_not_stored(self, backend):
        # Media-typed outputs carry output=None (payload on .media); the store
        # guard must skip them. DiffusionOutput validates .media's type, so we
        # exercise the output=None path directly.
        req = make_req()
        out = DiffusionOutput(output=None)
        backend.post_forward_store([req], [out], target_device=None, runner=None)
        hits, remaining = backend.short_circuit_requests([make_req()], target_device=None)
        assert hits == [] and len(remaining) == 1


class TestCollateResumeState:
    def _batch(self, specs):
        from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

        reqs = []
        for i, (step, latents) in enumerate(specs):
            sp = SimpleNamespace(
                resume_from_step=step,
                resume_latents=latents,
                generator=None,
            )
            reqs.append(SimpleNamespace(request_id=f"r{i}", prompt="p", sampling_params=sp))
        return DiffusionRequestBatch(requests=reqs)

    def test_single_resume(self):
        batch = self._batch([(3, torch.zeros(1, 2, 2))])
        step, latents = batch.collate_resume_state()
        assert step == 3
        assert latents.shape == (1, 2, 2)

    def test_all_zero_returns_none(self):
        batch = self._batch([(0, None), (0, None)])
        assert batch.collate_resume_state() == (0, None)

    def test_equal_steps_collate(self):
        batch = self._batch([(3, torch.zeros(1, 2, 2)), (3, torch.ones(1, 2, 2))])
        step, latents = batch.collate_resume_state()
        assert step == 3
        assert latents.shape == (2, 2, 2)
        assert torch.equal(latents[1], torch.ones(2, 2))

    def test_mixed_steps_fallback_to_full(self):
        batch = self._batch([(3, torch.zeros(1, 2, 2)), (0, None)])
        assert batch.collate_resume_state() == (0, None)

    def test_missing_latents_fallback(self):
        batch = self._batch([(3, None), (3, torch.zeros(1, 2, 2))])
        assert batch.collate_resume_state() == (0, None)
