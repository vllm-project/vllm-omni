# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import logging
from types import SimpleNamespace

import pytest
import torch
from diffusers import FlowMatchEulerDiscreteScheduler, FluxPipeline, FluxTransformer2DModel

from tests.diffusion.diffusion_backend.test_diffusers_backend import _make_od_config
from vllm_omni.diffusion.cache.cachedit.backend import CacheDiTBackend
from vllm_omni.diffusion.data import DiffusionCacheConfig
from vllm_omni.diffusion.models.diffusers_adapter import DiffusersAdapterPipeline
from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def make_pipeline():
    torch.manual_seed(42)
    transformer = FluxTransformer2DModel(
        in_channels=16,
        num_layers=2,
        num_single_layers=2,
        attention_head_dim=16,
        num_attention_heads=2,
        joint_attention_dim=32,
        pooled_projection_dim=32,
        axes_dims_rope=(4, 6, 6),
    )
    return FluxPipeline(
        scheduler=FlowMatchEulerDiscreteScheduler(),
        vae=None,
        text_encoder=None,
        tokenizer=None,
        text_encoder_2=None,
        tokenizer_2=None,
        transformer=transformer,
    )


def test_real_diffusers_cache_lifecycle():
    adapter = DiffusersAdapterPipeline(od_config=_make_od_config(cache_backend="cache_dit"))
    adapter._pipeline = make_pipeline()
    adapter._pipeline.set_progress_bar_config(disable=True)
    pipeline = adapter._pipeline
    torch.manual_seed(42)
    prompt = torch.randn(1, 8, 32)
    pooled = torch.randn(1, 32)
    latents = torch.randn(1, 4, 16)

    def run(steps):
        return pipeline(
            prompt_embeds=prompt,
            pooled_prompt_embeds=pooled,
            latents=latents.clone(),
            height=32,
            width=32,
            num_inference_steps=steps,
            output_type="latent",
        ).images

    with torch.inference_mode():
        reference = run(8)
    original_call = type(pipeline).__call__
    backend = CacheDiTBackend(
        DiffusionCacheConfig(
            Fn_compute_blocks=1,
            Bn_compute_blocks=0,
            max_warmup_steps=1,
            residual_diff_threshold=100.0,
            max_continuous_cached_steps=2,
        )
    )
    backend.enable(adapter)
    try:
        assert backend.is_enabled()
        assert type(pipeline).__call__ is not original_call
        assert adapter._cache_dit_targets == (pipeline,)
        with torch.inference_mode():
            backend.refresh(adapter, 8, verbose=False)
            first = run(8)
            backend.refresh(adapter, 12, verbose=False)
            different_steps = run(12)
            backend.refresh(adapter, 8, verbose=False)
            repeated = run(8)
        assert first.isfinite().all() and different_steps.isfinite().all()
        assert not torch.equal(first, reference), "High threshold should exercise cached blocks"
        torch.testing.assert_close(first, repeated, atol=0, rtol=0)
    finally:
        backend.disable(adapter)
    assert type(pipeline).__call__ is original_call
    assert not backend.is_enabled()
    with torch.inference_mode():
        restored = run(8)
    torch.testing.assert_close(restored, reference, atol=0, rtol=0)


def test_diffusers_delegation_refresh_is_inert():
    """The runner must not ask this delegation to resolve request step counts."""
    adapter = DiffusersAdapterPipeline(od_config=_make_od_config(cache_backend="cache_dit"))
    adapter._pipeline = make_pipeline()
    backend = CacheDiTBackend(DiffusionCacheConfig(residual_diff_threshold=100.0))
    assert backend.requires_request_refresh
    backend.enable(adapter)
    try:
        assert backend.requires_request_refresh is False
    finally:
        backend.disable(adapter)
    assert backend.requires_request_refresh


def test_runner_does_not_warn_about_missing_steps_for_the_delegation(caplog):
    """A request that omits num_inference_steps is normal here, not a failed refresh."""
    adapter = DiffusersAdapterPipeline(od_config=_make_od_config(cache_backend="cache_dit"))
    adapter._pipeline = make_pipeline()
    backend = CacheDiTBackend(DiffusionCacheConfig(residual_diff_threshold=100.0))
    backend.enable(adapter)
    try:
        runner = SimpleNamespace(cache_backend=backend, pipeline=adapter)
        od_config = SimpleNamespace(cache_backend="cache_dit")
        request = SimpleNamespace(
            sampling_params=SimpleNamespace(num_inference_steps=None, timesteps=None, sigmas=None)
        )
        with caplog.at_level(logging.WARNING):
            DiffusionModelRunner._refresh_cache_for_requests(runner, [request], od_config=od_config)
        assert "Failed to refresh" not in caplog.text
    finally:
        backend.disable(adapter)


def test_diffusers_cache_rejects_scm():
    adapter = DiffusersAdapterPipeline(od_config=_make_od_config(cache_backend="cache_dit"))
    backend = CacheDiTBackend(DiffusionCacheConfig(scm_steps_mask_policy="slow"))
    with pytest.raises(ValueError, match="SCM"):
        backend.enable(adapter)
    assert not backend.is_enabled()
