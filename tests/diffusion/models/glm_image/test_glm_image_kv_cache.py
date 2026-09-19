# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for GlmImageKVCache mode transitions and pipeline correctness.

Covers:
  - Cache API: store, get, set_mode, clear
  - Attention block behavior: READ concatenates cached KV, SKIP bypasses
  - Pipeline regression: unconditional CFG pass must use SKIP, not READ

Requires correct vllm version (see pyproject.toml). Run with:
  pytest tests/diffusion/models/glm_image/test_glm_image_kv_cache.py -v -m "core_model and cpu"
"""

from __future__ import annotations

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

NUM_LAYERS = 4
BATCH = 1
SEQ = 6
NUM_HEADS = 2
HEAD_DIM = 8


# ============================================================
# Lazy imports — avoid triggering vllm_omni platform detection
# at collection time (same pattern as test_image_kv_cache_manager.py)
# ============================================================


@pytest.fixture(scope="module")
def kv_cache_classes():
    from vllm_omni.diffusion.models.glm_image.glm_image_transformer import (
        GlmImageKVCache,
        GlmImageLayerKVCache,
        KVCacheMode,
    )

    return GlmImageKVCache, GlmImageLayerKVCache, KVCacheMode


# ============================================================
# Helpers
# ============================================================


def _make_filled_cache(glm_image_kv_cache_cls, num_tokens: int = SEQ, base: float = 1.0) -> object:
    """Build a GlmImageKVCache pre-filled with known values via WRITE pass."""
    cache = glm_image_kv_cache_cls(num_layers=NUM_LAYERS)
    cache.set_mode("write")
    for i in range(NUM_LAYERS):
        k = torch.full((BATCH, num_tokens, NUM_HEADS, HEAD_DIM), base + i, dtype=torch.float32)
        v = torch.full((BATCH, num_tokens, NUM_HEADS, HEAD_DIM), base + i + 0.5, dtype=torch.float32)
        cache[i].store(k, v)
    return cache


def _simulate_attention_kv(layer_cache, mode, current_k: torch.Tensor, current_v: torch.Tensor, kv_cache_mode_cls):
    """Reproduce lines 626-633 of glm_image_transformer.py (attention block KV cache logic)."""
    key, value = current_k.clone(), current_v.clone()
    if mode == kv_cache_mode_cls.WRITE:
        layer_cache.store(key, value)
    elif mode == kv_cache_mode_cls.READ:
        k_cached, v_cached = layer_cache.get()
        if k_cached is not None:
            key = torch.cat([k_cached, key], dim=1)
            value = torch.cat([v_cached, value], dim=1)
    # SKIP: no-op
    return key, value


# ============================================================
# Test 1: GlmImageKVCache API
# ============================================================


def test_set_mode_string_and_enum(kv_cache_classes):
    GlmImageKVCache, _, KVCacheMode = kv_cache_classes
    cache = GlmImageKVCache(num_layers=2)

    cache.set_mode("write")
    assert cache.mode == KVCacheMode.WRITE

    cache.set_mode("read")
    assert cache.mode == KVCacheMode.READ

    cache.set_mode("skip")
    assert cache.mode == KVCacheMode.SKIP

    cache.set_mode(KVCacheMode.WRITE)
    assert cache.mode == KVCacheMode.WRITE

    cache.set_mode(None)
    assert cache.mode is None


def test_set_mode_invalid_raises(kv_cache_classes):
    GlmImageKVCache, _, _ = kv_cache_classes
    cache = GlmImageKVCache(num_layers=2)
    with pytest.raises(ValueError, match="Invalid mode"):
        cache.set_mode("bad_mode")


def test_layer_store_and_get(kv_cache_classes):
    _, GlmImageLayerKVCache, _ = kv_cache_classes
    layer = GlmImageLayerKVCache()
    assert layer.is_empty

    k = torch.ones(BATCH, SEQ, NUM_HEADS, HEAD_DIM)
    v = torch.ones(BATCH, SEQ, NUM_HEADS, HEAD_DIM) * 2
    layer.store(k, v)

    k_out, v_out = layer.get()
    assert k_out is not None
    assert k_out.shape == (BATCH, SEQ, NUM_HEADS, HEAD_DIM)
    assert torch.allclose(k_out, k)
    assert torch.allclose(v_out, v)


def test_layer_store_accumulates(kv_cache_classes):
    """Second store concatenates along seq dim."""
    _, GlmImageLayerKVCache, _ = kv_cache_classes
    layer = GlmImageLayerKVCache()
    k1 = torch.ones(BATCH, 3, NUM_HEADS, HEAD_DIM)
    k2 = torch.ones(BATCH, 4, NUM_HEADS, HEAD_DIM) * 2
    layer.store(k1, k1)
    layer.store(k2, k2)

    k_out, _ = layer.get()
    assert k_out.shape[1] == 7


def test_clear_resets_cache_and_mode(kv_cache_classes):
    GlmImageKVCache, _, _ = kv_cache_classes
    cache = _make_filled_cache(GlmImageKVCache)
    cache.set_mode("read")
    cache.clear()

    assert cache.mode is None
    for i in range(NUM_LAYERS):
        assert cache[i].is_empty


def test_getitem_out_of_range(kv_cache_classes):
    GlmImageKVCache, _, _ = kv_cache_classes
    cache = GlmImageKVCache(num_layers=2)
    with pytest.raises(IndexError):
        _ = cache[2]


# ============================================================
# Test 2: Attention block KV cache behavior per mode
# ============================================================


def test_read_mode_prepends_cached_kv(kv_cache_classes):
    """READ: cached K/V must be prepended to current K/V."""
    GlmImageKVCache, _, KVCacheMode = kv_cache_classes
    cache = _make_filled_cache(GlmImageKVCache, num_tokens=SEQ, base=1.0)
    cache.set_mode("read")

    cur_k = torch.zeros(BATCH, 2, NUM_HEADS, HEAD_DIM)
    cur_v = torch.zeros(BATCH, 2, NUM_HEADS, HEAD_DIM)

    result_k, result_v = _simulate_attention_kv(cache[0], cache.mode, cur_k, cur_v, KVCacheMode)

    assert result_k.shape[1] == SEQ + 2, "READ must prepend cached tokens"
    assert torch.allclose(result_k[:, :SEQ], cache[0].k_cache)
    assert torch.allclose(result_k[:, SEQ:], cur_k)


def test_skip_mode_bypasses_cached_kv(kv_cache_classes):
    """SKIP: cached K/V must NOT be concatenated — current K/V passes through unchanged."""
    GlmImageKVCache, _, KVCacheMode = kv_cache_classes
    cache = _make_filled_cache(GlmImageKVCache, num_tokens=SEQ, base=1.0)
    cache.set_mode("skip")

    cur_k = torch.zeros(BATCH, 2, NUM_HEADS, HEAD_DIM)
    cur_v = torch.zeros(BATCH, 2, NUM_HEADS, HEAD_DIM)

    result_k, result_v = _simulate_attention_kv(cache[0], cache.mode, cur_k, cur_v, KVCacheMode)

    assert result_k.shape[1] == 2, "SKIP must not expand key sequence"
    assert torch.allclose(result_k, cur_k)
    assert torch.allclose(result_v, cur_v)


def test_write_mode_stores_does_not_concat(kv_cache_classes):
    """WRITE: stores current K/V into cache; returned key/value unchanged."""
    _, GlmImageLayerKVCache, KVCacheMode = kv_cache_classes
    layer = GlmImageLayerKVCache()
    assert layer.is_empty

    cur_k = torch.ones(BATCH, 3, NUM_HEADS, HEAD_DIM) * 5
    cur_v = torch.ones(BATCH, 3, NUM_HEADS, HEAD_DIM) * 6
    result_k, result_v = _simulate_attention_kv(layer, KVCacheMode.WRITE, cur_k, cur_v, KVCacheMode)

    k_stored, _ = layer.get()
    assert k_stored is not None
    assert torch.allclose(k_stored, cur_k)
    assert result_k.shape[1] == 3
    assert torch.allclose(result_k, cur_k)


# ============================================================
# Test 3: Pipeline regression — unconditional pass must use SKIP
# ============================================================


@pytest.fixture(scope="module")
def diffuse_func():
    from vllm_omni.diffusion.models.glm_image.pipeline_glm_image import GlmImagePipeline

    return GlmImagePipeline.diffuse


def test_diffuse_sequential_cfg_kv_cache_modes(kv_cache_classes, diffuse_func):
    """
    Exercise GlmImagePipeline.diffuse() with a mock transformer.

    Verifies that for each denoising step:
      - conditional pass sees mode READ
      - unconditional pass sees mode SKIP

    Prior to the fix, both passes see READ because set_mode("read") is called
    once before diffuse() and never toggled inside the loop.
    """
    from types import SimpleNamespace
    from unittest.mock import MagicMock, patch

    GlmImageKVCache, _, _ = kv_cache_classes

    kv_caches = _make_filled_cache(GlmImageKVCache)
    # Mirror forward(): _prepare_condition_image_kv_cache() writes the cache,
    # then forward() sets READ before calling diffuse().
    kv_caches.set_mode("read")

    NUM_STEPS = 2
    LATENT_C, H, W = 4, 8, 8
    latents = torch.randn(BATCH, LATENT_C, H, W)
    prior_token_id = torch.zeros(BATCH, 4, dtype=torch.long)
    prompt_embeds = torch.randn(BATCH, 3, 16)
    neg_prompt_embeds = torch.randn(BATCH, 3, 16)
    target_size = torch.tensor([[H, W]], dtype=torch.float32)
    crop_coords = torch.zeros(BATCH, 2)
    timesteps = torch.linspace(1000, 1, NUM_STEPS)

    observed_modes: list[str] = []

    def fake_transformer(**kwargs):
        kv = kwargs.get("kv_cache")
        observed_modes.append(kv.mode.value if (kv is not None and kv.mode is not None) else "none")
        return (torch.zeros_like(latents),)

    mock_transformer = MagicMock(side_effect=lambda *a, **kw: fake_transformer(**kw))
    mock_transformer.dtype = torch.float32

    mock_scheduler = MagicMock()
    mock_scheduler.step.side_effect = lambda pred, t, lat, return_dict: (lat,)

    # Build minimal pipeline namespace — only attributes diffuse() touches
    pipeline = SimpleNamespace(
        transformer=mock_transformer,
        scheduler=mock_scheduler,
    )

    _cfg_size_path = "vllm_omni.diffusion.models.glm_image.pipeline_glm_image.get_classifier_free_guidance_world_size"
    with patch(_cfg_size_path, return_value=1):
        diffuse_func(
            pipeline,
            latents=latents,
            prior_token_id=prior_token_id,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=neg_prompt_embeds,
            timesteps=timesteps,
            target_size=target_size,
            crop_coords=crop_coords,
            guidance_scale=2.0,
            do_classifier_free_guidance=True,
            kv_caches=kv_caches,
        )

    # Each step: [cond_call, uncond_call] × NUM_STEPS
    assert len(observed_modes) == NUM_STEPS * 2, f"Expected {NUM_STEPS * 2} calls, got {len(observed_modes)}"
    for step in range(NUM_STEPS):
        cond_mode = observed_modes[step * 2]
        uncond_mode = observed_modes[step * 2 + 1]
        assert cond_mode == "read", f"Step {step} conditional pass: expected 'read', got '{cond_mode}'"
        assert uncond_mode == "skip", f"Step {step} unconditional pass: expected 'skip', got '{uncond_mode}'"


def test_diffuse_cfg_parallel_rank_modes(kv_cache_classes, diffuse_func):
    """
    Exercise the CFG-parallel branch of GlmImagePipeline.diffuse().

    Rank 0 is conditional and must read the condition-image cache. Rank 1 is
    unconditional and must not read the cache; either SKIP mode or no cache is
    acceptable for rank 1.
    """
    from types import SimpleNamespace
    from unittest.mock import MagicMock, patch

    GlmImageKVCache, _, _ = kv_cache_classes

    NUM_STEPS = 1
    LATENT_C, H, W = 4, 8, 8
    latents = torch.randn(BATCH, LATENT_C, H, W)
    prior_token_id = torch.zeros(BATCH, 4, dtype=torch.long)
    prompt_embeds = torch.randn(BATCH, 3, 16)
    neg_prompt_embeds = torch.randn(BATCH, 3, 16)
    target_size = torch.tensor([[H, W]], dtype=torch.float32)
    crop_coords = torch.zeros(BATCH, 2)
    timesteps = torch.linspace(1000, 1, NUM_STEPS)

    def run_for_rank(cfg_rank):
        kv_caches = _make_filled_cache(GlmImageKVCache)
        kv_caches.set_mode("read")
        observed_modes = []

        def fake_transformer(**kwargs):
            kv = kwargs.get("kv_cache")
            observed_modes.append(kv.mode.value if (kv is not None and kv.mode is not None) else "none")
            return (torch.zeros_like(latents),)

        mock_transformer = MagicMock(side_effect=lambda *a, **kw: fake_transformer(**kw))
        mock_transformer.dtype = torch.float32

        mock_scheduler = MagicMock()
        mock_scheduler.step.side_effect = lambda pred, t, lat, return_dict: (lat,)

        cfg_group = MagicMock()
        cfg_group.all_gather.return_value = [torch.zeros_like(latents), torch.zeros_like(latents)]

        pipeline = SimpleNamespace(transformer=mock_transformer, scheduler=mock_scheduler)

        with (
            patch(
                "vllm_omni.diffusion.models.glm_image.pipeline_glm_image.get_classifier_free_guidance_world_size",
                return_value=2,
            ),
            patch(
                "vllm_omni.diffusion.models.glm_image.pipeline_glm_image.get_classifier_free_guidance_rank",
                return_value=cfg_rank,
            ),
            patch(
                "vllm_omni.diffusion.models.glm_image.pipeline_glm_image.get_cfg_group",
                return_value=cfg_group,
            ),
        ):
            diffuse_func(
                pipeline,
                latents=latents,
                prior_token_id=prior_token_id,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=neg_prompt_embeds,
                timesteps=timesteps,
                target_size=target_size,
                crop_coords=crop_coords,
                guidance_scale=2.0,
                do_classifier_free_guidance=True,
                kv_caches=kv_caches,
            )

        assert len(observed_modes) == NUM_STEPS
        return observed_modes[0]

    rank0_mode = run_for_rank(0)
    rank1_mode = run_for_rank(1)

    assert rank0_mode == "read", f"CFG rank 0 conditional pass: expected 'read', got '{rank0_mode}'"
    assert rank1_mode in {"skip", "none"}, (
        "CFG rank 1 unconditional pass must not read condition-image cache; "
        f"expected 'skip' or 'none', got '{rank1_mode}'"
    )


def test_diffuse_no_cfg_uses_read_only(kv_cache_classes, diffuse_func):
    """Without CFG, every step has a single cond pass with mode READ."""
    from types import SimpleNamespace
    from unittest.mock import MagicMock, patch

    GlmImageKVCache, _, KVCacheMode = kv_cache_classes

    kv_caches = _make_filled_cache(GlmImageKVCache)
    kv_caches.set_mode("read")

    NUM_STEPS = 2
    LATENT_C, H, W = 4, 8, 8
    latents = torch.randn(BATCH, LATENT_C, H, W)
    prior_token_id = torch.zeros(BATCH, 4, dtype=torch.long)
    prompt_embeds = torch.randn(BATCH, 3, 16)
    target_size = torch.tensor([[H, W]], dtype=torch.float32)
    crop_coords = torch.zeros(BATCH, 2)
    timesteps = torch.linspace(1000, 1, NUM_STEPS)

    observed_modes: list[str] = []

    def fake_transformer(**kwargs):
        kv = kwargs.get("kv_cache")
        observed_modes.append(kv.mode.value if (kv is not None and kv.mode is not None) else "none")
        return (torch.zeros_like(latents),)

    mock_transformer = MagicMock(side_effect=lambda *a, **kw: fake_transformer(**kw))
    mock_transformer.dtype = torch.float32

    mock_scheduler = MagicMock()
    mock_scheduler.step.side_effect = lambda pred, t, lat, return_dict: (lat,)

    pipeline = SimpleNamespace(transformer=mock_transformer, scheduler=mock_scheduler)

    _cfg_size_path = "vllm_omni.diffusion.models.glm_image.pipeline_glm_image.get_classifier_free_guidance_world_size"
    with patch(_cfg_size_path, return_value=1):
        diffuse_func(
            pipeline,
            latents=latents,
            prior_token_id=prior_token_id,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=None,
            timesteps=timesteps,
            target_size=target_size,
            crop_coords=crop_coords,
            guidance_scale=1.0,
            do_classifier_free_guidance=False,
            kv_caches=kv_caches,
        )

    assert len(observed_modes) == NUM_STEPS
    for step, mode in enumerate(observed_modes):
        assert mode == "read", f"Step {step}: expected 'read', got '{mode}'"
