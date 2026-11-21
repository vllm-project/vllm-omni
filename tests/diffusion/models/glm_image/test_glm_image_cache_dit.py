# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace
from typing import cast

import pytest
import torch
from cache_dit import BlockAdapter
from torch import nn

from vllm_omni.diffusion.cache.cachedit import CacheDiTBackend
from vllm_omni.diffusion.data import DiffusionCacheConfig
from vllm_omni.diffusion.models.glm_image.glm_image_transformer import GlmImageTransformer2DModel
from vllm_omni.diffusion.models.glm_image.pipeline_glm_image import GlmImagePipeline

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_diffuse_cache_dit_counts_steps_for_each_request_cfg_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("vllm_omni.diffusion.cache.cachedit.backend._cfg_parallel_world_size", lambda: 1)
    monkeypatch.setattr(
        "vllm_omni.diffusion.models.glm_image.pipeline_glm_image.get_classifier_free_guidance_world_size",
        lambda: 1,
    )

    class TinyBlock(nn.Module):
        def forward(self, hidden_states, encoder_hidden_states, **kwargs):
            return hidden_states + 0.1, encoder_hidden_states + 0.1

    class TinyTransformer(nn.Module):
        # Use GLM-Image's real Cache-DiT adapter configuration with tiny blocks.
        _cache_dit_adapter_config = GlmImageTransformer2DModel._cache_dit_adapter_config
        dtype = torch.float32

        def __init__(self):
            super().__init__()
            self.transformer_blocks = nn.ModuleList([TinyBlock() for _ in range(3)])
            self.forward_count = 0

        def forward(self, *, hidden_states, encoder_hidden_states, **kwargs):
            self.forward_count += 1
            for block in self.transformer_blocks:
                hidden_states, encoder_hidden_states = block(hidden_states, encoder_hidden_states, **kwargs)
            return (hidden_states,)

    class TinyScheduler:
        @staticmethod
        def step(noise, timestep, latents, return_dict=False):
            return (latents,)

    transformer = TinyTransformer()
    pipeline = cast(GlmImagePipeline, SimpleNamespace(transformer=transformer, scheduler=TinyScheduler()))
    backend = CacheDiTBackend(DiffusionCacheConfig())
    backend.enable(pipeline)
    try:
        assert BlockAdapter.is_cached(transformer)
        num_steps = 8

        # Reuse one installed cache across requests to catch stale CFG mode.
        for do_cfg in (True, False, True):
            backend.refresh(pipeline, num_steps, verbose=False)
            transformer.forward_count = 0

            GlmImagePipeline.diffuse(
                pipeline,
                latents=torch.zeros(1, 1, 2, 2),
                prior_token_id=torch.zeros(1, 1, dtype=torch.long),
                prompt_embeds=torch.zeros(1, 1, 2),
                negative_prompt_embeds=torch.zeros(1, 1, 2) if do_cfg else None,
                timesteps=torch.linspace(1.0, 0.0, num_steps),
                target_size=torch.tensor([[16.0, 16.0]]),
                crop_coords=torch.zeros(1, 2),
                guidance_scale=4.0 if do_cfg else 1.0,
                do_classifier_free_guidance=do_cfg,
            )

            expected_forwards_per_step = 2 if do_cfg else 1
            context = transformer._context_manager.get_context(transformer._context_names[0])
            assert transformer.forward_count == num_steps * expected_forwards_per_step
            assert context.cache_config.enable_separate_cfg is do_cfg
            assert context.cache_config.num_inference_steps == num_steps
            assert context.executed_steps == num_steps
            assert context.transformer_executed_steps == transformer.forward_count
            assert context.get_cached_steps(), "Repeated denoising inputs should retain Cache-DiT acceleration"
    finally:
        backend.disable(pipeline)
