# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch
from cache_dit import BlockAdapter
from diffusers import FlowMatchEulerDiscreteScheduler
from torch import nn

from vllm_omni.diffusion.cache import cachedit
from vllm_omni.diffusion.data import DiffusionCacheConfig
from vllm_omni.diffusion.distributed import cfg_parallel, parallel_state
from vllm_omni.diffusion.models.longcat_image.longcat_image_transformer import LongCatImageTransformer2DModel
from vllm_omni.diffusion.models.longcat_image.pipeline_longcat_image_edit import LongCatImageEditPipeline
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


class _Block(nn.Module):
    def forward(self, hidden_states, encoder_hidden_states):
        return encoder_hidden_states, hidden_states + 0.1


class _Transformer(nn.Module):
    # Exercise the real LongCat two-block-list adapter with inexpensive CPU blocks.
    _cache_dit_adapter_config = LongCatImageTransformer2DModel._cache_dit_adapter_config

    def __init__(self):
        super().__init__()
        self.transformer_blocks = nn.ModuleList([_Block() for _ in range(3)])
        self.single_transformer_blocks = nn.ModuleList([_Block() for _ in range(3)])
        self.forward_count = 0

    def forward(self, hidden_states, encoder_hidden_states, **kwargs):
        self.forward_count += 1
        for blocks in (self.transformer_blocks, self.single_transformer_blocks):
            for block in blocks:
                encoder_hidden_states, hidden_states = block(hidden_states, encoder_hidden_states)
        return (hidden_states,)


@pytest.fixture
def pipeline(monkeypatch):
    pipeline = object.__new__(LongCatImageEditPipeline)
    nn.Module.__init__(pipeline)
    pipeline.transformer = _Transformer()
    pipeline.device = torch.device("cpu")
    pipeline.vae_scale_factor = 8
    pipeline.scheduler = FlowMatchEulerDiscreteScheduler()
    monkeypatch.setattr(pipeline, "encode_prompt", lambda **kwargs: (torch.ones(1, 2, 4), torch.zeros(2, 3)))
    monkeypatch.setattr(
        pipeline,
        "prepare_latents",
        lambda *args: (torch.ones(1, 2, 4), torch.ones(1, 2, 4), torch.zeros(2, 3), torch.zeros(2, 3)),
    )
    return pipeline


def _request(guidance, negative_prompt, steps=8):
    return DiffusionRequestBatch(
        requests=[
            OmniDiffusionRequest(
                request_id="longcat-cache-cfg",
                prompt={
                    "prompt": "Edit the image",
                    "negative_prompt": negative_prompt,
                    "additional_information": {
                        "prompt_image": None,
                        "preprocessed_image": torch.ones(1, 3, 16, 16),
                    },
                },
                sampling_params=OmniDiffusionSamplingParams(
                    seed=0,
                    height=16,
                    width=16,
                    guidance_scale=guidance,
                    num_inference_steps=steps,
                    output_type="latent",
                ),
            )
        ]
    )


def _contexts(pipeline):
    return list(pipeline.transformer._context_manager._cached_context_manager.values())


@pytest.mark.parametrize("cfg_world_size", [1, 2])
@pytest.mark.parametrize("cfg_first", [False, True])
def test_edit_cache_counts_steps_across_cfg_requests(pipeline, monkeypatch, request, cfg_world_size, cfg_first):
    # Simulate rank-local CFG routing; no distributed communication is needed here.
    monkeypatch.setattr(parallel_state, "model_parallel_is_initialized", lambda: True)
    monkeypatch.setattr(parallel_state, "get_classifier_free_guidance_world_size", lambda: cfg_world_size)
    monkeypatch.setattr(cfg_parallel, "_get_cfg_world_size_or_one", lambda: cfg_world_size)
    monkeypatch.setattr(cfg_parallel, "get_classifier_free_guidance_rank", lambda: 0)
    monkeypatch.setattr(
        cfg_parallel,
        "get_cfg_group",
        lambda: SimpleNamespace(all_gather=lambda tensor, **kwargs: [tensor, tensor]),
    )
    backend = cachedit.CacheDiTBackend(DiffusionCacheConfig())
    backend.enable(pipeline)
    request.addfinalizer(lambda: backend.disable(pipeline))
    cases = [(1.0, "", 6), (4.5, "", 8), (1.0, "", 10), (4.5, None, 6), (4.5, "", 8), (0.5, "", 6)]
    if cfg_first:
        cases.insert(0, (4.5, "", 8))

    for guidance, negative_prompt, steps in cases:
        backend.refresh(pipeline, steps, verbose=False)
        pipeline.transformer.forward_count = 0
        with torch.no_grad():
            output = pipeline.forward(_request(guidance, negative_prompt, steps))

        separate_cfg = guidance > 1 and negative_prompt is not None and cfg_world_size == 1
        expected_calls = steps * (2 if separate_cfg else 1)
        assert output.output.shape == (1, 2, 4)
        assert torch.isfinite(output.output).all()
        assert pipeline.transformer.forward_count == expected_calls
        assert BlockAdapter.is_cached(pipeline.transformer)
        contexts = _contexts(pipeline)
        assert len(contexts) == 2
        for context in contexts:
            assert context.get_accumulated_executed_steps() == steps
            assert context.get_accumulated_transformer_executed_steps() == expected_calls
            assert context.cache_config.enable_separate_cfg is separate_cfg
            assert context.cache_config.num_inference_steps == steps
            assert context.get_cached_steps(), "No-CFG requests must retain cache acceleration"


def test_edit_cache_cfg_update_preserves_request_cache_settings(pipeline, request):
    backend = cachedit.CacheDiTBackend(
        DiffusionCacheConfig(
            scm_steps_mask_policy="slow",
            force_refresh_step_hint=5,
            force_refresh_step_policy="once",
            max_warmup_steps=2,
            max_continuous_cached_steps=2,
        )
    )
    backend.enable(pipeline)
    request.addfinalizer(lambda: backend.disable(pipeline))
    backend.refresh(pipeline, 8, verbose=False)
    expected = [asdict(context.cache_config) for context in _contexts(pipeline)]
    assert all(config["steps_computation_mask"] for config in expected)
    assert all(config["force_refresh_step_hint"] == 5 for config in expected)

    for do_cfg in (False, True, False):
        cachedit.set_separate_cfg(pipeline.transformer, do_cfg)
        for context, previous in zip(_contexts(pipeline), expected):
            assert asdict(context.cache_config) == {**previous, "enable_separate_cfg": do_cfg}


@pytest.mark.parametrize(
    ("guidance", "negative_prompt", "expected_calls"), [(1.0, "", 8), (4.5, "", 16), (4.5, None, 8)]
)
def test_edit_runs_without_cache_dit(pipeline, guidance, negative_prompt, expected_calls):
    with torch.no_grad():
        output = pipeline.forward(_request(guidance, negative_prompt))
    assert output.output.shape == (1, 2, 4)
    assert torch.isfinite(output.output).all()
    assert pipeline.transformer.forward_count == expected_calls
    assert not BlockAdapter.is_cached(pipeline.transformer)
