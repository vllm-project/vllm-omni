# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from cache_dit import ForwardPattern
from torch import nn

from vllm_omni.diffusion.cache.cachedit import CacheDiTAdapterConfig, CacheDiTBackend
from vllm_omni.diffusion.cache.teacache import TeaCacheBackend, TeaCacheConfig
from vllm_omni.diffusion.cache.teacache.extractors import extract_hidream_o1_context
from vllm_omni.diffusion.cache.teacache.hook import apply_teacache_hook
from vllm_omni.diffusion.data import DiffusionCacheConfig
from vllm_omni.diffusion.models.hidream_o1_image.hidream_o1_image_transformer import (
    HiDreamO1TextModel,
)
from vllm_omni.diffusion.models.hidream_o1_image.pipeline_hidream_o1_image import (
    HiDreamO1ImagePipeline,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _empty_text_model() -> HiDreamO1TextModel:
    model = HiDreamO1TextModel.__new__(HiDreamO1TextModel)
    nn.Module.__init__(model)
    model.layers = nn.ModuleList([nn.Identity()])
    return model


class _RotaryEmbedding(nn.Module):
    def forward(self, hidden_states, position_ids):
        return hidden_states, hidden_states


class _DecoderLayer(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.input_layernorm = nn.LayerNorm(hidden_size)
        self.calls = 0

    def forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask,
        full_attn_spans,
    ):
        self.calls += 1
        return hidden_states + 0.25


def _tiny_text_model(hidden_size: int = 4) -> HiDreamO1TextModel:
    model = _empty_text_model()
    model.layers = nn.ModuleList([_DecoderLayer(hidden_size)])
    model.norm = nn.LayerNorm(hidden_size)
    model.rotary_emb = _RotaryEmbedding()
    return model


def test_cache_dit_targets_the_decoder_module_that_owns_layers():
    text_model = _empty_text_model()
    pipeline = SimpleNamespace(
        _dit_modules=HiDreamO1ImagePipeline._dit_modules,
        model=SimpleNamespace(
            model=SimpleNamespace(language_model=text_model),
        ),
    )

    with (
        patch("vllm_omni.diffusion.cache.cachedit.backend.BlockAdapter") as block_adapter_cls,
        patch("vllm_omni.diffusion.cache.cachedit.backend.cache_dit") as cache_dit,
    ):
        backend = CacheDiTBackend()
        backend.enable(pipeline)
        backend.refresh(pipeline, num_inference_steps=50)

    adapter_config = HiDreamO1TextModel._cache_dit_adapter_config
    assert isinstance(adapter_config, CacheDiTAdapterConfig)
    assert adapter_config.block_forward_patterns == {"layers": ForwardPattern.Pattern_3}
    assert adapter_config.has_separate_cfg is True

    block_adapter_cls.assert_called_once_with(
        transformer=text_model,
        blocks=[text_model.layers],
        forward_pattern=[ForwardPattern.Pattern_3],
        has_separate_cfg=True,
        check_forward_pattern=True,
    )
    cache_dit.enable_cache.assert_called_once()
    enable_call = cache_dit.enable_cache.call_args
    assert enable_call.args == (block_adapter_cls.return_value,)
    assert enable_call.kwargs["cache_config"].Fn_compute_blocks == 1
    assert enable_call.kwargs["calibrator_config"] is None
    cache_dit.refresh_context.assert_called_once_with(
        text_model,
        num_inference_steps=50,
        verbose=True,
    )


@pytest.mark.parametrize(
    ("configured_threshold", "expected_threshold"),
    [(None, 0.1), (0.05, 0.05)],
)
def test_teacache_backend_targets_the_decoder_module_that_owns_layers(
    configured_threshold: float | None,
    expected_threshold: float,
):
    text_model = _empty_text_model()
    pipeline = HiDreamO1ImagePipeline.__new__(HiDreamO1ImagePipeline)
    nn.Module.__init__(pipeline)
    pipeline.model = SimpleNamespace(
        model=SimpleNamespace(language_model=text_model),
    )

    with patch("vllm_omni.diffusion.cache.teacache.backend.apply_teacache_hook") as apply_hook:
        cache_config = (
            DiffusionCacheConfig()
            if configured_threshold is None
            else DiffusionCacheConfig(rel_l1_thresh=configured_threshold)
        )
        backend = TeaCacheBackend(cache_config)
        backend.enable(pipeline)

    transformer, config = apply_hook.call_args.args
    assert transformer is text_model
    assert config.transformer_type == "HiDreamO1TextModel"
    assert config.rel_l1_thresh == expected_threshold
    assert backend.is_enabled()

    text_model._hook_registry = SimpleNamespace(
        get_hook=lambda _: object(),
        reset_hook=lambda _: None,
    )
    with patch.object(text_model._hook_registry, "reset_hook") as reset_hook:
        backend.refresh(pipeline, num_inference_steps=50)
    reset_hook.assert_called_once_with("teacache")


def test_hidream_o1_teacache_extractor_matches_text_model_forward():
    model = _tiny_text_model()
    inputs_embeds = torch.randn(1, 3, 4)
    position_ids = torch.arange(3).unsqueeze(0)
    full_attn_spans = [[(0, 3)]]

    expected = model(inputs_embeds, position_ids, None, full_attn_spans)
    context = extract_hidream_o1_context(
        model,
        inputs_embeds,
        position_ids,
        None,
        full_attn_spans,
    )
    context.validate()
    (hidden_states,) = context.run_transformer_blocks()
    actual = context.postprocess(hidden_states)

    torch.testing.assert_close(context.modulated_input, inputs_embeds)
    torch.testing.assert_close(actual, expected)


def test_hidream_o1_teacache_keeps_cfg_branch_states_separate():
    model = _tiny_text_model()
    model.do_true_cfg = True
    apply_teacache_hook(
        model,
        TeaCacheConfig(
            transformer_type="HiDreamO1TextModel",
            coefficients=[0.0, 0.0, 0.0, 1.0, 0.0],
            rel_l1_thresh=0.2,
        ),
    )

    position_ids = torch.arange(3).unsqueeze(0)
    full_attn_spans = [[(0, 3)]]
    cond = torch.randn(1, 3, 4)
    uncond = torch.randn(1, 3, 4)

    with patch(
        "vllm_omni.diffusion.cache.teacache.hook.get_classifier_free_guidance_world_size",
        return_value=1,
    ):
        model(cond, position_ids, None, full_attn_spans)
        model(uncond, position_ids, None, full_attn_spans)
        model(cond * 1.001, position_ids, None, full_attn_spans)
        model(uncond * 1.001, position_ids, None, full_attn_spans)

    # Each CFG branch computes its first step and independently reuses that
    # branch's residual at the next sufficiently similar step.
    assert model.layers[0].calls == 2
