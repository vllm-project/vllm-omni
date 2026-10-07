# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import vllm_omni.diffusion.models.flux2_klein.pipeline_flux2_klein as pipeline_module
from vllm_omni.diffusion.models.flux2_klein.flux2_klein_transformer import (
    Flux2SingleTransformerBlock,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


class _FakeComponent(nn.Module):
    def __init__(self, *, vae: bool = False):
        super().__init__()
        if vae:
            self.config = SimpleNamespace(block_out_channels=[1, 1, 1, 1], latent_channels=32)


def test_pipeline_passes_od_config_to_transformer(monkeypatch):
    captured_kwargs = {}

    class _FakeTransformer(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            captured_kwargs.update(kwargs)

    components = iter([_FakeComponent(), _FakeComponent(vae=True)])
    monkeypatch.setattr(pipeline_module, "get_local_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(pipeline_module, "prefetch_subfolders", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        pipeline_module,
        "from_pretrained_with_prefetch",
        lambda *args, **kwargs: next(components),
    )
    monkeypatch.setattr(
        pipeline_module.FlowMatchEulerDiscreteScheduler,
        "from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        pipeline_module.Qwen2TokenizerFast,
        "from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(pipeline_module, "get_transformer_config_kwargs", lambda *args, **kwargs: {})
    monkeypatch.setattr(pipeline_module, "Flux2Transformer2DModel", _FakeTransformer)
    monkeypatch.setattr(
        pipeline_module.Flux2KleinPipeline,
        "setup_diffusion_pipeline_profiler",
        lambda *args, **kwargs: None,
    )

    quantization_config = object()
    od_config = SimpleNamespace(
        model="local-model",
        tf_model_config={},
        quantization_config=quantization_config,
        enable_diffusion_pipeline_profiler=False,
    )

    pipeline_module.Flux2KleinPipeline(od_config=od_config)

    assert captured_kwargs["od_config"] is od_config
    assert captured_kwargs["quant_config"] is quantization_config


def test_single_block_passes_text_seq_len_to_attention():
    captured_kwargs = {}

    class _FakeAttention(nn.Module):
        def forward(self, hidden_states, image_rotary_emb=None, **kwargs):
            captured_kwargs.update(kwargs)
            return torch.zeros_like(hidden_states)

    block = object.__new__(Flux2SingleTransformerBlock)
    nn.Module.__init__(block)
    block.norm = nn.Identity()
    block.attn = _FakeAttention()

    hidden_states = torch.randn(1, 7, 4)
    modulation = (
        torch.zeros(1, 1, 4),
        torch.zeros(1, 1, 4),
        torch.zeros(1, 1, 4),
    )

    block(
        hidden_states=hidden_states,
        encoder_hidden_states=None,
        temb_mod_params=modulation,
        text_seq_len=3,
    )

    assert captured_kwargs["text_seq_len"] == 3
