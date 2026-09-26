# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tests for FLUX.2-dev quantization routing."""

import pytest
from torch import nn

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


class _RecordingBlock(nn.Module):
    def __init__(self, *, quant_config=None, prefix: str = "", **kwargs):
        super().__init__()
        self.quant_config = quant_config
        self.prefix = prefix


def test_flux2_propagates_online_quant_config_to_all_transformer_blocks(monkeypatch):
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config

    from vllm_omni.diffusion.models.flux2 import flux2_transformer

    monkeypatch.setattr(flux2_transformer, "Flux2TransformerBlock", _RecordingBlock)
    monkeypatch.setattr(flux2_transformer, "Flux2SingleTransformerBlock", _RecordingBlock)

    quant_config = Fp8Config(
        is_checkpoint_fp8_serialized=False,
        activation_scheme="dynamic",
    )
    model = flux2_transformer.Flux2Transformer2DModel(
        in_channels=4,
        out_channels=4,
        num_layers=2,
        num_single_layers=3,
        attention_head_dim=8,
        num_attention_heads=2,
        joint_attention_dim=16,
        timestep_guidance_channels=8,
        axes_dims_rope=(2, 2, 2, 2),
        quant_config=quant_config,
    )

    assert [block.quant_config for block in model.transformer_blocks] == [quant_config, quant_config]
    assert [block.prefix for block in model.transformer_blocks] == [
        "transformer_blocks.0",
        "transformer_blocks.1",
    ]
    assert [block.quant_config for block in model.single_transformer_blocks] == [
        quant_config,
        quant_config,
        quant_config,
    ]
    assert [block.prefix for block in model.single_transformer_blocks] == [
        "single_transformer_blocks.0",
        "single_transformer_blocks.1",
        "single_transformer_blocks.2",
    ]

    assert isinstance(model.x_embedder, nn.Linear)
    assert isinstance(model.context_embedder, nn.Linear)
    assert isinstance(model.proj_out, nn.Linear)
