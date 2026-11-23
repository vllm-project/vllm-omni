# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tests for the shared Pi-family inference dtype policy."""

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.models.pi.common import inference_dtype

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _TinyInner(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(2, 2)
        self.vision_tower = nn.Linear(2, 2)
        self.multi_modal_projector = nn.Linear(2, 2)
        self.input_layernorm = nn.LayerNorm(2)
        self.post_attention_layernorm = nn.LayerNorm(2)
        self.model = nn.Module()
        self.model.norm = nn.LayerNorm(2)
        self.register_buffer("floating_buffer", torch.ones(1))
        self.register_buffer("embed_scale", torch.ones(1))


class _TinyPi(nn.Module):
    def __init__(self):
        super().__init__()
        self.paligemma_with_expert = _TinyInner()
        self.state_proj = nn.Linear(2, 2)
        self.action_in_proj = nn.Linear(2, 2)
        self.action_out_proj = nn.Linear(2, 2)
        self.time_mlp = nn.Linear(2, 2)
        self.register_buffer("outer_floating_buffer", torch.ones(1))


def test_float32_policy_casts_every_floating_tensor_to_float32():
    model = _TinyPi().to(dtype=torch.bfloat16)

    inference_dtype.apply_pi_inference_dtype(model, torch.float32)

    assert {parameter.dtype for parameter in model.parameters()} == {torch.float32}
    assert {buffer.dtype for buffer in model.buffers() if buffer.is_floating_point()} == {torch.float32}


def test_bfloat16_policy_matches_lerobot_mixed_layout():
    model = _TinyPi().to(dtype=torch.bfloat16)

    inference_dtype.apply_pi_inference_dtype(model, torch.bfloat16)

    inner = model.paligemma_with_expert
    assert inner.backbone.weight.dtype is torch.bfloat16
    assert inner.floating_buffer.dtype is torch.bfloat16
    assert inner.embed_scale.dtype is torch.bfloat16

    for module in (
        inner.vision_tower,
        inner.multi_modal_projector,
        inner.input_layernorm,
        inner.post_attention_layernorm,
        inner.model.norm,
        model.state_proj,
        model.action_in_proj,
        model.action_out_proj,
        model.time_mlp,
    ):
        assert module.weight.dtype is torch.float32
    assert model.outer_floating_buffer.dtype is torch.float32


def test_policy_rejects_unvalidated_dtype():
    with pytest.raises(ValueError, match="Unsupported Pi-family inference dtype"):
        inference_dtype.apply_pi_inference_dtype(_TinyPi(), torch.float16)


def test_match_module_input_dtype_casts_only_when_needed():
    module = nn.Linear(2, 2).to(dtype=torch.bfloat16)
    source = torch.ones(1, 2, dtype=torch.float32)

    aligned = inference_dtype.match_module_input_dtype(source, module)

    assert aligned.dtype is torch.bfloat16
    assert inference_dtype.match_module_input_dtype(aligned, module) is aligned
