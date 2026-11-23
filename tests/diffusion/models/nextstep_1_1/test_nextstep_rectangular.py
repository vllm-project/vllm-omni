# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from transformers import LlamaConfig, StaticCache

from vllm_omni.diffusion.models.nextstep_1_1 import pipeline_nextstep_1_1 as pipeline_module
from vllm_omni.diffusion.models.nextstep_1_1.modeling_nextstep import NextStepModel
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.mark.parametrize("capacity", [5, 8])
def test_static_cache_mask_preserves_capacity_causality_and_padding(capacity):
    config = LlamaConfig(hidden_size=16, num_attention_heads=2, num_key_value_heads=2, num_hidden_layers=1)
    cache = StaticCache(config=config, max_cache_len=capacity)
    model = NextStepModel.__new__(NextStepModel)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(_attn_implementation="sdpa")
    inputs = torch.zeros(1, 2, 16)
    mask = model._update_causal_mask(
        attention_mask=torch.tensor([[1, 0, 1, 1]]),
        input_tensor=inputs,
        cache_position=torch.tensor([2, 3]),
        past_key_values=cache,
        output_attentions=False,
    )
    assert mask.shape == (1, 1, 2, capacity)
    allowed = torch.zeros(2, capacity, dtype=torch.bool)
    allowed[0, [0, 2]] = True
    allowed[1, [0, 2, 3]] = True
    torch.testing.assert_close(mask[0, 0] == 0, allowed)
    assert (mask[0, 0][~allowed] == torch.finfo(inputs.dtype).min).all()


@pytest.mark.parametrize("height,width", [(32, 32), (32, 48), (48, 32)])
@pytest.mark.parametrize("batch", [1, 2])
def test_forward_preserves_latent_grid(monkeypatch, height, width, batch):
    pipeline = object.__new__(pipeline_module.NextStep11Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline._NextStep11Pipeline__device = torch.device("cpu")
    pipeline.config = SimpleNamespace(use_gen_pos_embed=False)
    pipeline.down_factor = 16
    pipeline.scaling_factor = 2.0
    pipeline.shift_factor = 0.25
    grid = SimpleNamespace(config=SimpleNamespace(latent_channels=2, latent_patch_size=2))
    expected = torch.arange(batch * 2 * (height // 8) * (width // 8), dtype=torch.float32)
    expected = expected.reshape(batch, 2, height // 8, width // 8)
    tokens = NextStepModel.patchify(grid, expected)
    pipeline.model = Mock(
        return_value=SimpleNamespace(last_hidden_state=torch.zeros(batch, 1, 4), past_key_values=None)
    )
    pipeline.model.forward_model.return_value = pipeline.model.return_value
    pipeline.model.unpatchify = MethodType(NextStepModel.unpatchify, grid)
    pipeline.tokenizer = Mock(
        bos_token=None,
        return_value=SimpleNamespace(
            input_ids=torch.ones(batch, 2, dtype=torch.long), attention_mask=torch.ones(batch, 2, dtype=torch.long)
        ),
    )
    pipeline._check_input = Mock(return_value=(["teapot"], None))
    pipeline._build_captions = Mock(return_value=(["teapot"] * batch, None, 1, 1.0))
    pipeline._add_prefix_ids = lambda _hw, ids, mask: (ids, mask)
    pipeline.decoding = Mock(return_value=tokens)
    pipeline.vae = SimpleNamespace(dtype=torch.float32, decode=lambda latent: SimpleNamespace(sample=latent))
    monkeypatch.setattr(pipeline_module, "StaticCache", lambda **_kwargs: None)
    params = OmniDiffusionSamplingParams(
        height=height, width=width, guidance_scale=1.0, num_outputs_per_prompt=batch, seed=None
    )
    output = pipeline.forward(SimpleNamespace(prompts=["teapot"], sampling_params=params))
    torch.testing.assert_close(output.output, expected / 2 + 0.25, rtol=0, atol=0)
