# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.model_executor.models.bagel import PositionEmbedding
from vllm.model_executor.models.interfaces import supports_encoder_cudagraph
from vllm.transformers_utils.configs.bagel import BagelConfig

from vllm_omni.model_executor.models.bagel.bagel import OmniBagelForConditionalGeneration

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class PatchEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Conv2d(3, 8, kernel_size=2, stride=2)

    def forward(self, pixels):
        return self.projection(pixels).flatten(2).transpose(1, 2)


@pytest.fixture
def model():
    # Exercise the real BAGEL eager and protocol methods without loading its
    # language model or VAE. The small tower keeps metadata tests CPU-only.
    model = OmniBagelForConditionalGeneration.__new__(OmniBagelForConditionalGeneration)
    nn.Module.__init__(model)
    model.config = BagelConfig(
        vit_config=dict(image_size=8, patch_size=2, num_channels=3),
        llm_config=dict(hidden_size=12),
        vit_max_num_patch_per_side=5,
    )
    model.vit_model = PatchEncoder()
    model.connector = nn.Linear(8, 12)
    model.vit_pos_embed = PositionEmbedding(5, 12)
    return model.eval()


def test_protocol_and_config(model):
    assert supports_encoder_cudagraph(model)
    config = model.get_encoder_cudagraph_config()
    assert config.modalities == ["image"]
    assert "img2img" not in config.modalities
    assert config.buffer_keys == ["pixel_values"]
    assert config.out_hidden_size == 12


@pytest.mark.parametrize("max_tokens,max_seqs,expected", [(100, 3, (16, 48)), (8, 3, (16, 16))])
def test_budget_range(model, max_tokens, max_seqs, expected):
    config = VllmConfig()
    config.scheduler_config.max_num_batched_tokens = max_tokens
    config.scheduler_config.max_num_seqs = max_seqs
    assert model.get_encoder_cudagraph_budget_range(config) == expected


@pytest.mark.parametrize("extra_dim", [False, True])
def test_item_specs_and_reordered_selection(model, extra_dim):
    pixels = torch.arange(3 * 3 * 8 * 8).reshape(3, 3, 8, 8).float()
    kwargs = {"pixel_values": pixels.unsqueeze(1) if extra_dim else pixels}
    specs = model.get_encoder_cudagraph_item_specs(kwargs)
    assert [(s.input_size, s.output_tokens) for s in specs] == [(16, 16)] * 3
    selected = model.select_encoder_cudagraph_items(kwargs, [2, 0])
    torch.testing.assert_close(selected["pixel_values"], pixels[[2, 0]])
    assert model.select_encoder_cudagraph_items(kwargs, [])["pixel_values"].shape == (0, 3, 8, 8)


def test_incompatible_shape_rejected_before_replay(model):
    with pytest.raises(ValueError, match="BAGEL image encoder expects"):
        model.get_encoder_cudagraph_item_specs({"pixel_values": torch.zeros(1, 3, 4, 16)})


@pytest.mark.parametrize("budget,capacity", [(8, 1), (16, 1), (31, 1), (32, 2), (128, 3)])
def test_capture_capacity(model, budget, capacity):
    inputs = model.prepare_encoder_cudagraph_capture_inputs(budget, 3, 0, torch.device("cpu"), torch.float32)
    assert inputs.values["pixel_values"].shape == (capacity, 3, 8, 8)
    assert inputs.values["pos_embeds"].shape == (1, 16, 12)


@torch.inference_mode()
def test_capture_forward_and_output_order(model):
    pixels = torch.randn(3, 3, 8, 8)
    kwargs = model.select_encoder_cudagraph_items({"pixel_values": pixels}, [2, 0])
    inputs = model.prepare_encoder_cudagraph_capture_inputs(48, 3, 0, torch.device("cpu"), torch.float32)
    replay = model.prepare_encoder_cudagraph_replay_buffers(kwargs, 3, 0)
    inputs.values["pixel_values"][:2].copy_(replay.values["pixel_values"])
    actual = model.encoder_cudagraph_forward(inputs.values)
    expected = model.encoder_eager_forward(kwargs)
    torch.testing.assert_close(actual[:32], expected)
    dest: dict[int, torch.Tensor] = {}
    model.postprocess_encoder_output({"default": actual}, [2, 0], [16] * 3, dest, clone=True)
    torch.testing.assert_close(dest[2], expected[:16])
    torch.testing.assert_close(dest[0], expected[16:])
    actual.zero_()
    torch.testing.assert_close(dest[2], expected[:16])
