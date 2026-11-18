# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Official precision boundaries using CPU tensors and unchanged BF16 weights."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from tests.diffusion.models.lingbot_world import test_lingbot_world_attention as attention_tests
from tests.diffusion.models.lingbot_world.test_lingbot_world_transformer import _tiny_model

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def test_fp32_linear_keeps_canonical_parameters_and_disables_outer_autocast():
    module = attention_tests._load_module()
    layer = nn.Linear(4, 3).bfloat16()
    value = torch.tensor([[1.001, -0.007, 0.313, 0.997]])
    original = {key: x.clone() for key, x in layer.state_dict().items()}
    expected = F.linear(value, layer.weight.float(), layer.bias.float())
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = module._linear_fp32(layer, value)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not torch.equal(actual, actual.bfloat16().float())
    for name, tensor in layer.state_dict().items():
        assert tensor.dtype == torch.bfloat16
        torch.testing.assert_close(tensor, original[name], rtol=0, atol=0)


def test_timestep_mlp_matches_fp32_reference_with_bf16_checkpoint_values():
    module = attention_tests._load_module()
    model = _tiny_model(module, num_layers=1).bfloat16()
    original = {key: x.clone() for key, x in model.state_dict().items()}
    timesteps = torch.tensor([999.0])
    positions = timesteps.double().expand(3)
    phase = positions[:, None] * torch.pow(10000, -torch.arange(2).double() / 2)[None]
    sinusoid = torch.cat([phase.cos(), phase.sin()], -1).float()
    first, _, second = model.time_embedding
    expected = F.linear(
        F.silu(F.linear(sinusoid, first.weight.float(), first.bias.float())), second.weight.float(), second.bias.float()
    ).reshape(1, 3, model.dim)
    projection = model.time_projection[1]
    expected_projection = F.linear(F.silu(expected), projection.weight.float(), projection.bias.float())
    expected_projection = expected_projection.unflatten(-1, (6, model.dim))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual, actual_projection = model._timestep_embeddings(timesteps, batch_size=1, frames=3, dtype=torch.bfloat16)
    assert actual.dtype == actual_projection.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_projection, expected_projection, rtol=0, atol=0)
    for key, tensor in model.state_dict().items():
        assert tensor.dtype == original[key].dtype
        torch.testing.assert_close(tensor, original[key], rtol=0, atol=0)


def test_fp64_rotary_matches_official_complex_rotation_after_model_bfloat16_cast():
    module = attention_tests._load_module()
    model = _tiny_model(module, num_layers=1, attention_head_dim=128).bfloat16()
    cosine, sine = model._rotary_embedding(
        frames=3, height=2, width=3, start_frame=1077, dtype=torch.bfloat16, device=torch.device("cpu")
    )
    assert cosine.dtype == sine.dtype == torch.float64
    value = torch.randn(1, 18, 2, 128, generator=torch.Generator().manual_seed(42)).bfloat16()
    axes = torch.meshgrid(
        torch.arange(1077, 1080).double(), torch.arange(2).double(), torch.arange(3).double(), indexing="ij"
    )
    multipliers = []
    for positions, width in zip(axes, [44, 42, 42], strict=True):
        frequencies = 1 / (10000 ** (torch.arange(0, width, 2).double() / width))
        angle = positions.flatten()[:, None] * frequencies[None]
        multipliers.append(torch.polar(torch.ones_like(angle), angle))
    multiplier = torch.cat(multipliers, -1)[None, :, None]
    expected = torch.view_as_real(torch.view_as_complex(value.double().unflatten(-1, (-1, 2))) * multiplier)
    expected = expected.flatten(-2).bfloat16()
    actual = model.blocks[0].self_attn.rotary_embedding(value, cosine, sine)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("input_dtype", [torch.bfloat16, torch.float32])
def test_block_keeps_non_bfloat16_representable_residual(input_dtype):
    module = attention_tests._load_module()
    model = _tiny_model(module, num_layers=1).bfloat16()
    block = model.blocks[0]
    dim = model.dim

    class SelfAttention(nn.Module):
        ulysses_world_size = 1
        norm_q = SimpleNamespace(weight=torch.ones(1, dtype=torch.bfloat16))

        def forward(self, value, **kwargs):
            assert value.dtype == torch.bfloat16
            return torch.ones_like(value)

    class CrossAttention(nn.Module):
        def forward(self, value, context, **kwargs):
            return torch.zeros_like(value, dtype=torch.bfloat16), None

    class FeedForward(nn.Module):
        def forward(self, value):
            assert value.dtype == torch.bfloat16
            return torch.zeros_like(value)

    block.self_attn = SelfAttention()
    block.cross_attn = CrossAttention()
    block.ffn = FeedForward()
    block.norm3 = nn.Identity()
    block.camera_modulation = lambda value: (torch.zeros_like(value), torch.zeros_like(value))
    with torch.no_grad():
        block.modulation.zero_()
    hidden = torch.ones(1, 4, dim, dtype=input_dtype)
    norm_dtypes = []
    block.norm1.register_forward_hook(lambda layer, inputs, output: norm_dtypes.append(output.dtype))
    projection = torch.zeros(1, 1, 6, dim)
    projection[:, :, 2] = 0.001
    actual, _ = block(
        hidden,
        None,
        projection,
        torch.zeros_like(hidden),
        self_cache=None,
        cross_cache=None,
        current_start=0,
        sink_tokens=0,
        update_cache=False,
        rotary_emb=(torch.ones(4, 1), torch.zeros(4, 1)),
    )
    assert norm_dtypes == [input_dtype]
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, torch.full_like(actual, 1.001), rtol=0, atol=0)
    assert not torch.equal(actual, actual.bfloat16().float())


@pytest.mark.parametrize("input_dtype", [torch.bfloat16, torch.float32])
def test_head_keeps_fp32_projection_without_changing_bf16_parameters(input_dtype):
    module = attention_tests._load_module()
    head = module._LingBotHead(dim=4, out_channels=2, patch_size=(1, 1, 1), eps=1e-6).bfloat16()
    hidden = torch.tensor([[[1.001, 1.007, -0.333, 0.71], [0.1, 0.2, 0.3, 0.5]]], dtype=input_dtype)
    embedding = torch.tensor([[[0.123, -0.221, 0.357, 0.817]]])
    shift, scale = (head.modulation.unsqueeze(1).float() + embedding.unsqueeze(2)).chunk(2, dim=2)
    normalized = F.layer_norm(hidden, (4,), eps=1e-6).unflatten(1, (1, 2))
    normalized = (normalized * (1 + scale) + shift).flatten(1, 2)
    expected = F.linear(normalized, head.head.weight.float(), head.head.bias.float())
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = head(hidden, embedding)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert all(parameter.dtype == torch.bfloat16 for parameter in head.parameters())


def test_postload_promotion_preserves_native_stage_rounding_including_partial_reload():
    module = attention_tests._load_module()
    model = _tiny_model(module, num_layers=1).bfloat16()
    names = [
        name
        for name, _ in model.named_parameters()
        if name.startswith(("time_embedding.", "time_projection.", "head.head."))
    ]
    parameters = dict(model.named_parameters())
    value = torch.tensor(1.001).bfloat16().float().item()
    model.load_weights((name, torch.full(parameters[name].shape, 1.001)) for name in names)
    parameters = dict(model.named_parameters())
    for name in names:
        assert parameters[name].dtype == torch.float32
        assert torch.all(parameters[name] == value)
    assert model.patch_embedding.weight.dtype == torch.bfloat16
    pointer = model.time_projection[1].weight.data_ptr()
    assert model.time_projection[1].weight.float().data_ptr() == pointer
    name = "time_projection.1.weight"
    model.load_weights([(name, torch.full(parameters[name].shape, 1.007))])
    assert torch.all(model.time_projection[1].weight == torch.tensor(1.007).bfloat16().float())
