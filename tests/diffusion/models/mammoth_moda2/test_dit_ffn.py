# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Focused regression tests for MammothModa2 DiT quantizable linears."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from vllm_omni.diffusion.models.mammoth_moda2.mammothmoda2_dit_model import LuminaFeedForward
from vllm_omni.diffusion.models.mammoth_moda2.pipeline_mammothmoda2_dit import MammothModa2DiTPipeline

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.fixture(autouse=True)
def _mock_tp1(monkeypatch):
    monkeypatch.setattr("vllm.model_executor.layers.linear.get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr("vllm.model_executor.layers.linear.get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr("vllm.model_executor.parameter.get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr("vllm.model_executor.parameter.get_tensor_model_parallel_world_size", lambda: 1)


def test_merged_ffn_matches_original_swiglu():
    torch.manual_seed(0)
    dim, inner_dim = 8, 16
    ffn = LuminaFeedForward(
        dim=dim,
        inner_dim=inner_dim,
        multiple_of=8,
        ffn_dim_multiplier=1.0,
    ).eval()

    linear_1 = torch.randn(inner_dim, dim)
    linear_3 = torch.randn(inner_dim, dim)
    linear_2 = torch.randn(dim, inner_dim)
    ffn.gate_up_proj.weight.weight_loader(ffn.gate_up_proj.weight, linear_1, 0)
    ffn.gate_up_proj.weight.weight_loader(ffn.gate_up_proj.weight, linear_3, 1)
    ffn.linear_2.weight.weight_loader(ffn.linear_2.weight, linear_2)

    x = torch.randn(2, 5, dim)
    expected = F.linear(
        F.silu(F.linear(x, linear_1).float()).to(x.dtype) * F.linear(x, linear_3),
        linear_2,
    )

    with torch.no_grad():
        actual = ffn(x)

    assert isinstance(actual, torch.Tensor)
    torch.testing.assert_close(actual, expected)


class _WeightTarget(nn.Module):
    def __init__(self, loader):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(1))
        self.weight.weight_loader = loader


def _pipeline_with_weight_targets(stacked_loader, regular_loader):
    pipeline = object.__new__(MammothModa2DiTPipeline)
    nn.Module.__init__(pipeline)

    transformer_block = nn.Module()
    transformer_block.feed_forward = nn.Module()
    transformer_block.feed_forward.gate_up_proj = _WeightTarget(stacked_loader)
    transformer_block.feed_forward.linear_2 = _WeightTarget(regular_loader)
    pipeline.gen_transformer = nn.Module()
    pipeline.gen_transformer.layers = nn.ModuleList([transformer_block])

    qformer_layer = nn.Module()
    qformer_layer.ffn = nn.Module()
    qformer_layer.ffn.gate_up_proj = _WeightTarget(stacked_loader)
    pipeline.gen_image_condition_refiner = nn.Module()
    pipeline.gen_image_condition_refiner.layers = nn.ModuleList([qformer_layer])
    return pipeline


def test_load_weights_merges_transformer_and_qformer_ffns():
    stacked_calls = []
    regular_calls = []

    def stacked_loader(param, weight, shard_id):
        del param
        stacked_calls.append((shard_id, weight.clone()))

    def regular_loader(param, weight):
        del param
        regular_calls.append(weight.clone())

    pipeline = _pipeline_with_weight_targets(stacked_loader, regular_loader)
    weights = {
        "gen_transformer.layers.0.feed_forward.linear_1.weight": torch.tensor([1.0]),
        "gen_transformer.layers.0.feed_forward.linear_3.weight": torch.tensor([3.0]),
        "gen_transformer.layers.0.feed_forward.linear_2.weight": torch.tensor([2.0]),
        "gen_image_condition_refiner.layers.0.ffn.linear_1.weight": torch.tensor([4.0]),
        "gen_image_condition_refiner.layers.0.ffn.linear_3.weight": torch.tensor([5.0]),
        "llm_model.ignored.weight": torch.tensor([6.0]),
        "gen_tokenizer.ignored.weight": torch.tensor([7.0]),
    }

    loaded = pipeline.load_weights(weights.items())

    assert [(shard, weight.item()) for shard, weight in stacked_calls] == [
        (0, 1.0),
        (1, 3.0),
        (0, 4.0),
        (1, 5.0),
    ]
    assert [weight.item() for weight in regular_calls] == [2.0]
    assert loaded == {
        "gen_transformer.layers.0.feed_forward.gate_up_proj.weight",
        "gen_transformer.layers.0.feed_forward.linear_2.weight",
        "gen_image_condition_refiner.layers.0.ffn.gate_up_proj.weight",
    }
