# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Qwen-Image AdaLN modulation projections are column-sharded under TP."""

import pytest
import torch
import torch.nn.functional as F
from pytest_mock import MockerFixture
from vllm.model_executor.layers.linear import ColumnParallelLinear

from vllm_omni.diffusion.models.qwen_image import qwen_image_transformer
from vllm_omni.diffusion.models.qwen_image.qwen_image_transformer import _modulation_linear

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

TP_SIZE = 2
DIM = 16


def _patch_tp(mocker: MockerFixture, rank: int) -> None:
    for module in ("vllm.model_executor.layers.linear", "vllm.model_executor.parameter"):
        mocker.patch(f"{module}.get_tensor_model_parallel_world_size", return_value=TP_SIZE)
        mocker.patch(f"{module}.get_tensor_model_parallel_rank", return_value=rank)


def _build_rank(mocker: MockerFixture, rank: int, weight: torch.Tensor, bias: torch.Tensor) -> ColumnParallelLinear:
    _patch_tp(mocker, rank)
    layer = _modulation_linear(DIM, quant_config=None, prefix="transformer_blocks.0.img_mod.1")
    layer.weight.weight_loader(layer.weight, weight)
    layer.bias.weight_loader(layer.bias, bias)
    return layer


def test_modulation_linear_shards_and_gathers(mocker: MockerFixture):
    torch.manual_seed(0)
    weight = torch.randn(6 * DIM, DIM)
    bias = torch.randn(6 * DIM)
    temb = torch.randn(3, DIM)

    ranks = [_build_rank(mocker, rank, weight, bias) for rank in range(TP_SIZE)]
    shard = 6 * DIM // TP_SIZE
    for rank, layer in enumerate(ranks):
        assert isinstance(layer, ColumnParallelLinear)
        assert layer.gather_output
        assert layer.weight.shape == (shard, DIM)
        torch.testing.assert_close(layer.weight.data, weight[rank * shard : (rank + 1) * shard])
        torch.testing.assert_close(layer.bias.data, bias[rank * shard : (rank + 1) * shard])

    # Stand in for the TP all-gather: concatenate every rank's local output.
    local = [F.linear(temb, layer.weight, layer.bias) for layer in ranks]
    mocker.patch(
        "vllm.model_executor.layers.linear.tensor_model_parallel_all_gather",
        side_effect=lambda _output: torch.cat(local, dim=-1),
    )
    torch.testing.assert_close(ranks[0](temb), F.linear(temb, weight, bias))


def test_quantized_modulation_stays_replicated(mocker: MockerFixture):
    replicated = mocker.patch.object(qwen_image_transformer, "ReplicatedLinear")
    quant_config = mocker.sentinel.quant_config
    layer = _modulation_linear(DIM, quant_config=quant_config, prefix="transformer_blocks.0.txt_mod.1")
    assert layer is replicated.return_value
    replicated.assert_called_once_with(
        DIM,
        6 * DIM,
        bias=True,
        return_bias=False,
        quant_config=quant_config,
        prefix="transformer_blocks.0.txt_mod.1",
    )
