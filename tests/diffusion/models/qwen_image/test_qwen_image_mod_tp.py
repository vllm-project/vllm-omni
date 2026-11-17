# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Qwen-Image AdaLN modulation projections are column-sharded under TP."""

import pytest
import torch
import torch.nn.functional as F
from pytest_mock import MockerFixture
from vllm.config.lora import LoRAConfig
from vllm.model_executor.layers.linear import ColumnParallelLinear

from vllm_omni.diffusion.lora.layers.column_parallel_linear import DiffusionColumnParallelLinearWithLoRA
from vllm_omni.diffusion.lora.utils import from_layer_diffusion
from vllm_omni.diffusion.models.qwen_image import qwen_image_transformer
from vllm_omni.diffusion.models.qwen_image.qwen_image_transformer import _modulation_linear

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

TP_SIZE = 2
DIM = 16
LORA_RANK = 8
LINEAR_GATHER = "vllm.model_executor.layers.linear.tensor_model_parallel_all_gather"
LORA_GATHER = "vllm.lora.layers.column_parallel_linear.tensor_model_parallel_all_gather"


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


def _sharded_forward(
    mocker: MockerFixture, layers: list[torch.nn.Module], x: torch.Tensor, gather_path: str
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Run every rank's real forward and simulate the TP all-gather.

    Pass 1 records each rank's actual local (pre-gather) output.  Pass 2 runs
    rank 0 again and gathers its live local output with the other ranks'
    recorded outputs, so a wrong local result on any rank changes the result.
    """
    local: list[torch.Tensor] = []

    def capture(output: torch.Tensor) -> torch.Tensor:
        local.append(output.clone())
        return output

    mocker.patch(gather_path, side_effect=capture)
    for layer in layers:
        layer(x)
    assert len(local) == len(layers)

    mocker.patch(gather_path, side_effect=lambda output: torch.cat([output, *local[1:]], dim=-1))
    return layers[0](x), local


@pytest.fixture
def full_weights():
    torch.manual_seed(0)
    return torch.randn(6 * DIM, DIM), torch.randn(6 * DIM), torch.randn(3, DIM)


def test_modulation_linear_shards_and_gathers(mocker: MockerFixture, full_weights):
    weight, bias, temb = full_weights
    ranks = [_build_rank(mocker, rank, weight, bias) for rank in range(TP_SIZE)]
    shard = 6 * DIM // TP_SIZE
    for rank, layer in enumerate(ranks):
        assert isinstance(layer, ColumnParallelLinear)
        assert layer.gather_output
        assert layer.weight.shape == (shard, DIM)
        torch.testing.assert_close(layer.weight.data, weight[rank * shard : (rank + 1) * shard])
        torch.testing.assert_close(layer.bias.data, bias[rank * shard : (rank + 1) * shard])

    output, local = _sharded_forward(mocker, ranks, temb, LINEAR_GATHER)
    expected = F.linear(temb, weight, bias)
    for rank, local_output in enumerate(local):
        torch.testing.assert_close(local_output, expected[:, rank * shard : (rank + 1) * shard])
    torch.testing.assert_close(output, expected)


def test_modulation_lora_matches_unsharded(mocker: MockerFixture, full_weights):
    weight, bias, temb = full_weights
    lora_a = torch.randn(LORA_RANK, DIM)
    lora_b = torch.randn(6 * DIM, LORA_RANK)
    lora_config = LoRAConfig(max_lora_rank=LORA_RANK, max_loras=1, lora_dtype=torch.float32)

    ranks = []
    for rank in range(TP_SIZE):
        base = _build_rank(mocker, rank, weight, bias)
        layer = from_layer_diffusion(base, max_loras=1, lora_config=lora_config, packed_modules_list=[])
        assert isinstance(layer, DiffusionColumnParallelLinearWithLoRA)
        layer.set_lora(0, lora_a, lora_b)
        ranks.append(layer)

    output, _ = _sharded_forward(mocker, ranks, temb, LORA_GATHER)
    expected = F.linear(temb, weight, bias) + (temb @ lora_a.t()) @ lora_b.t()
    torch.testing.assert_close(output, expected)


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
