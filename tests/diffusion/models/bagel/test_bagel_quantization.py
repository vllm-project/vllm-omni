# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch
from torch.nn import Module
from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.linear import LinearBase

import vllm_omni.quantization.int8_config as int8_config
from vllm_omni.diffusion.layers.mot.mot_qkv_parallel_linear import (
    MoTQKVParallelLinear,
)
from vllm_omni.diffusion.layers.mot.mot_row_parallel_linear import (
    MoTRowParallelLinear,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _cpu_vllm_config() -> VllmConfig:
    return VllmConfig(device_config=DeviceConfig(device="cpu"))


def _patch_online_int8(mocker, tp_size: int) -> int8_config.DiffusionInt8Config:
    mocker.patch("vllm.model_executor.layers.linear.get_tensor_model_parallel_rank", return_value=0)
    mocker.patch("vllm.model_executor.layers.linear.get_tensor_model_parallel_world_size", return_value=tp_size)
    mocker.patch("vllm.model_executor.parameter.get_tensor_model_parallel_rank", return_value=0)
    mocker.patch("vllm.model_executor.parameter.get_tensor_model_parallel_world_size", return_value=tp_size)
    mocker.patch.object(int8_config.current_omni_platform, "is_cuda", return_value=True)
    mocker.patch.object(int8_config.current_omni_platform, "is_npu", return_value=False)

    kernel = mocker.Mock()
    kernel.layer_param_names = (
        "weight",
        "weight_scale",
        "input_scale",
        "input_zero_point",
        "azp_adj",
    )
    mocker.patch.object(int8_config, "init_int8_linear_kernel", return_value=kernel)
    return int8_config.DiffusionInt8Config()


def test_bagel_row_secondary_expert_uses_shared_full_row_scale(mocker):
    quant_config = _patch_online_int8(mocker, tp_size=2)
    native_quant = mocker.patch.object(int8_config.ops, "scaled_int8_quant")
    tp_group = mocker.sentinel.tp_group
    mocker.patch.object(
        int8_config,
        "get_tp_group",
        return_value=SimpleNamespace(device_group=tp_group),
    )

    full_weight = torch.tensor(
        [[1, -2, 8, -64], [2, -3, 7, -56]],
        dtype=torch.bfloat16,
    )
    full_amax = full_weight.abs().amax(dim=1, keepdim=True).float()

    def max_reduce(row_amax, op, group):
        assert op is torch.distributed.ReduceOp.MAX
        assert group is tp_group
        row_amax.copy_(full_amax)

    all_reduce = mocker.patch("torch.distributed.all_reduce", side_effect=max_reduce)

    with set_current_vllm_config(_cpu_vllm_config()):
        layer = MoTRowParallelLinear(
            input_size=4,
            output_size=2,
            bias=False,
            vae_bias=False,
            quant_config=quant_config,
            prefix="bagel.o_proj",
        )

    assert type(layer.gen_exp) is Module
    assert not isinstance(layer.gen_exp, LinearBase)
    assert layer.input_size == layer.gen_exp.input_size == 4
    assert layer.input_size_per_partition == layer.gen_exp.input_size_per_partition == 2

    layer.weight.weight_loader(layer.weight, full_weight)
    layer.gen_exp.weight.weight_loader(layer.gen_exp.weight, full_weight)

    inv_scale = torch.iinfo(torch.int8).max / full_amax
    expected = full_weight.float().mul(inv_scale).round().clamp(-127, 127).to(torch.int8)
    assert all_reduce.call_count == 2
    native_quant.assert_not_called()
    assert torch.equal(layer.weight.t(), expected[:, :2])
    assert torch.equal(layer.gen_exp.weight.t(), expected[:, :2])
    expected_scale = full_amax / torch.iinfo(torch.int8).max
    assert torch.equal(layer.weight_scale, expected_scale)
    assert torch.equal(layer.gen_exp.weight_scale, expected_scale)


def test_bagel_qkv_secondary_expert_uses_native_quantization_without_reduce(mocker):
    quant_config = _patch_online_int8(mocker, tp_size=2)
    captured_weight = None
    expected_qweight = torch.arange(24, dtype=torch.int8).reshape(6, 4)
    expected_scale = torch.arange(1, 7, dtype=torch.float32).reshape(6, 1)

    def native_quant(weight, scale):
        nonlocal captured_weight
        assert scale is None
        captured_weight = weight.detach().clone()
        return expected_qweight, expected_scale, None

    scaled_int8_quant = mocker.patch.object(
        int8_config.ops,
        "scaled_int8_quant",
        side_effect=native_quant,
    )
    get_tp_group = mocker.patch.object(int8_config, "get_tp_group")
    all_reduce = mocker.patch("torch.distributed.all_reduce")

    with set_current_vllm_config(_cpu_vllm_config()):
        layer = MoTQKVParallelLinear(
            hidden_size=4,
            head_size=2,
            total_num_heads=2,
            total_num_kv_heads=2,
            bias=False,
            vae_bias=False,
            quant_config=quant_config,
            prefix="bagel.qkv_proj",
        )

    assert type(layer.gen_exp) is Module
    assert not isinstance(layer.gen_exp, LinearBase)
    assert layer.gen_exp.input_size == 4
    assert layer.gen_exp.input_size_per_partition == 4

    full_weight = torch.arange(48, dtype=torch.bfloat16).reshape(12, 4)
    layer.gen_exp.weight.weight_loader(layer.gen_exp.weight, full_weight)

    expected_local = torch.cat((full_weight[0:2], full_weight[4:6], full_weight[8:10]))
    assert captured_weight is not None
    assert torch.equal(captured_weight, expected_local)
    scaled_int8_quant.assert_called_once()
    get_tp_group.assert_not_called()
    all_reduce.assert_not_called()
    assert torch.equal(layer.gen_exp.weight, expected_qweight.t())
    assert torch.equal(layer.gen_exp.weight_scale, expected_scale)


def test_bagel_tp1_row_secondary_expert_keeps_full_input_dimension(mocker):
    quant_config = _patch_online_int8(mocker, tp_size=1)

    with set_current_vllm_config(_cpu_vllm_config()):
        layer = MoTRowParallelLinear(
            input_size=4,
            output_size=2,
            bias=False,
            vae_bias=False,
            quant_config=quant_config,
            prefix="bagel.o_proj",
        )

    assert layer.gen_exp.input_size == 4
    assert layer.gen_exp.input_size_per_partition == 4
