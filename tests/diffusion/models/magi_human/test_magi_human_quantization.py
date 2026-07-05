# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for MagiHuman quantization config propagation."""

import torch.nn as nn
from pytest_mock import MockerFixture

import vllm_omni.diffusion.models.magi_human.magi_human_dit as magi_human_dit
from vllm_omni.diffusion.models.magi_human.magi_human_dit import (
    DiTModel,
    MagiHumanDiTConfig,
    MoEColumnParallelLinear,
    MoEQKVParallelLinear,
    MoERowParallelLinear,
)


class _FakeParallelLinear(nn.Module):
    def __init__(self, *args, quant_config=None, **kwargs):
        super().__init__()
        self.quant_config = quant_config
        self.num_heads = kwargs.get("total_num_heads", 1)
        self.num_kv_heads = kwargs.get("total_num_kv_heads", 1)


def _tiny_magi_human_config() -> MagiHumanDiTConfig:
    return MagiHumanDiTConfig(
        num_layers=2,
        hidden_size=128,
        head_dim=16,
        num_query_groups=2,
        video_in_channels=16,
        audio_in_channels=8,
        text_in_channels=32,
        mm_layers=[0],
        gelu7_layers=[0],
        enable_attn_gating=True,
    )


class TestMagiHumanDiTQuantization:
    def _patch_parallel_linears(self, mocker: MockerFixture):
        mocker.patch.object(magi_human_dit, "get_tensor_model_parallel_world_size", return_value=1)
        mocker.patch.object(magi_human_dit, "QKVParallelLinear", _FakeParallelLinear)
        mocker.patch.object(magi_human_dit, "ColumnParallelLinear", _FakeParallelLinear)
        mocker.patch.object(magi_human_dit, "RowParallelLinear", _FakeParallelLinear)

    def test_quant_config_propagates_to_shared_and_moe_parallel_linears(self, mocker: MockerFixture):
        self._patch_parallel_linears(mocker)
        mock_quant_config = mocker.MagicMock()

        model = DiTModel(_tiny_magi_human_config(), quant_config=mock_quant_config)

        moe_layer = model.block.layers[0]
        assert isinstance(moe_layer.attention.linear_qkv, MoEQKVParallelLinear)
        assert isinstance(moe_layer.attention.linear_proj, MoERowParallelLinear)
        assert isinstance(moe_layer.attention.linear_gating, MoEColumnParallelLinear)
        assert isinstance(moe_layer.mlp.up_gate_proj, MoEColumnParallelLinear)
        assert isinstance(moe_layer.mlp.down_proj, MoERowParallelLinear)

        for expert in moe_layer.attention.linear_qkv.experts:
            assert expert.quant_config is mock_quant_config
        for expert in moe_layer.attention.linear_proj.experts:
            assert expert.quant_config is mock_quant_config
        for expert in moe_layer.attention.linear_gating.experts:
            assert expert.quant_config is mock_quant_config
        for expert in moe_layer.mlp.up_gate_proj.experts:
            assert expert.quant_config is mock_quant_config
        for expert in moe_layer.mlp.down_proj.experts:
            assert expert.quant_config is mock_quant_config

        shared_layer = model.block.layers[1]
        assert shared_layer.attention.linear_qkv.quant_config is mock_quant_config
        assert shared_layer.attention.linear_proj.quant_config is mock_quant_config
        assert shared_layer.attention.linear_gating.quant_config is mock_quant_config
        assert shared_layer.mlp.up_gate_proj.quant_config is mock_quant_config
        assert shared_layer.mlp.down_proj.quant_config is mock_quant_config

    def test_none_quant_config_is_accepted(self, mocker: MockerFixture):
        self._patch_parallel_linears(mocker)
        model = DiTModel(_tiny_magi_human_config(), quant_config=None)

        moe_layer = model.block.layers[0]
        assert moe_layer.attention.linear_qkv.experts[0].quant_config is None
        assert moe_layer.attention.linear_proj.experts[0].quant_config is None
        assert moe_layer.attention.linear_gating.experts[0].quant_config is None
        assert moe_layer.mlp.up_gate_proj.experts[0].quant_config is None
        assert moe_layer.mlp.down_proj.experts[0].quant_config is None

        shared_layer = model.block.layers[1]
        assert shared_layer.attention.linear_qkv.quant_config is None
        assert shared_layer.attention.linear_proj.quant_config is None
        assert shared_layer.attention.linear_gating.quant_config is None
        assert shared_layer.mlp.up_gate_proj.quant_config is None
        assert shared_layer.mlp.down_proj.quant_config is None
