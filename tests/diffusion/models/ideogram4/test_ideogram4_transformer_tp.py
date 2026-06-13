# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for Ideogram4 tensor-parallel wiring."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm_omni.diffusion.models.ideogram4.ideogram4_transformer import (
    Ideogram4Config,
    Ideogram4Transformer,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _make_config(**overrides) -> Ideogram4Config:
    defaults = {
        "emb_dim": 32,
        "num_layers": 2,
        "num_heads": 4,
        "intermediate_size": 64,
        "adanln_dim": 16,
        "in_channels": 8,
        "llm_features_dim": 16,
        "rope_theta": 10000,
        "mrope_section": (4, 4, 4),
    }
    defaults.update(overrides)
    return Ideogram4Config(**defaults)


def _make_od_config(world_size: int = 1, sequence_parallel_size: int = 1):
    parallel_config = SimpleNamespace(
        world_size_across_dp=world_size,
        sequence_parallel_size=sequence_parallel_size,
        tensor_parallel_size=world_size,
    )
    return SimpleNamespace(
        dtype=torch.float32,
        parallel_config=parallel_config,
        use_layerwise_offload=False,
    )


def _build_transformer(world_size: int = 1):
    cfg = _make_config()
    od_cfg = _make_od_config(world_size=world_size)

    from vllm_omni.diffusion.models.ideogram4 import ideogram4_transformer as M

    instances: dict[str, object] = {}

    def fake_qkv(hidden_size, head_size, total_num_heads, bias, quant_config, prefix):
        inst = SimpleNamespace(
            hidden_size=hidden_size,
            head_size=head_size,
            total_num_heads=total_num_heads,
            num_heads=total_num_heads // world_size,
            num_kv_heads=total_num_heads // world_size,
            bias=bias,
            prefix=prefix,
        )
        instances[f"qkv::{prefix}"] = inst
        return inst

    def fake_col(in_features, out_features, bias, gather_output, return_bias, quant_config, prefix):
        inst = SimpleNamespace(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            gather_output=gather_output,
            return_bias=return_bias,
            prefix=prefix,
        )
        instances[f"col::{prefix}"] = inst
        return inst

    def fake_row(in_features, out_features, bias, input_is_parallel, return_bias, quant_config, prefix):
        inst = SimpleNamespace(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            input_is_parallel=input_is_parallel,
            return_bias=return_bias,
            prefix=prefix,
        )
        instances[f"row::{prefix}"] = inst
        return inst

    with (
        patch.object(M, "QKVParallelLinear", side_effect=fake_qkv),
        patch.object(M, "ColumnParallelLinear", side_effect=fake_col),
        patch.object(M, "RowParallelLinear", side_effect=fake_row),
    ):
        transformer = Ideogram4Transformer(od_config=od_cfg, config=cfg)

    return transformer, instances


def test_packed_modules_mapping_declares_qkv_merge():
    assert Ideogram4Transformer.packed_modules_mapping == {
        "to_qkv": ["to_q", "to_k", "to_v"],
    }


def test_sp_plan_covers_input_proj_and_final_layer():
    plan = Ideogram4Transformer._sp_plan
    assert "input_proj" in plan
    assert 0 in plan["input_proj"]
    assert "final_layer" in plan
    assert "linear" in plan["final_layer"]


@pytest.mark.parametrize("world_size", [1, 2, 4])
def test_attention_uses_qkv_parallel_linear(world_size):
    transformer, instances = _build_transformer(world_size=world_size)

    cfg = transformer.config
    head_dim = cfg.emb_dim // cfg.num_heads

    qkv_keys = [k for k in instances if k.startswith("qkv::")]
    assert len(qkv_keys) == cfg.num_layers, qkv_keys

    for key in qkv_keys:
        qkv = instances[key]
        assert qkv.total_num_heads == cfg.num_heads
        assert qkv.head_size == head_dim
        assert qkv.num_heads == cfg.num_heads // world_size
        assert qkv.bias is False


@pytest.mark.parametrize("world_size", [1, 2])
def test_attention_output_is_row_parallel(world_size):
    transformer, instances = _build_transformer(world_size=world_size)

    row_keys = [k for k in instances if k.startswith("row::") and k.endswith(".attention.to_out")]
    assert len(row_keys) == transformer.config.num_layers
    for key in row_keys:
        row = instances[key]
        assert row.input_is_parallel is True, "to_out must accept a parallel input"
        assert row.bias is False


@pytest.mark.parametrize("world_size", [1, 2])
def test_mlp_uses_column_parallel_gate_up_and_row_parallel_down(world_size):
    transformer, instances = _build_transformer(world_size=world_size)

    for i in range(transformer.config.num_layers):
        for name in ("w1", "w3"):
            key = f"col::layers.{i}.attention.mlp.{name}"
            assert key in instances, f"missing {key}"
            col = instances[key]
            assert col.gather_output is False, f"{name} must keep intermediate dim sharded"
            assert col.bias is False
        key = f"row::layers.{i}.attention.mlp.w2"
        assert key in instances, f"missing {key}"
        row = instances[key]
        assert row.input_is_parallel is True, "w2 must take the parallel hidden state"
        assert row.bias is False
