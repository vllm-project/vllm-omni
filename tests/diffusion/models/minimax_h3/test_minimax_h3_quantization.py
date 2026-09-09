# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


class _FakeLinear(nn.Module):
    def __init__(
        self,
        *args,
        bias=False,
        params_dtype=None,
        quant_config=None,
        prefix="",
        total_num_heads=None,
        total_num_kv_heads=None,
        **kwargs,
    ):
        del args, kwargs
        super().__init__()
        self.prefix = prefix
        self.quant_config = quant_config
        self.weight = nn.Parameter(torch.empty(1, dtype=params_dtype))
        if bias:
            self.bias = nn.Parameter(torch.empty(1, dtype=params_dtype))
        else:
            self.register_parameter("bias", None)
        self.num_heads = total_num_heads
        self.num_kv_heads = total_num_kv_heads


class _FakeAttention(nn.Module):
    def __init__(self, **kwargs):
        del kwargs
        super().__init__()


def _small_od_config():
    arch = {
        "num_layers": 1,
        "token_refiner_num_layers": 1,
        "hidden_size": 8,
        "num_attention_heads": 2,
        "attention_head_dim": 4,
        "ffn_hidden_size": 16,
        "latents_dim": 2,
        "audio_latents_dim": 2,
        "patch_size": (1, 2, 2),
        "text_dim": 6,
        "timestep_input_dim": 4,
        "time_embed_hidden_size": 8,
        "time_embed_dim": 4,
        "adaln_out_features": 18 * 8,
        "final_adaln_out_features": 2 * 8,
        "rope_inv_freq_len": 2,
    }
    return SimpleNamespace(
        tf_model_config=arch,
        parallel_config=SimpleNamespace(ulysses_degree=1),
    )


def test_fp8_scope_and_prefix_propagation(monkeypatch):
    from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        is_layer_skipped,
    )

    from vllm_omni.diffusion.models.minimax_h3 import minimax_h3_transformer as h3

    monkeypatch.setattr(h3, "ColumnParallelLinear", _FakeLinear)
    monkeypatch.setattr(h3, "MergedColumnParallelLinear", _FakeLinear)
    monkeypatch.setattr(h3, "QKVParallelLinear", _FakeLinear)
    monkeypatch.setattr(h3, "RowParallelLinear", _FakeLinear)
    monkeypatch.setattr(h3, "Attention", _FakeAttention)
    monkeypatch.setattr(h3, "get_tensor_model_parallel_world_size", lambda: 1)

    ignored_layers = {
        "token_refiner.blocks.0.mlp.fc2",
        "blocks.0.attn.qkv_proj",
        "condition_proj",
        "blocks.0.adaln_proj.linear",
        "final_layer.adaln_proj.linear",
    }
    fp8_config = Fp8Config(ignored_layers=sorted(ignored_layers))
    model = h3.MiniMaxH3DiTModel(
        _small_od_config(),
        quant_config=fp8_config,
    )
    linears = {module.prefix: module.quant_config for module in model.modules() if isinstance(module, _FakeLinear)}

    quantized = {
        "token_refiner.blocks.0.attn.qkv_proj",
        "token_refiner.blocks.0.attn.out_proj",
        "token_refiner.blocks.0.mlp.fc1",
        "token_refiner.blocks.0.mlp.fc2",
        "blocks.0.attn.qkv_proj",
        "blocks.0.attn.out_proj",
        "blocks.0.mlp.fc1",
        "blocks.0.mlp.fc2",
        "condition_proj",
        "blocks.0.adaln_proj.linear",
        "final_layer.adaln_proj.linear",
    }
    full_precision = {
        "video_patch_proj",
        "audio_patch_proj",
        "time_embedder.proj_in",
        "time_embedder.proj_out",
        "final_layer.video_out",
        "final_layer.audio_out",
    }

    assert quantized <= linears.keys()
    assert full_precision <= linears.keys()
    assert all(linears[prefix] is fp8_config for prefix in quantized)
    assert all(linears[prefix] is None for prefix in full_precision)
    assert {prefix for prefix in quantized if is_layer_skipped(prefix, fp8_config.ignored_layers)} == ignored_layers
    linear = Mock(spec=LinearBase)
    assert all(
        isinstance(
            fp8_config.get_quant_method(linear, prefix),
            UnquantizedLinearMethod,
        )
        for prefix in ignored_layers
    )


class _WeightTarget(nn.Module):
    def __init__(self, loader):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(1))
        self.weight.weight_loader = loader


def test_model_load_weights_transforms_before_calling_vllm_loader():
    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import (
        MiniMaxH3DiTArchConfig,
        MiniMaxH3DiTModel,
    )

    qkv_calls = []
    fc1_calls = []

    def qkv_loader(param, loaded_weight):
        del param
        qkv_calls.append(loaded_weight.clone())

    def fc1_loader(param, loaded_weight, shard_id):
        del param
        fc1_calls.append((shard_id, loaded_weight.clone()))

    model = object.__new__(MiniMaxH3DiTModel)
    nn.Module.__init__(model)
    model.arch = MiniMaxH3DiTArchConfig(
        hidden_size=1,
        num_attention_heads=2,
        attention_head_dim=1,
        ffn_hidden_size=2,
    )
    model.blocks = nn.ModuleList([nn.Module()])
    model.blocks[0].attn = nn.Module()
    model.blocks[0].attn.qkv_proj = _WeightTarget(qkv_loader)
    model.blocks[0].mlp = nn.Module()
    model.blocks[0].mlp.fc1 = _WeightTarget(fc1_loader)

    qkv = torch.arange(6, dtype=torch.float32).reshape(6, 1)
    fc1 = torch.arange(4, dtype=torch.float32).reshape(4, 1)
    loaded = model.load_weights(
        [
            ("blocks.0.attn.qkv_proj.weight", qkv),
            ("blocks.0.mlp.fc1.weight", fc1),
        ]
    )

    assert loaded == {
        "blocks.0.attn.qkv_proj.weight",
        "blocks.0.mlp.fc1.weight",
    }
    assert qkv_calls[0][:, 0].tolist() == [0, 3, 1, 4, 2, 5]
    assert [(shard_id, tensor[:, 0].tolist()) for shard_id, tensor in fc1_calls] == [
        (0, [0, 1]),
        (1, [2, 3]),
    ]


def test_loader_adapter_declares_equivalent_direct_mmap_transform(monkeypatch):
    from vllm_omni.diffusion.model_loader.checkpoint_adapters import (
        get_direct_mmap_adapter,
    )
    from vllm_omni.diffusion.models.minimax_h3 import minimax_h3_transformer as h3

    monkeypatch.setattr(h3, "QKVParallelLinear", _FakeLinear)
    monkeypatch.setattr(h3, "RowParallelLinear", _FakeLinear)
    monkeypatch.setattr(h3, "Attention", _FakeAttention)

    arch = h3.MiniMaxH3DiTArchConfig(
        hidden_size=1,
        num_attention_heads=2,
        attention_head_dim=1,
        rope_inv_freq_len=1,
    )
    attention = h3.MiniMaxH3Attention(
        arch,
        quant_config=None,
        prefix="blocks.0.attn",
    )
    transformer = object.__new__(h3.MiniMaxH3DiTModel)
    nn.Module.__init__(transformer)
    transformer.blocks = nn.ModuleList([nn.Module()])
    transformer.blocks[0].attn = attention
    pipeline = nn.Module()
    pipeline.transformer = transformer

    adapter = get_direct_mmap_adapter(pipeline)
    assert adapter is not None
    policy = adapter.policy_for(
        "transformer.blocks.0.attn.qkv_proj.weight",
        attention.qkv_proj.weight,
    )
    assert policy is not None
    assert policy.allow_custom_loader
    assert policy.transform is not None
    checkpoint_weight = torch.arange(6, dtype=torch.float32).reshape(6, 1)

    assert policy.transform(checkpoint_weight)[:, 0].tolist() == [0, 3, 1, 4, 2, 5]
    assert not hasattr(attention.qkv_proj.weight, "mmap_weight_transform")


def test_pipeline_resolves_transformer_component_quant_config():
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import (
        _resolve_component_quant_config,
    )
    from vllm_omni.quantization import build_quant_config

    ignored_layers = ["blocks.0.attn.qkv_proj"]
    component_config = build_quant_config(
        {
            "transformer": {
                "method": "fp8",
                "ignored_layers": ignored_layers,
            }
        }
    )
    transformer_config = component_config.resolve("transformer")

    assert transformer_config.ignored_layers == ignored_layers
    assert _resolve_component_quant_config(component_config, "transformer") is transformer_config
    assert _resolve_component_quant_config(transformer_config, "transformer") is transformer_config
