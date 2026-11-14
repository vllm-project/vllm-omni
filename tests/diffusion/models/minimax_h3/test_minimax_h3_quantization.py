# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import sys
from types import ModuleType, SimpleNamespace
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

    def add_schedule_candidate_check(self, check):
        del check


def _small_od_config(*, class_name=None):
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
    if class_name is not None:
        arch["_class_name"] = class_name
    return SimpleNamespace(
        tf_model_config=arch,
        parallel_config=SimpleNamespace(ulysses_degree=1),
    )


def test_fp8_scope_and_prefix_propagation(monkeypatch):
    from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        is_layer_skipped,
    )

    from vllm_omni.diffusion.models.minimax_h3 import minimax_h3_blocks as blocks
    from vllm_omni.diffusion.models.minimax_h3 import minimax_h3_transformer as h3
    from vllm_omni.quantization.fp8_config import DiffusionFp8Config

    monkeypatch.setattr(h3, "ColumnParallelLinear", _FakeLinear)
    monkeypatch.setattr(h3, "RowParallelLinear", _FakeLinear)
    monkeypatch.setattr(blocks, "ColumnParallelLinear", _FakeLinear)
    monkeypatch.setattr(blocks, "MergedColumnParallelLinear", _FakeLinear)
    monkeypatch.setattr(blocks, "QKVParallelLinear", _FakeLinear)
    monkeypatch.setattr(blocks, "RowParallelLinear", _FakeLinear)
    monkeypatch.setattr(blocks, "Attention", _FakeAttention)
    monkeypatch.setattr(h3, "get_tensor_model_parallel_world_size", lambda: 1)

    ignored_layers = {
        "token_refiner.blocks.0.mlp.fc2",
        "blocks.0.attn.qkv_proj",
        "condition_proj",
        "blocks.0.adaln_proj.linear",
        "final_layer.adaln_proj.linear",
    }
    fp8_config = DiffusionFp8Config(ignored_layers=sorted(ignored_layers))
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


def test_explicit_native_weight_format_overrides_diffusers_class_name(monkeypatch):
    from vllm_omni.diffusion.models.minimax_h3 import minimax_h3_blocks as blocks
    from vllm_omni.diffusion.models.minimax_h3 import minimax_h3_transformer as h3

    monkeypatch.setattr(blocks, "ColumnParallelLinear", _FakeLinear)
    monkeypatch.setattr(blocks, "MergedColumnParallelLinear", _FakeLinear)
    monkeypatch.setattr(blocks, "QKVParallelLinear", _FakeLinear)
    monkeypatch.setattr(blocks, "RowParallelLinear", _FakeLinear)
    monkeypatch.setattr(blocks, "Attention", _FakeAttention)
    monkeypatch.setattr(h3, "ColumnParallelLinear", _FakeLinear)
    monkeypatch.setattr(h3, "RowParallelLinear", _FakeLinear)
    monkeypatch.setattr(h3, "get_tensor_model_parallel_world_size", lambda: 1)

    model = h3.MiniMaxH3DiTModel(
        _small_od_config(class_name="MiniMaxH3Transformer3DModel"),
        diffusers_weights=False,
    )

    assert model._diffusers_weights is False


class _WeightTarget(nn.Module):
    def __init__(self, loader):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(1))
        self.weight.weight_loader = loader


class _MultiWeightTarget(nn.Module):
    def __init__(self, loaders):
        super().__init__()
        for name, loader in loaders.items():
            parameter = nn.Parameter(torch.empty(1))
            parameter.weight_loader = loader
            self.register_parameter(name, parameter)


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


def test_model_load_weights_reorders_serialized_svd_qkv_tensors():
    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import (
        MiniMaxH3DiTArchConfig,
        MiniMaxH3DiTModel,
    )

    calls = {}

    def capture(name):
        def loader(param, loaded_weight):
            del param
            calls[name] = loaded_weight.clone()

        return loader

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
    model.blocks[0].attn.qkv_proj = _MultiWeightTarget(
        {name: capture(name) for name in ("qweight", "wscales", "proj_up", "wcscales")}
    )

    qweight = torch.arange(12, dtype=torch.int8).reshape(6, 2)
    wscales = torch.arange(18, dtype=torch.uint8).reshape(3, 6)
    proj_up = torch.arange(6, dtype=torch.bfloat16).reshape(6, 1)
    wcscales = torch.arange(6, dtype=torch.bfloat16)
    names = ("qweight", "wscales", "proj_up", "wcscales")
    loaded = model.load_weights(
        [
            (f"blocks.0.attn.qkv_proj.{name}", tensor)
            for name, tensor in zip(names, (qweight, wscales, proj_up, wcscales))
        ]
    )

    assert loaded == {f"blocks.0.attn.qkv_proj.{name}" for name in names}
    expected_rows = [0, 3, 1, 4, 2, 5]
    assert calls["qweight"][:, 0].tolist() == [row * qweight.shape[1] for row in expected_rows]
    assert calls["proj_up"][:, 0].tolist() == expected_rows
    assert calls["wcscales"].tolist() == expected_rows
    assert calls["wscales"][0, :].tolist() == expected_rows


def test_loader_adapter_declares_equivalent_direct_mmap_transform(monkeypatch):
    from vllm_omni.diffusion.model_loader.checkpoint_adapters import (
        get_direct_mmap_adapter,
    )
    from vllm_omni.diffusion.models.minimax_h3 import minimax_h3_blocks as blocks
    from vllm_omni.diffusion.models.minimax_h3 import minimax_h3_transformer as h3

    monkeypatch.setattr(blocks, "QKVParallelLinear", _FakeLinear)
    monkeypatch.setattr(blocks, "RowParallelLinear", _FakeLinear)
    monkeypatch.setattr(blocks, "Attention", _FakeAttention)

    arch = blocks.MiniMaxH3DiTArchConfig(
        hidden_size=1,
        num_attention_heads=2,
        attention_head_dim=1,
        rope_inv_freq_len=1,
    )
    attention = blocks.MiniMaxH3Attention(
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


@pytest.mark.parametrize("case", ["offline", "online", "missing_op", "non_int8", "wrapped"])
def test_mlp_optional_mindiesd_int8_swiglu(mocker, case):
    from vllm.model_executor.layers.linear import MergedColumnParallelLinear, UnquantizedLinearMethod

    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_blocks import MiniMaxH3MLP
    from vllm_omni.quantization.int8_config import (
        DiffusionInt8Config,
        NPUInt8LinearMethod,
        NPUInt8OnlineLinearMethod,
    )

    input_tensor = torch.zeros(2, 3, 4, dtype=torch.bfloat16)
    x = mocker.Mock(
        wraps=input_tensor,
        device=mocker.Mock(type="npu"),
        shape=input_tensor.shape,
        dtype=input_tensor.dtype,
    )
    linear_output = torch.zeros(2, 3, 10, dtype=x.dtype)
    activated = torch.zeros(2, 3, 5, dtype=x.dtype)
    output = torch.zeros_like(input_tensor)
    fc1 = MergedColumnParallelLinear.__new__(MergedColumnParallelLinear)
    nn.Module.__init__(fc1)
    method = NPUInt8OnlineLinearMethod if case == "online" else NPUInt8LinearMethod
    fc1.quant_method = (
        UnquantizedLinearMethod()
        if case == "non_int8"
        else method(DiffusionInt8Config(is_checkpoint_int8_serialized=case != "online"))
    )
    weight_dtype = x.dtype if case == "non_int8" else torch.int8
    fc1.weight = nn.Parameter(torch.zeros(4, 10, dtype=weight_dtype), requires_grad=False)
    fc1.weight_scale = nn.Parameter(torch.ones(10), requires_grad=False)
    fc1.forward = mocker.Mock(return_value=(linear_output, None))
    mlp = MiniMaxH3MLP.__new__(MiniMaxH3MLP)
    nn.Module.__init__(mlp)
    mlp.fc1 = mocker.Mock(wraps=fc1, quant_method=fc1.quant_method) if case == "wrapped" else fc1
    mlp.act_fn = mocker.Mock(return_value=activated)
    mlp.fc2 = mocker.Mock(return_value=(output, None))

    quantized = torch.zeros(6, 4, dtype=torch.int8)
    pertoken_scale = torch.ones(6)
    fused_output = torch.arange(30, dtype=x.dtype).reshape(6, 5)
    torch_npu = ModuleType("torch_npu")
    torch_npu.__dict__["npu_dynamic_quant"] = mocker.Mock(return_value=(quantized, pertoken_scale))
    custom_ops = ModuleType("mindiesd.layers._custom_ops")
    fused_op = mocker.Mock(return_value=fused_output)
    if case != "missing_op":
        custom_ops.__dict__["eagle_quant_batch_matmul_swiglu"] = fused_op
    mocker.patch.dict(
        sys.modules,
        {
            "torch_npu": torch_npu,
            "mindiesd": ModuleType("mindiesd"),
            "mindiesd.layers": ModuleType("mindiesd.layers"),
            "mindiesd.layers._custom_ops": custom_ops,
        },
    )

    assert mlp(x) is output

    if case in {"offline", "online"}:
        x.reshape.assert_called_once_with(-1, 4)
        torch_npu.npu_dynamic_quant.assert_called_once()
        torch.testing.assert_close(torch_npu.npu_dynamic_quant.call_args.args[0], input_tensor.reshape(6, 4))
        fused_op.assert_called_once_with(
            quantized, fc1.weight, fc1.weight_scale, pertoken_scale=pertoken_scale, output_dtype=x.dtype
        )
        mlp.fc2.assert_called_once()
        torch.testing.assert_close(mlp.fc2.call_args.args[0], fused_output.reshape(2, 3, 5))
        fc1.forward.assert_not_called()
        mlp.act_fn.assert_not_called()
    else:
        x.reshape.assert_not_called()
        torch_npu.npu_dynamic_quant.assert_not_called()
        fused_op.assert_not_called()
        fc1.forward.assert_called_once_with(x)
        mlp.act_fn.assert_called_once_with(linear_output)
        mlp.fc2.assert_called_once_with(activated)
        if case == "wrapped":
            mlp.fc1.assert_called_once_with(x)
