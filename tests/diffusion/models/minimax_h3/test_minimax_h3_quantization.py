# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import sys
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


def _swiglu_inputs(mocker, quant_method, *, device_name="A2", wrapped=False):
    from vllm.model_executor.layers.linear import MergedColumnParallelLinear

    from vllm_omni.diffusion.models.minimax_h3 import minimax_h3_blocks as blocks

    mocker.patch.object(blocks, "_is_ascend_device_type", side_effect=lambda *allowed: device_name in allowed)
    x = Mock(device=SimpleNamespace(type="npu"), shape=(2, 3, 4), dtype=torch.bfloat16)
    fc1 = MergedColumnParallelLinear.__new__(MergedColumnParallelLinear)
    nn.Module.__init__(fc1)
    fc1.quant_method = quant_method
    fc1.weight = mocker.sentinel.weight
    fc1.weight_scale = mocker.sentinel.weight_scale
    fc1.forward = Mock(return_value=(mocker.sentinel.projected, None))

    mlp = blocks.MiniMaxH3MLP.__new__(blocks.MiniMaxH3MLP)
    nn.Module.__init__(mlp)
    mlp.fc1 = Mock(wraps=fc1, quant_method=quant_method) if wrapped else fc1
    mlp.act_fn = Mock()
    mlp.fc2 = Mock(return_value=(mocker.sentinel.output, None))

    quantize = Mock(return_value=(mocker.sentinel.quantized, mocker.sentinel.scale))
    fused = Mock(return_value=torch.arange(30, dtype=x.dtype).reshape(6, 5))
    mocker.patch.dict(
        sys.modules,
        {
            "torch_npu": SimpleNamespace(npu_dynamic_quant=quantize),
            "mindiesd.fused_ops": SimpleNamespace(eagle_quant_batch_matmul_swiglu=fused),
        },
    )
    return mlp, x, quantize, fused


def test_mlp_calls_mindiesd_int8_swiglu(mocker):
    from vllm_omni.quantization.int8_config import DiffusionInt8Config, NPUInt8OnlineLinearMethod

    mlp, x, quantize, fused = _swiglu_inputs(mocker, NPUInt8OnlineLinearMethod(DiffusionInt8Config()))
    assert mlp(x) is mocker.sentinel.output

    x.reshape.assert_called_once_with(-1, 4)
    quantize.assert_called_once_with(x.reshape.return_value)
    fused.assert_called_once_with(
        mocker.sentinel.quantized,
        mlp.fc1.weight,
        mlp.fc1.weight_scale,
        pertoken_scale=mocker.sentinel.scale,
        output_dtype=x.dtype,
    )
    mlp.fc2.assert_called_once()
    torch.testing.assert_close(mlp.fc2.call_args.args[0], fused.return_value.reshape(2, 3, 5))
    mlp.fc1.forward.assert_not_called()
    mlp.act_fn.assert_not_called()


@pytest.mark.parametrize("case", ["missing_sdk", "missing_native", "non_int8", "wrapped", "A5"])
def test_mlp_preserves_swiglu_fallback(mocker, case):
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    from vllm_omni.quantization.int8_config import DiffusionInt8Config, NPUInt8LinearMethod

    method = NPUInt8LinearMethod(DiffusionInt8Config(is_checkpoint_int8_serialized=True))
    if case == "non_int8":
        method = UnquantizedLinearMethod()
    mlp, x, quantize, fused = _swiglu_inputs(
        mocker, method, device_name="A5" if case == "A5" else "A2", wrapped=case == "wrapped"
    )
    if case == "missing_sdk":
        mocker.patch.dict(sys.modules, {"mindiesd.fused_ops": None})
    elif case == "missing_native":
        fused.side_effect = ImportError("native SwiGLU unavailable")

    assert mlp(x) is mocker.sentinel.output

    expected_calls = 1 if case == "missing_native" else 0
    assert quantize.call_count == fused.call_count == expected_calls
    if case == "wrapped":
        mlp.fc1.assert_called_once_with(x)
    else:
        mlp.fc1.forward.assert_called_once_with(x)
    mlp.act_fn.assert_called_once_with(mocker.sentinel.projected)
    mlp.fc2.assert_called_once_with(mlp.act_fn.return_value)
