# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import json
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file

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


def test_fp8_scope_and_prefix_propagation(monkeypatch, mocker):
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
    linear = mocker.Mock(spec=LinearBase)
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


class _QuantWeightTarget(_WeightTarget):
    def __init__(self, weight_loader, scale_loader):
        from vllm.model_executor.parameter import ChannelQuantScaleParameter

        super().__init__(weight_loader)
        self.weight_scale = ChannelQuantScaleParameter(
            data=torch.empty((1, 1), dtype=torch.float32),
            output_dim=0,
            weight_loader=scale_loader,
        )


class _OtherQuantWeightTarget(_WeightTarget):
    def __init__(self, weight_loader, scale_loader):
        super().__init__(weight_loader)
        self.weight_scale = nn.Parameter(torch.empty(1))
        self.weight_scale.weight_loader = scale_loader


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


def test_model_load_weights_applies_fused_transforms_to_int8_scales(monkeypatch):
    import vllm.model_executor.parameter as parameter_module

    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import (
        MiniMaxH3DiTArchConfig,
        MiniMaxH3DiTModel,
    )

    monkeypatch.setattr(parameter_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(parameter_module, "get_tensor_model_parallel_world_size", lambda: 1)

    qkv_scale_calls = []
    fc1_scale_calls = []
    other_scale_calls = []

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
    model.blocks[0].attn.qkv_proj = _QuantWeightTarget(
        lambda *_: None,
        lambda _param, value: qkv_scale_calls.append(value.clone()),
    )
    model.blocks[0].attn.out_proj = _OtherQuantWeightTarget(
        lambda *_: None,
        lambda _param, value: other_scale_calls.append(value.clone()),
    )
    model.blocks[0].mlp = nn.Module()
    model.blocks[0].mlp.fc1 = _QuantWeightTarget(
        lambda *_: None,
        lambda _param, value, shard_id: fc1_scale_calls.append((shard_id, value.clone())),
    )

    qkv_scale = torch.arange(6, dtype=torch.float32)
    fc1_scale = torch.arange(4, dtype=torch.float32)
    loaded = model.load_weights(
        [
            ("blocks.0.attn.qkv_proj.weight_scale", qkv_scale),
            ("blocks.0.mlp.fc1.weight_scale", fc1_scale),
            ("blocks.0.attn.out_proj.weight_scale", torch.arange(3, dtype=torch.float32)),
            (
                "blocks.0.attn.qkv_proj.comfy_quant",
                torch.tensor(list(b"{}"), dtype=torch.uint8),
            ),
        ]
    )

    assert loaded == {
        "blocks.0.attn.qkv_proj.weight_scale",
        "blocks.0.mlp.fc1.weight_scale",
        "blocks.0.attn.out_proj.weight_scale",
        "blocks.0.attn.qkv_proj.comfy_quant",
    }
    assert qkv_scale_calls[0][:, 0].tolist() == [0, 3, 1, 4, 2, 5]
    assert [(shard_id, tensor[:, 0].tolist()) for shard_id, tensor in fc1_scale_calls] == [
        (0, [0, 1]),
        (1, [2, 3]),
    ]
    assert other_scale_calls[0].shape == (3,)


def test_model_load_weights_preserves_comfy_int8_runtime_qkv_layout(monkeypatch):
    import vllm.model_executor.parameter as parameter_module

    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import (
        MiniMaxH3DiTArchConfig,
        MiniMaxH3DiTModel,
    )

    monkeypatch.setattr(parameter_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(parameter_module, "get_tensor_model_parallel_world_size", lambda: 1)

    weight_calls = []
    scale_calls = []
    model = object.__new__(MiniMaxH3DiTModel)
    nn.Module.__init__(model)
    model.arch = MiniMaxH3DiTArchConfig(
        hidden_size=1,
        num_attention_heads=2,
        attention_head_dim=1,
        ffn_hidden_size=2,
    )
    model._qkv_checkpoint_is_runtime_layout = True
    model.blocks = nn.ModuleList([nn.Module()])
    model.blocks[0].attn = nn.Module()
    model.blocks[0].attn.qkv_proj = _QuantWeightTarget(
        lambda _param, value: weight_calls.append(value.clone()),
        lambda _param, value: scale_calls.append(value.clone()),
    )

    # Comfy serializes all query rows first, then key rows, then value rows.
    qkv = torch.arange(6, dtype=torch.float32).reshape(6, 1)
    qkv_scale = torch.arange(6, dtype=torch.float32)
    loaded = model.load_weights(
        [
            ("blocks.0.attn.qkv_proj.weight", qkv),
            ("blocks.0.attn.qkv_proj.weight_scale", qkv_scale),
        ]
    )

    assert loaded == {
        "blocks.0.attn.qkv_proj.weight",
        "blocks.0.attn.qkv_proj.weight_scale",
    }
    torch.testing.assert_close(weight_calls[0], qkv)
    torch.testing.assert_close(scale_calls[0], qkv_scale.unsqueeze(1))


def test_model_load_weights_leaves_non_channel_fused_scales_to_their_loader():
    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import (
        MiniMaxH3DiTArchConfig,
        MiniMaxH3DiTModel,
    )

    qkv_calls = []
    fc1_calls = []
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
    model.blocks[0].attn.qkv_proj = _OtherQuantWeightTarget(
        lambda *_: None,
        lambda _param, value: qkv_calls.append(value.clone()),
    )
    model.blocks[0].mlp = nn.Module()
    model.blocks[0].mlp.fc1 = _OtherQuantWeightTarget(
        lambda *_: None,
        lambda _param, value: fc1_calls.append(value.clone()),
    )
    qkv_scale = torch.tensor([1.0, 2.0, 3.0])
    fc1_scale = torch.tensor([4.0, 5.0])

    loaded = model.load_weights(
        [
            ("blocks.0.attn.qkv_proj.weight_scale", qkv_scale),
            ("blocks.0.mlp.fc1.weight_scale", fc1_scale),
        ]
    )

    assert loaded == {
        "blocks.0.attn.qkv_proj.weight_scale",
        "blocks.0.mlp.fc1.weight_scale",
    }
    torch.testing.assert_close(qkv_calls[0], qkv_scale)
    torch.testing.assert_close(fc1_calls[0], fc1_scale)


def test_int8_convrot_factory_routes_only_checkpoint_marked_layers(mocker):
    from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod

    from vllm_omni.quantization import build_quant_config
    from vllm_omni.quantization.int8_convrot_config import Int8ConvRotLinearMethod

    config = build_quant_config(
        {
            "method": "int8_convrot",
            "quantized_layers": ["blocks.0.attn.qkv_proj"],
        }
    )
    linear = mocker.Mock(spec=LinearBase)

    assert config.get_name() == "int8_convrot"
    assert isinstance(config.get_quant_method(linear, "blocks.0.attn.qkv_proj"), Int8ConvRotLinearMethod)
    assert isinstance(config.get_quant_method(linear, "blocks.0.attn.out_proj"), UnquantizedLinearMethod)


def test_int8_convrot_rejects_ignored_checkpoint_layers():
    from vllm_omni.quantization.int8_convrot_config import (
        DiffusionInt8ConvRotConfig,
        Int8ConvRotLayerConfig,
    )

    prefix = "blocks.0.attn.qkv_proj"
    with pytest.raises(ValueError, match="cannot also be ignored"):
        DiffusionInt8ConvRotConfig(
            quantized_layers=[prefix],
            ignored_layers=[prefix],
        )
    config = DiffusionInt8ConvRotConfig(ignored_layers=[prefix])
    with pytest.raises(ValueError, match="cannot also be ignored"):
        config.configure_layers({prefix: Int8ConvRotLayerConfig(True, 256)})


@pytest.mark.parametrize("prefix", ["final_layer.video_out", "blocks.99.missing"])
def test_int8_convrot_rejects_checkpoint_markers_without_executable_binding(prefix):
    from vllm_omni.quantization.int8_convrot_config import (
        DiffusionInt8ConvRotConfig,
    )

    config = DiffusionInt8ConvRotConfig(quantized_layers=[prefix])

    with pytest.raises(ValueError, match="unbound markers"):
        config.validate_model_bindings(nn.Module())


def test_int8_convrot_accepts_exact_executable_bindings():
    from vllm_omni.quantization.int8_convrot_config import (
        DiffusionInt8ConvRotConfig,
        Int8ConvRotLayerConfig,
        Int8ConvRotLinearMethod,
    )

    prefix = "blocks.0.attn.qkv_proj"
    config = DiffusionInt8ConvRotConfig(quantized_layers=[prefix])
    model = nn.Module()
    model.linear = nn.Linear(1, 1)
    model.linear.quant_method = Int8ConvRotLinearMethod(
        config,
        Int8ConvRotLayerConfig(convrot=True),
        prefix=prefix,
    )

    config.validate_model_bindings(model)


def test_component_config_reports_offline_only_when_every_quantizer_is_offline():
    from vllm_omni.quantization import build_quant_config

    offline = build_quant_config(
        {
            "transformer": {
                "method": "int8_convrot",
                "quantized_layers": ["blocks.0.attn.qkv_proj"],
            }
        }
    )
    mixed = build_quant_config(
        {
            "transformer": {
                "method": "int8_convrot",
                "quantized_layers": ["blocks.0.attn.qkv_proj"],
            },
            "text_encoder": {"method": "fp8"},
        }
    )

    assert offline.is_checkpoint_quantized
    assert not mixed.is_checkpoint_quantized


def test_int8_convrot_checkpoint_and_quantization_must_be_configured_together():
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import (
        _validate_int8_convrot_override_pair,
    )
    from vllm_omni.quantization import build_quant_config

    config = build_quant_config({"method": "int8_convrot"})

    _validate_int8_convrot_override_pair(False, None)
    _validate_int8_convrot_override_pair(True, config)
    with pytest.raises(ValueError, match="requires a validated Comfy checkpoint"):
        _validate_int8_convrot_override_pair(False, config)
    with pytest.raises(ValueError, match="requires component quantization"):
        _validate_int8_convrot_override_pair(True, None)


def test_int8_convrot_apply_resolves_native_cuda_backend_once(monkeypatch, mocker):
    from vllm_omni.quantization.int8_convrot_config import (
        DiffusionInt8ConvRotConfig,
        Int8ConvRotLayerConfig,
        Int8ConvRotLinearMethod,
    )

    output = object()
    implementation = mocker.Mock(return_value=output)
    custom_op = mocker.Mock(return_value=output)
    registry = SimpleNamespace(get_implementation=mocker.Mock(return_value=implementation))
    kitchen = SimpleNamespace(registry=registry)
    method = Int8ConvRotLinearMethod(
        DiffusionInt8ConvRotConfig(),
        Int8ConvRotLayerConfig(convrot=True, convrot_groupsize=256),
        prefix="blocks.0.attn.qkv_proj",
    )
    monkeypatch.setattr(method, "_load_comfy_kitchen", lambda: kitchen)
    monkeypatch.setattr(method, "_run_custom_op", custom_op)
    activation = mocker.Mock(spec=torch.Tensor)
    activation.is_cuda = True
    activation.dtype = torch.bfloat16
    layer = nn.Module()
    layer.weight = object()
    layer.weight_scale = object()

    assert method.apply(layer, activation) is output
    assert method.apply(layer, activation) is output
    registry.get_implementation.assert_called_once()
    assert registry.get_implementation.call_args.kwargs["backend"] == "cuda"
    implementation.assert_not_called()
    assert custom_op.call_count == 2

    activation.dtype = torch.float32
    with pytest.raises(TypeError, match="supports only FP16/BF16"):
        method.apply(layer, activation)
    assert custom_op.call_count == 2


def test_comfy_checkpoint_inspection_derives_quantized_layers_and_curve_arch(tmp_path):
    from vllm_omni.diffusion.models.minimax_h3.comfy_checkpoint import (
        inspect_comfy_checkpoint,
        resolve_comfy_checkpoint_path,
    )

    marker = {
        "format": "int8_tensorwise",
        "convrot": True,
        "convrot_groupsize": 4,
    }
    checkpoint_path = tmp_path / "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
    prefix = "blocks.0.attn.qkv_proj"
    save_file(
        {
            f"{prefix}.weight": torch.ones((6, 4), dtype=torch.int8),
            f"{prefix}.weight_scale": torch.ones((6, 1), dtype=torch.float32),
            f"{prefix}.comfy_quant": torch.tensor(list(json.dumps(marker).encode()), dtype=torch.uint8),
            "adaln_t_table": torch.arange(10, dtype=torch.float32).reshape(5, 2),
        },
        checkpoint_path,
    )

    info = inspect_comfy_checkpoint(checkpoint_path)

    assert resolve_comfy_checkpoint_path(str(tmp_path)) == checkpoint_path
    assert info.partition == "fl2va"
    assert set(info.layer_configs) == {prefix}
    assert info.layer_configs[prefix].convrot
    assert info.layer_configs[prefix].convrot_groupsize == 4
    assert info.arch_overrides == {"adaln_curve_grid": 5, "adaln_curve_dim": 2}


def test_comfy_checkpoint_resolution_preserves_snapshot_symlink_suffix(tmp_path):
    from vllm_omni.diffusion.models.minimax_h3.comfy_checkpoint import (
        inspect_comfy_checkpoint,
        resolve_comfy_checkpoint_path,
    )

    blob = tmp_path / "blobs" / "0123456789abcdef"
    blob.parent.mkdir()
    marker = json.dumps(
        {
            "format": "int8_tensorwise",
            "convrot": True,
            "convrot_groupsize": 4,
        }
    ).encode()
    save_file(
        {
            "blocks.0.attn.qkv_proj.weight": torch.ones((4, 4), dtype=torch.int8),
            "blocks.0.attn.qkv_proj.weight_scale": torch.ones((4, 1), dtype=torch.float32),
            "blocks.0.attn.qkv_proj.comfy_quant": torch.tensor(list(marker), dtype=torch.uint8),
        },
        blob,
    )
    snapshot = tmp_path / "snapshots" / "main" / "minimax_h3_ref2va_int8_convrot.safetensors"
    snapshot.parent.mkdir(parents=True)
    snapshot.symlink_to(blob)

    resolved = resolve_comfy_checkpoint_path(str(snapshot))

    assert resolved == snapshot.absolute()
    assert resolved.suffix == ".safetensors"
    assert resolved.is_symlink()
    info = inspect_comfy_checkpoint(resolved)
    assert info.partition == "ref2va"
    assert set(info.layer_configs) == {"blocks.0.attn.qkv_proj"}


def test_comfy_checkpoint_resolution_uses_shared_hf_downloader(monkeypatch, tmp_path):
    from vllm_omni.diffusion.models.minimax_h3 import comfy_checkpoint

    calls = []
    snapshot = tmp_path / "snapshot"
    target = snapshot / "diffusion_models" / "minimax_h3_fl2va_pruned_int8_convrot.safetensors"

    def fake_download(**kwargs):
        calls.append(kwargs)
        return str(snapshot)

    monkeypatch.setattr(
        comfy_checkpoint,
        "download_weights_from_hf_specific",
        fake_download,
    )

    result = comfy_checkpoint.resolve_comfy_checkpoint_path(
        "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/"
        "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors"
    )

    assert result == target
    assert calls == [
        {
            "model_name_or_path": "Comfy-Org/MiniMax-H3",
            "cache_dir": None,
            "allow_patterns": ["diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors"],
            "revision": "main",
            "require_all": True,
        }
    ]


@pytest.mark.parametrize(
    "name",
    [
        "transformer.safetensors",
        "minimax_h3_notfl2va_int8_convrot.safetensors",
        "minimax_h3_fl2va_ref2va_int8_convrot.safetensors",
    ],
)
def test_comfy_checkpoint_rejects_missing_or_ambiguous_partition_token(tmp_path, name):
    from vllm_omni.diffusion.models.minimax_h3.comfy_checkpoint import inspect_comfy_checkpoint

    checkpoint = tmp_path / name
    save_file({}, checkpoint)

    with pytest.raises(ValueError, match="must identify exactly one partition"):
        inspect_comfy_checkpoint(checkpoint)


def test_comfy_checkpoint_rejects_wrong_serving_partition(tmp_path):
    from vllm_omni.diffusion.models.minimax_h3.comfy_checkpoint import inspect_comfy_checkpoint

    checkpoint = tmp_path / "minimax_h3_ref2va_int8_convrot.safetensors"
    save_file({}, checkpoint)

    with pytest.raises(ValueError, match="REF2VA checkpoint .* cannot serve the FL2VA partition"):
        inspect_comfy_checkpoint(checkpoint, expected_partition="fl2va")


def test_comfy_checkpoint_rejects_unmarked_int8_weights(tmp_path):
    from vllm_omni.diffusion.models.minimax_h3.comfy_checkpoint import inspect_comfy_checkpoint

    marker = json.dumps(
        {
            "format": "int8_tensorwise",
            "convrot": True,
            "convrot_groupsize": 4,
        }
    ).encode()
    checkpoint = tmp_path / "minimax_h3_fl2va_unmarked.safetensors"
    save_file(
        {
            "blocks.0.attn.qkv_proj.weight": torch.ones((4, 4), dtype=torch.int8),
            "blocks.0.attn.qkv_proj.weight_scale": torch.ones((4, 1), dtype=torch.float32),
            "blocks.0.attn.qkv_proj.comfy_quant": torch.tensor(list(marker), dtype=torch.uint8),
            "blocks.1.attn.out_proj.weight": torch.ones((4, 4), dtype=torch.int8),
            "blocks.1.attn.out_proj.weight_scale": torch.ones((4, 1), dtype=torch.float32),
        },
        checkpoint,
    )

    with pytest.raises(ValueError, match="missing markers"):
        inspect_comfy_checkpoint(checkpoint)


def test_adaln_curve_model_interpolates_without_dense_time_embedder(monkeypatch):
    from vllm_omni.diffusion.models.minimax_h3 import minimax_h3_transformer as h3

    monkeypatch.setattr(h3, "ColumnParallelLinear", _FakeLinear)
    monkeypatch.setattr(h3, "MergedColumnParallelLinear", _FakeLinear)
    monkeypatch.setattr(h3, "QKVParallelLinear", _FakeLinear)
    monkeypatch.setattr(h3, "RowParallelLinear", _FakeLinear)
    monkeypatch.setattr(h3, "Attention", _FakeAttention)
    monkeypatch.setattr(h3, "get_tensor_model_parallel_world_size", lambda: 1)

    model = h3.MiniMaxH3DiTModel(
        _small_od_config(),
        arch_overrides={"adaln_curve_grid": 3, "adaln_curve_dim": 2},
    )
    model.adaln_t_table.copy_(
        torch.tensor(
            [
                [0.0, 10.0],
                [2.0, 12.0],
                [4.0, 14.0],
            ]
        )
    )

    assert model.time_embedder is None
    assert model.blocks[0].adaln_proj.linear.weight.dtype == torch.float32
    torch.testing.assert_close(
        model._embed_timesteps(torch.tensor([0.0, 0.25, 1.0])),
        torch.tensor([[0.0, 10.0], [1.0, 11.0], [4.0, 14.0]]),
    )


@pytest.mark.parametrize("adaln_dtype", [torch.float32, torch.bfloat16])
def test_final_layer_matches_comfy_adaln_precision(adaln_dtype):
    from vllm_omni.diffusion.attention.ops.minimax_h3_modulation import (
        indexed_scale_shift_,
    )
    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import (
        MiniMaxH3FinalLayer,
    )

    class _FixedAdaln(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_dtype = adaln_dtype
            self.register_buffer(
                "shift",
                torch.tensor([[0.0031, -0.0073]], dtype=adaln_dtype),
            )
            self.register_buffer(
                "scale",
                torch.tensor([[0.0127, -0.0189]], dtype=adaln_dtype),
            )

        def forward(self, _t_emb):
            return self.shift, self.scale

    class _IdentityHead(nn.Module):
        def forward(self, hidden):
            return hidden, None

    layer = object.__new__(MiniMaxH3FinalLayer)
    nn.Module.__init__(layer)
    layer.norm = nn.Identity()
    layer.adaln_proj = _FixedAdaln()
    layer.video_out = _IdentityHead()
    layer.audio_out = _IdentityHead()

    hidden = torch.tensor([[1.25, -0.75]], dtype=torch.bfloat16)
    indices = torch.tensor([0], dtype=torch.long)
    shift, scale = layer.adaln_proj(torch.empty(0))
    # This is Comfy's out-of-place expression. FP32 curve modulation promotes
    # the result; dense BF16 modulation intentionally retains BF16 rounding.
    comfy_result = (hidden * (1.0 + scale[indices]) + shift[indices]).to(torch.float32)
    legacy_in_place_result = indexed_scale_shift_(
        hidden.clone(),
        shift,
        scale,
        indices,
    ).to(torch.float32)
    expected = comfy_result if adaln_dtype == torch.float32 else legacy_in_place_result

    video, audio = layer(
        hidden,
        t_emb=torch.empty(0),
        inverse_indices=indices,
    )

    torch.testing.assert_close(video, expected, rtol=0, atol=0)
    torch.testing.assert_close(audio, expected, rtol=0, atol=0)
    if adaln_dtype == torch.float32:
        assert not torch.equal(comfy_result, legacy_in_place_result)


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
