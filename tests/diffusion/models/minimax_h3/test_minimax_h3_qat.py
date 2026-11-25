# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import json

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import MiniMaxH3DiTArchConfig, MiniMaxH3DiTModel
from vllm_omni.diffusion.models.minimax_h3.qat import (
    is_serialized_mxfp4,
    resolve_h3_transformer_checkpoint,
    resolve_h3_transformer_quant_config,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

_WeightCall = tuple[str, torch.Tensor, tuple[int, ...]]


def _native_model():
    model = MiniMaxH3DiTModel.__new__(MiniMaxH3DiTModel)
    nn.Module.__init__(model)
    model.arch = MiniMaxH3DiTArchConfig(num_attention_heads=2, attention_head_dim=1)
    model.blocks = nn.ModuleList([nn.Module()])
    model.blocks[0].attn = nn.Module()
    model.blocks[0].attn.qkv_proj = nn.Module()
    model.blocks[0].mlp = nn.Module()
    model.blocks[0].mlp.fc1 = nn.Module()
    return model


def _target(module: nn.Module, field: str, value: torch.Tensor, calls: list[_WeightCall]) -> None:
    param = nn.Parameter(torch.empty_like(value), requires_grad=False)

    def loader(param: nn.Parameter, weight: torch.Tensor, *shard: int) -> None:
        del param
        calls.append((field, weight.clone(), shard))

    param.weight_loader = loader
    module.register_parameter(field, param)


def test_native_qkv_scales_follow_weight_rows_and_smooth_keeps_input_order():
    model = _native_model()
    calls: list[_WeightCall] = []
    values = {
        "weight": torch.arange(12, dtype=torch.float32).reshape(6, 2),
        "weight_scale": torch.arange(12, dtype=torch.uint8).reshape(6, 2),
        "weight_dual_scale": torch.arange(12, dtype=torch.float32).reshape(6, 2, 1),
        "mul_scale": torch.tensor([1.25, 2.5]),
    }
    for field, value in values.items():
        _target(model.blocks[0].attn.qkv_proj, field, value, calls)
    loaded = model.load_weights((f"blocks.0.attn.qkv_proj.{field}", value) for field, value in values.items())
    assert len(loaded) == len(values)
    for field, value, shard in calls:
        expected = values[field] if field == "mul_scale" else values[field][[0, 3, 1, 4, 2, 5]]
        torch.testing.assert_close(value, expected)
        assert shard == ()


def test_native_fc1_scales_split_gate_up_and_smooth_is_replicated():
    model = _native_model()
    calls: list[_WeightCall] = []
    values = {
        "weight": torch.arange(8, dtype=torch.float32).reshape(4, 2),
        "weight_scale": torch.arange(8, dtype=torch.uint8).reshape(4, 2),
        "weight_dual_scale": torch.arange(8, dtype=torch.float32).reshape(4, 2, 1),
        "mul_scale": torch.tensor([1.25, 2.5]),
    }
    for field, value in values.items():
        _target(model.blocks[0].mlp.fc1, field, value, calls)
    model.load_weights((f"blocks.0.mlp.fc1.{field}", value) for field, value in values.items())
    for field in ("weight", "weight_scale", "weight_dual_scale"):
        assert [(value.tolist(), shard) for name, value, shard in calls if name == field] == [
            (values[field][:2].tolist(), (0,)),
            (values[field][2:].tolist(), (1,)),
        ]
    assert calls[-1][0] == "mul_scale"
    assert calls[-1][1].tolist() == [1.25, 2.5]
    assert calls[-1][2] == ()


def test_native_mxfp4_loads_floating_sla_projection_in_same_stream():
    model = _native_model()
    model._serialized_mxfp4 = True
    model.blocks[0].attn.sla_module = nn.Module()
    model.blocks[0].attn.sla_module.proj_l = nn.Linear(1, 1, dtype=torch.bfloat16)
    calls: list[_WeightCall] = []
    values = {"weight": torch.ones(6, 2), "weight_scale": torch.ones(6, 1, dtype=torch.uint8)}
    for field, value in values.items():
        _target(model.blocks[0].attn.qkv_proj, field, value, calls)
    stream = [(f"blocks.0.attn.qkv_proj.{field}", value) for field, value in values.items()]
    for name, param in model.blocks[0].attn.sla_module.proj_l.named_parameters():
        stream.append((f"blocks.0.attn.sla_module.proj_l.{name}", torch.full_like(param, 0.25)))
    loaded = model.load_weights(stream)
    assert len(loaded) == 4
    for param in model.blocks[0].attn.sla_module.proj_l.parameters():
        assert param.dtype == torch.bfloat16
        torch.testing.assert_close(param, torch.full_like(param, 0.25))


@pytest.mark.parametrize("missing", ["weight", "weight_scale", "weight_dual_scale", "mul_scale"])
def test_serialized_mxfp4_rejects_incomplete_checkpoint(missing):
    model = _native_model()
    model._serialized_mxfp4 = True
    calls: list[_WeightCall] = []
    values = {
        "weight": torch.ones(6, 2),
        "weight_scale": torch.ones(6, 2, dtype=torch.uint8),
        "weight_dual_scale": torch.ones(6, 1, 1),
        "mul_scale": torch.ones(2),
    }
    for field, value in values.items():
        _target(model.blocks[0].attn.qkv_proj, field, value, calls)
    with pytest.raises(ValueError, match="MXFP4 checkpoint is missing weights"):
        model.load_weights(
            (f"blocks.0.attn.qkv_proj.{field}", value) for field, value in values.items() if field != missing
        )


@pytest.mark.parametrize(
    "method,flag", [("mxfp4", "is_checkpoint_mxfp4_serialized"), ("mxfp4_dualscale", "is_checkpoint_serialized")]
)
def test_resolve_serialized_mxfp4_uses_checkpoint_policy(method, flag):
    from vllm_omni.quantization import build_quant_config

    active = build_quant_config({"method": method, "ignored_layers": ["blocks.0.attn.out_proj"]})
    resolved = resolve_h3_transformer_quant_config(
        active,
        {"quant_method": method, flag: True, "ignored_layers": ["blocks.0.mlp.fc2"]},
        diffusers_weights=False,
        runtime_lora=False,
    )
    assert is_serialized_mxfp4(resolved)
    assert resolved is not None
    assert resolved.ignored_layers == ["blocks.0.mlp.fc2"]


@pytest.mark.parametrize(
    "diffusers,lora,message", [(True, False, "native H3 tensor names"), (False, True, "LoRA to be merged")]
)
def test_serialized_mxfp4_rejects_unsupported_combinations(diffusers, lora, message):
    with pytest.raises(ValueError, match=message):
        resolve_h3_transformer_quant_config(
            None,
            {"quant_method": "mxfp4", "is_checkpoint_mxfp4_serialized": True},
            diffusers_weights=diffusers,
            runtime_lora=lora,
        )


def test_mxfp4_mismatch_fails_before_loading():
    from vllm_omni.quantization import build_quant_config

    with pytest.raises(ValueError, match="active quantization config"):
        resolve_h3_transformer_quant_config(
            build_quant_config("mxfp4"),
            {"quant_method": "mxfp4_dualscale", "is_checkpoint_serialized": True},
            diffusers_weights=False,
            runtime_lora=False,
        )


def test_explicit_serialized_config_without_disk_metadata_still_rejects_lora():
    from vllm_omni.quantization import build_quant_config

    config = build_quant_config({"method": "mxfp4", "is_checkpoint_mxfp4_serialized": True})
    with pytest.raises(ValueError, match="LoRA to be merged"):
        resolve_h3_transformer_quant_config(config, None, diffusers_weights=False, runtime_lora=True)


def test_unrelated_checkpoint_format_is_unchanged():
    assert (
        resolve_h3_transformer_quant_config(
            None, {"quant_method": "modelopt"}, diffusers_weights=True, runtime_lora=False
        )
        is None
    )


def test_serialized_mxfp4_rejects_dynamic_adapter_before_file_access(mocker):
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline
    from vllm_omni.errors import OmniClientError

    pipeline = MiniMaxH3Pipeline.__new__(MiniMaxH3Pipeline)
    nn.Module.__init__(pipeline)
    pipeline.transformer = _native_model()
    pipeline.transformer._serialized_mxfp4 = True
    with pytest.raises(OmniClientError, match="LoRA to be merged"):
        pipeline._load_diffusion_lora_adapter(
            lora_request=mocker.Mock(), lora_path="/does-not-exist.safetensors", dtype=torch.bfloat16
        )


@pytest.mark.parametrize("metadata", [{"quant_scheme": "mxfp4_c7"}, {"use_torch_fake_quant": True}])
def test_training_only_metadata_cannot_silently_select_runtime_quantization(metadata):
    from vllm_omni.quantization import build_quant_config

    disk_config = {"quant_method": "mxfp4", **metadata}
    assert resolve_h3_transformer_quant_config(None, disk_config, diffusers_weights=False, runtime_lora=False) is None
    with pytest.raises(ValueError, match="training fake-quant metadata"):
        resolve_h3_transformer_quant_config(
            build_quant_config("mxfp4"), disk_config, diffusers_weights=False, runtime_lora=False
        )


@pytest.mark.parametrize("recipe", ["e2m1_rne", "mxfp4_c7"])
def test_explicit_fake_recipe_accepts_c7_training_metadata(recipe):
    from vllm_omni.quantization import build_quant_config

    config = build_quant_config(
        {"method": "mxfp4_fake", "recipe": recipe, "ignored_layers": ["token_refiner.blocks.0.attn.qkv_proj"]}
    )
    disk = {
        "quant_method": "mxfp4",
        "quant_scheme": "mxfp4_c7",
        "use_torch_fake_quant": True,
        "ignored_layers": ["token_refiner.blocks.0.attn.qkv_proj"],
    }
    resolved = resolve_h3_transformer_quant_config(config, disk, diffusers_weights=False, runtime_lora=False)
    assert resolved is config and resolved.recipe == recipe
    invalid_options: list[tuple[dict[str, object], bool, str]] = [
        (disk, True, "LoRA to be merged"),
        ({**disk, "is_checkpoint_mxfp4_serialized": True}, False, "floating weights"),
        ({**disk, "ignored_layers": []}, False, "explicitly match"),
        ({"quant_method": "fp8"}, False, "floating MXFP4"),
    ]
    for metadata, lora, message in invalid_options:
        with pytest.raises(ValueError, match=message):
            resolve_h3_transformer_quant_config(config, metadata, diffusers_weights=False, runtime_lora=lora)


def test_fake_reference_rejects_dynamic_lora_before_file_access(mocker):
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline
    from vllm_omni.errors import OmniClientError

    pipeline = MiniMaxH3Pipeline.__new__(MiniMaxH3Pipeline)
    nn.Module.__init__(pipeline)
    pipeline.transformer = nn.Module()
    pipeline.transformer._fake_mxfp4 = True
    with pytest.raises(OmniClientError, match="LoRA to be merged"):
        pipeline._load_diffusion_lora_adapter(
            lora_request=mocker.Mock(), lora_path="/does-not-exist.safetensors", dtype=torch.bfloat16
        )


def _merged_checkpoint(tmp_path, *, config=None, release=None):
    root = tmp_path / "merged" / "transformer"
    root.mkdir(parents=True)
    (root / "config.json").write_text(json.dumps(config or {}))
    # Only metadata resolution is under test; no weight loader reads this file.
    (root / "model.safetensors").touch()
    if release is None:
        release = {
            "partition": "fl2va",
            "tasks": ["t2va"],
            "base_schedule": [1, 0.7, 0.4, 0.15, 0],
            "sigma_shift_scales": {"video": 12, "audio": 3},
        }
    (root.parent / "model_index.json").write_text(json.dumps({"_minimax_h3": release}))
    return root


def _resolve_checkpoint(path, **kwargs):
    return resolve_h3_transformer_checkpoint(
        str(path), **{"partition": "fl2va", "modular": False, "runtime_lora": False, "base_config": {}, **kwargs}
    )


def test_merged_transformer_keeps_its_four_step_sampling_and_quant_metadata(tmp_path):
    path = _merged_checkpoint(tmp_path, config={"quantization_config": {"quant_method": "mxfp4"}})
    checkpoint = _resolve_checkpoint(path)
    assert checkpoint.path == path
    assert checkpoint.config["quantization_config"]["quant_method"] == "mxfp4"
    assert checkpoint.release["base_schedule"] == [1, 0.7, 0.4, 0.15, 0]
    assert checkpoint.release["tasks"] == ["t2va"]
    assert checkpoint.release["sigma_shift_scales"] == {"video": 12, "audio": 3}


@pytest.mark.parametrize("missing", ["config.json", "model.safetensors", "../model_index.json"])
def test_merged_transformer_requires_complete_packaging(tmp_path, missing):
    path = _merged_checkpoint(tmp_path)
    (path / missing).unlink()
    with pytest.raises(ValueError, match="requires config.json, safetensors weights"):
        _resolve_checkpoint(path)


@pytest.mark.parametrize(
    "options, message",
    [
        ({"partition": "combined"}, "single native H3 partition"),
        ({"modular": True}, "single native H3 partition"),
        ({"runtime_lora": True}, "merged adapters"),
        ({"base_config": {"hidden_size": 256}}, "architecture must match"),
        ({"base_config": {"rope_theta": 20000}}, "architecture must match"),
    ],
)
def test_merged_transformer_rejects_unsupported_or_incompatible_configuration(tmp_path, options, message):
    with pytest.raises(ValueError, match=message):
        _resolve_checkpoint(_merged_checkpoint(tmp_path), **options)


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"partition": "ref2va"}, "partition/tasks"),
        ({"tasks": ["d2d"]}, "partition/tasks"),
        ({"base_schedule": None}, "declare its distilled base_schedule"),
        ({"base_schedule": [1, 0.4, 0.7, 0]}, "strictly decreasing"),
        ({"sigma_shift_scales": {"video": float("nan"), "audio": 3}}, "finite positive"),
        ({"sigma_shift_scales": {"video": 12}}, "finite positive"),
        ({"sigma_shift_scales": {"video": 12, "audio": 0}}, "finite positive"),
    ],
)
def test_merged_transformer_rejects_invalid_release_contract(tmp_path, changes, message):
    path = _merged_checkpoint(tmp_path)
    index = json.loads((path.parent / "model_index.json").read_text())
    index["_minimax_h3"].update(changes)
    (path.parent / "model_index.json").write_text(json.dumps(index))
    with pytest.raises(ValueError, match=message):
        _resolve_checkpoint(path)


def test_merged_transformer_rejects_dynamic_adapter_before_file_access(mocker):
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline
    from vllm_omni.errors import OmniClientError

    pipeline = MiniMaxH3Pipeline.__new__(MiniMaxH3Pipeline)
    nn.Module.__init__(pipeline)
    pipeline._transformer_checkpoint = object()
    with pytest.raises(OmniClientError, match="merged adapters"):
        pipeline._load_diffusion_lora_adapter(
            lora_request=mocker.Mock(), lora_path="/does-not-exist.safetensors", dtype=torch.bfloat16
        )
