# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU coverage for the opt-in recipe, load lifecycle and rotation algebra."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import NoReturn

import pytest
import torch
from torch import nn
from vllm.model_executor.kernels.linear.scaled_mm.pytorch import RowWiseTorchFP8ScaledMMLinearKernel
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization.online.fp8 import Fp8PtpcOnlineLinearMethod
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape

from vllm_omni.diffusion.data import DiffusionParallelConfig, OmniDiffusionConfig
from vllm_omni.diffusion.models.minimax_h3 import hadamard_fp8 as h64
from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3_image as h3i

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@dataclass
class _Quantizer:
    static: bool = False
    group_shape: GroupShape = field(default_factory=lambda: GroupShape(1, -1))
    is_group_quant: bool = False
    use_aiter: bool = True


def _config() -> OmniDiffusionConfig:
    # Only exercise validation, with no model download or worker construction.
    config = object.__new__(OmniDiffusionConfig)
    config.parallel_config = DiffusionParallelConfig(data_parallel_size=1)
    config.quantization_config = None
    return config


def _toy_model() -> nn.Module:
    dit = nn.Module()
    dit.blocks = nn.ModuleList()
    for _ in range(50):
        block = nn.Module()
        block.mlp = nn.Module()
        for side in (1, 2):
            layer = nn.Module()
            layer.tp_size = 1
            layer.weight = nn.Parameter(torch.arange(256).reshape(4, 64).remainder(17).sub(8).bfloat16() / 16)
            # Keep the real contract classes, but do not initialize a GPU kernel.
            method = object.__new__(Fp8PtpcOnlineLinearMethod)
            method.fp8_linear = object.__new__(RowWiseTorchFP8ScaledMMLinearKernel)
            method.fp8_linear.quant_fp8 = _Quantizer()
            layer.quant_method = method
            setattr(block.mlp, f"fc{side}", layer)
        dit.blocks.append(block)
    return dit


def test_signed_rotation_is_deterministic_layer_specific_and_orthogonal():
    signs = h64._signs("blocks.0.mlp.fc2", 128)
    assert torch.equal(signs, h64._signs("blocks.0.mlp.fc2", 128))
    assert not torch.equal(signs, h64._signs("blocks.1.mlp.fc2", 128))
    assert set(signs.tolist()) == {-1.0, 1.0}
    x = torch.arange(3 * 128).reshape(3, 128).remainder(9).sub(4).bfloat16()
    w = torch.arange(5 * 128).reshape(5, 128).remainder(7).sub(3).bfloat16()
    xr, wr = h64._rotate_rows(x, signs), h64._rotate_rows(w, signs)
    assert xr.dtype == wr.dtype == torch.float32
    assert torch.equal(x.double() @ w.double().t(), xr.double() @ wr.double().t())
    assert torch.equal(x.float().square().sum(-1), xr.square().sum(-1))


@pytest.mark.parametrize("shape", [(2, 63), (2, 65), (2, 2, 64)])
def test_rotation_rejects_invalid_shapes(shape):
    with pytest.raises(ValueError, match="divisible by 64"):
        h64._rotate_rows(torch.ones(shape), torch.ones(64))


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    [
        ("enable_cpu_offload", True, "resident"),
        ("enable_layerwise_offload", True, "resident"),
        ("enable_distributed_layerwise_offload", True, "resident"),
        ("diffusion_offload_config", {"mode": "module", "components": ["text_encoder"]}, "resident"),
        ("enable_sleep_mode", True, "resident"),
        ("lora_path", "adapter", "LoRA"),
        ("lora_application", "merged", "LoRA"),
        ("weight_transfer_config", {"backend": "omni_lora_ipc"}, "weight updates"),
    ],
)
def test_config_rejects_unsupported_weight_lifecycles(monkeypatch, field_name, value, message):
    monkeypatch.setattr(h64, "_validate_device", lambda device: None)
    config = _config()
    setattr(config, field_name, value)
    with pytest.raises(ValueError, match=message):
        h64._validate_config(config, torch.device("cuda", 0))


@pytest.mark.parametrize("field_name", ["tensor_parallel_size", "sequence_parallel_size", "pipeline_parallel_size"])
def test_config_rejects_parallel_recipes(monkeypatch, field_name):
    monkeypatch.setattr(h64, "_validate_device", lambda device: None)
    config = _config()
    setattr(config.parallel_config, field_name, 2)
    with pytest.raises(ValueError, match="TP1/SP1"):
        h64._validate_config(config, torch.device("cuda", 0))


def test_native_load_lifecycle_retains_fp8_gemm_and_fc1(monkeypatch):
    from vllm.model_executor.layers.quantization.online.fp8 import _fp8_channel_scale, _fp8_quant_per_channel
    from vllm.model_executor.utils import replace_parameter

    calls = []

    def original(method, layer):
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return
        calls.append(layer)
        scale = _fp8_channel_scale(layer.weight.float().abs().amax(-1, keepdim=True))
        replace_parameter(layer, "weight", _fp8_quant_per_channel(layer.weight, scale).t())
        replace_parameter(layer, "weight_scale", scale)
        layer._already_called_process_weights_after_loading = True

    monkeypatch.setattr(h64, "_validate_config", lambda config, device: None)
    monkeypatch.setattr(h64, "_FC2_SHAPE", (4, 64))
    original_signs = h64._signs
    monkeypatch.setattr(h64, "_signs", lambda name, width=64: original_signs(name, width))
    monkeypatch.setattr(Fp8PtpcOnlineLinearMethod, "process_weights_after_loading", original)
    dit = _toy_model()
    fc1 = dit.blocks[0].mlp.fc1
    fc2 = dit.blocks[0].mlp.fc2
    source = fc2.weight.detach().clone()
    original_kernel = fc2.quant_method.fp8_linear
    h64.install_fc2_hadamard_fp8(dit, od_config=_config(), device=torch.device("cuda", 0))
    fc2.quant_method.process_weights_after_loading(fc2)
    rotated = h64._rotate_rows(source, h64._signs("blocks.0.mlp.fc2", 64))
    expected_scale = _fp8_channel_scale(rotated.abs().amax(-1, keepdim=True))
    expected_weight = _fp8_quant_per_channel(rotated, expected_scale)
    assert torch.equal(fc2.weight.view(torch.uint8), expected_weight.t().view(torch.uint8))
    assert torch.equal(fc2.weight_scale, expected_scale)
    assert fc2.quant_method.fp8_linear is original_kernel
    assert isinstance(original_kernel.quant_fp8, h64.Hadamard64QuantFP8)
    assert isinstance(fc1.quant_method.fp8_linear.quant_fp8, _Quantizer)
    assert not fc2._forward_pre_hooks
    saved_weight, saved_quantizer = fc2.weight, original_kernel.quant_fp8
    fc2.quant_method.process_weights_after_loading(fc2)
    assert fc2.weight is saved_weight and original_kernel.quant_fp8 is saved_quantizer
    assert calls == [fc2]
    with pytest.raises(ValueError, match="already installed"):
        h64.install_fc2_hadamard_fp8(dit, od_config=_config(), device=torch.device("cuda", 0))


def test_recipe_rejects_missing_mlp_and_quantized_attention(monkeypatch):
    monkeypatch.setattr(h64, "_validate_config", lambda config, device: None)
    dit = _toy_model()
    del dit.blocks[0].mlp.fc1.quant_method
    with pytest.raises(ValueError, match="all 100"):
        h64.install_fc2_hadamard_fp8(dit, od_config=_config(), device=torch.device("cuda", 0))
    dit = _toy_model()
    attention = object.__new__(LinearBase)
    nn.Module.__init__(attention)
    attention.quant_method = dit.blocks[0].mlp.fc1.quant_method
    dit.quantized_attention = attention
    with pytest.raises(ValueError, match="unquantized transformer layer: quantized_attention"):
        h64.install_fc2_hadamard_fp8(dit, od_config=_config(), device=torch.device("cuda", 0))


def test_dynamic_lora_is_rejected_only_in_h64_mode():
    from vllm_omni.lora.request import LoRARequest

    pipeline = object.__new__(h3i.MiniMaxH3ImagePipeline)
    nn.Module.__init__(pipeline)
    request = LoRARequest(lora_name="unused", lora_int_id=1, lora_path="unused")
    pipeline._mlp_fp8_hadamard = False
    assert pipeline._load_diffusion_lora_adapter(lora_request=request, lora_path="unused", dtype=torch.bfloat16) is None
    pipeline._mlp_fp8_hadamard = True
    with pytest.raises(ValueError, match="does not support LoRA"):
        pipeline._load_diffusion_lora_adapter(lora_request=request, lora_path="unused", dtype=torch.bfloat16)


@pytest.mark.parametrize("component", ["text_encoder", "vae"])
def test_config_requires_other_components_unquantized(monkeypatch, component):
    from vllm_omni.quantization import build_quant_config

    monkeypatch.setattr(h64, "_validate_device", lambda device: None)
    config = _config()
    config.quantization_config = build_quant_config({"default": None, component: {"method": "fp8"}})
    with pytest.raises(ValueError, match=f"quantization disabled for {component}"):
        h64._validate_config(config, torch.device("cuda", 0))


class _StopAfterTransformerError(Exception):
    pass


def _pipeline_config(tmp_path: Path, additional: dict[str, object]) -> OmniDiffusionConfig:
    (tmp_path / "model_index.json").write_text(
        '{"_class_name":"MiniMaxH3ImagePipeline","_minimax_h3":'
        '{"partition":"image","tasks":["t2i"],"pdd_fused_heads":"heads.safetensors"}}'
    )
    (tmp_path / "pdd_config.json").write_text('{"num_inference_steps":1,"sigmas":[1.0,0.0]}')
    config = _config()
    config.model = str(tmp_path)
    config.revision = None
    config.additional_config = additional
    return config


@pytest.mark.parametrize(
    "additional,expected_calls",
    [({}, 0), ({"minimax_h3_mlp_fp8_hadamard": False}, 0), ({"minimax_h3_mlp_fp8_hadamard": True}, 1)],
)
def test_pipeline_default_preserves_bf16_and_explicit_true_installs(monkeypatch, tmp_path, additional, expected_calls):
    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_blocks import MiniMaxH3DiTArchConfig

    calls = []

    class Transformer(nn.Module):
        def __init__(self, od_config: OmniDiffusionConfig, quant_config: object) -> None:
            super().__init__()
            self.arch = MiniMaxH3DiTArchConfig(audio_enabled=False)
            assert quant_config is None

    def install(dit: nn.Module, *, od_config: OmniDiffusionConfig, device: torch.device) -> None:
        calls.append(dit)
        assert od_config.additional_config["minimax_h3_mlp_fp8_hadamard"] is True
        assert device == torch.device("cpu")

    def stop(root: str, *, subfolder: str, local_files_only: bool) -> NoReturn:
        raise _StopAfterTransformerError

    monkeypatch.setattr(h3i, "get_local_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(h3i, "MiniMaxH3DiTModel", Transformer)
    monkeypatch.setattr(h64, "install_fc2_hadamard_fp8", install)
    monkeypatch.setattr(h3i.Qwen2TokenizerFast, "from_pretrained", stop)
    config = _pipeline_config(tmp_path, additional)
    with pytest.raises(_StopAfterTransformerError):
        h3i.MiniMaxH3ImagePipeline(od_config=config)
    assert len(calls) == expected_calls


@pytest.mark.parametrize("invalid", [None, "true", 1, []])
def test_pipeline_hadamard_option_requires_a_real_boolean(tmp_path, monkeypatch, invalid):
    monkeypatch.setattr(h3i, "get_local_device", lambda: torch.device("cpu"))
    config = _pipeline_config(tmp_path, {"minimax_h3_mlp_fp8_hadamard": invalid})
    with pytest.raises(ValueError, match="must be a bool"):
        h3i.MiniMaxH3ImagePipeline(od_config=config)
