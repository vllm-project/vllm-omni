# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Native safetensors through real Wan name mapping and vLLM TP loaders."""

import sys
from types import SimpleNamespace

import pytest
import torch
from vllm.config.load import LoadConfig
from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear, UnquantizedLinearMethod

from tests.diffusion.quantization.test_mxfp4_single_checkpoint_conversion import _edit, _fixture
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.model_loader.checkpoint_adapters.mxfp4_native import prepare_native_mxfp4
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.wan2_2.wan2_2_transformer import WanTransformer3DModel
from vllm_omni.quantization.mxfp4_config import DiffusionMXFP4Config

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.fixture(autouse=True)
def _tp(monkeypatch):
    monkeypatch.setattr("vllm.model_executor.parameter.get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr("vllm.model_executor.parameter.get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr("vllm_omni.platforms.current_omni_platform.is_npu", lambda: True)
    monkeypatch.setattr(
        "vllm_omni.diffusion.models.wan2_2.wan2_2_transformer.get_tensor_model_parallel_rank", lambda: 0
    )
    monkeypatch.setattr(
        "vllm_omni.diffusion.models.wan2_2.wan2_2_transformer.get_tensor_model_parallel_world_size", lambda: 1
    )


class _TinyWan(torch.nn.Module):
    load_weights = WanTransformer3DModel.load_weights

    def __init__(self, config, tp_size=1):
        super().__init__()
        self._native_mxfp4_checkpoint = config.native_checkpoint
        blocks = []
        for i in range(2):
            block = torch.nn.Module()
            block.attn1 = torch.nn.Module()
            block.attn1.to_qkv = QKVParallelLinear(
                hidden_size=64,
                head_size=2,
                total_num_heads=2,
                bias=i == 0,
                params_dtype=torch.bfloat16,
                quant_config=config,
                prefix=f"blocks.{i}.attn1.to_qkv",
                disable_tp=tp_size == 1,
            )
            blocks.append(block)
        self.blocks = torch.nn.ModuleList(blocks)


@pytest.mark.parametrize("scale_alg", [0, 2])
@pytest.mark.parametrize("smooth", [False, True])
@pytest.mark.parametrize("tp_size,rank", [(1, 0), (2, 0), (2, 1)])
def test_native_experts_stream_packed_qkv_and_actual_float(tmp_path, scale_alg, smooth, tp_size, rank, monkeypatch):
    monkeypatch.setattr("vllm.model_executor.layers.linear.get_tensor_model_parallel_rank", lambda: rank)
    monkeypatch.setattr("vllm.model_executor.layers.linear.get_tensor_model_parallel_world_size", lambda: tp_size)
    original, quant, output = _fixture(tmp_path, smooth)
    active = DiffusionMXFP4Config(
        native_checkpoint_path=str(quant),
        mxfp4_scale_alg=scale_alg,
        w4a8_fallback_steps=[0],
        w4a8_fallback_layers=["blocks.1.attn1.to_qkv"],
    )
    configs = [prepare_native_mxfp4(active, str(original), name) for name in ("transformer", "transformer_2")]
    assert active.native_checkpoint is None and active.ignored_layers == []
    assert configs[0].native_checkpoint is not configs[1].native_checkpoint
    pipeline = torch.nn.Module()
    loader = DiffusersPipelineLoader(LoadConfig(), OmniDiffusionConfig(model=str(original), quantization_config=active))
    monkeypatch.setattr(loader, "_prepare_weights", lambda *a: pytest.fail("must use native source plan"))
    for i, (component, config) in enumerate(zip(("transformer", "transformer_2"), configs)):
        model = _TinyWan(config, tp_size)
        setattr(pipeline, component, model)
        source = loader.ComponentSource(str(original), component, None, component + ".")
        items = list(loader._get_weights_iterator(source, model=pipeline))
        # Only load the two fixture QKV layers; all mapped tensors remain available.
        weights = [
            (name.removeprefix(component + "."), value)
            for name, value in items
            if ".attn1.to_" in name and any(f".to_{part}." in name for part in "qkv")
        ]
        loaded = model.load_weights(iter(weights))
        layer = model.blocks[0].attn1.to_qkv
        float_layer = model.blocks[1].attn1.to_qkv
        assert "blocks.0.attn1.to_qkv.weight" in loaded
        assert layer.weight.dtype == torch.uint8 and layer.weight.shape == (12 // tp_size, 32)
        expected = dict(weights)
        torch.testing.assert_close(
            layer.weight, torch.cat([expected[f"blocks.0.attn1.to_{p}.weight"].chunk(tp_size)[rank] for p in "qkv"])
        )
        torch.testing.assert_close(
            layer.weight_scale,
            torch.cat([expected[f"blocks.0.attn1.to_{p}.weight_scale"].chunk(tp_size)[rank] for p in "qkv"]),
        )
        assert isinstance(float_layer.quant_method, UnquantizedLinearMethod)
        torch.testing.assert_close(float_layer.weight, torch.full((12 // tp_size, 64), i + 3, dtype=torch.bfloat16))
        assert config.mxfp4_scale_alg == scale_alg
        assert config.require_smooth_scale == smooth
        pointer = layer.weight.data_ptr()
        monkeypatch.setitem(
            sys.modules,
            "torch_npu",
            SimpleNamespace(
                float4_e2m1fn_x2="fp4",
                float8_e8m0fnu=torch.uint8,
                npu_dtype_cast=lambda *a: pytest.fail("packed bytes must not be numerically cast"),
                npu_dynamic_mx_quant=lambda *a, **kw: pytest.fail("offline W4 must not be requantized"),
            ),
        )
        layer.quant_method.process_weights_after_loading(layer)
        assert layer.weight.data_ptr() == pointer
    assert not output.exists()


@pytest.mark.parametrize("rank", [0, 1])
def test_native_row_parallel_slices_packed_weight_scale_and_smooth(tmp_path, rank, monkeypatch):
    monkeypatch.setattr("vllm.model_executor.layers.linear.get_tensor_model_parallel_rank", lambda: rank)
    monkeypatch.setattr("vllm.model_executor.layers.linear.get_tensor_model_parallel_world_size", lambda: 2)
    original, quant, _ = _fixture(tmp_path, smooth=True)
    config = prepare_native_mxfp4(DiffusionMXFP4Config(native_checkpoint_path=str(quant)), str(original), "transformer")
    layer = RowParallelLinear(
        64, 4, bias=False, params_dtype=torch.bfloat16, quant_config=config, prefix="blocks.0.ffn.net_2"
    )
    assert layer.weight.shape == (4, 16)
    assert layer.weight_scale.shape == (4, 1)
    assert layer.mul_scale.shape == (32,)
    source = dict(config.native_checkpoint.weights())
    for name, size in (("weight", 16), ("weight_scale", 1), ("mul_scale", 32)):
        param = getattr(layer, name)
        full = source["blocks.0.ffn.net.2." + name]
        param.weight_loader(param, full)
        dim = 0 if name == "mul_scale" else 1
        torch.testing.assert_close(param, full.narrow(dim, rank * size, size), rtol=0, atol=0)


@pytest.mark.parametrize("ignored", ["blocks.0.attn1.to_qkv", "blocks.0.ffn.net_2"])
def test_native_rejects_ignore_of_packed_weight(tmp_path, ignored):
    original, quant, _ = _fixture(tmp_path)
    with pytest.raises(ValueError, match="not BF16"):
        prepare_native_mxfp4(
            DiffusionMXFP4Config(native_checkpoint_path=str(quant), ignored_layers=[ignored]),
            str(original),
            "transformer",
        )


def test_native_rejects_missing_quant_tensor_without_float_repair(tmp_path):
    original, quant, _ = _fixture(tmp_path)
    _edit((original, quant, None), lambda raw, desc: raw.pop("blocks.0.ffn.2.weight"))
    with pytest.raises(ValueError, match="禁止 BF16"):
        prepare_native_mxfp4(DiffusionMXFP4Config(native_checkpoint_path=str(quant)), str(original), "transformer_2")


def test_native_config_binds_before_transformer_allocation(tmp_path, monkeypatch):
    from vllm_omni.diffusion.models.wan2_2 import pipeline_wan2_2
    from vllm_omni.quantization import build_quant_config

    original, quant, _ = _fixture(tmp_path, smooth=True)
    active = build_quant_config(
        {
            "transformer": {"method": "mxfp4", "native_checkpoint_path": str(quant), "mxfp4_scale_alg": 2},
            "transformer_2": {"method": "mxfp4", "native_checkpoint_path": str(quant)},
        }
    )
    captured = []

    def construct(**kwargs):
        config = kwargs["quant_config"]
        assert config.native_checkpoint is not None
        assert config.is_checkpoint_mxfp4_serialized and config.require_smooth_scale
        captured.append(config)
        return torch.nn.Module()

    monkeypatch.setattr(pipeline_wan2_2, "WanTransformer3DModel", construct)
    for component in ("transformer", "transformer_2"):
        model = pipeline_wan2_2.create_transformer_from_config(
            {}, active, component=component, original_model_path=str(original)
        )
        assert model._native_mxfp4_checkpoint is captured[-1].native_checkpoint
    assert [config.mxfp4_scale_alg for config in captured] == [2, 0]


def test_native_runtime_rejects_sharded_export(tmp_path):
    from safetensors.torch import save_file

    original, quant, _ = _fixture(tmp_path)
    save_file({"extra": torch.zeros(1)}, str(quant / "high_noise_model" / "extra.safetensors"))
    with pytest.raises(ValueError, match="one safetensors"):
        prepare_native_mxfp4(DiffusionMXFP4Config(native_checkpoint_path=str(quant)), str(original), "transformer")


@pytest.mark.parametrize("native", ["head.head", "time_embedding.0"])
def test_native_rejects_quantized_root_float_only_modules(tmp_path, native):
    original, quant, _ = _fixture(tmp_path)

    def quantize_root(raw, desc):
        raw[native + ".weight"] = torch.ones((4, 32), dtype=torch.uint8)
        raw[native + ".weight_scale"] = torch.full((4, 2), 127, dtype=torch.uint8)
        desc[native + ".weight"] = desc[native + ".weight_scale"] = "W4A4_MXFP4"

    _edit((original, quant, None), quantize_root)
    with pytest.raises(ValueError, match="not supported by the Wan Linear method"):
        prepare_native_mxfp4(DiffusionMXFP4Config(native_checkpoint_path=str(quant)), str(original), "transformer_2")


def test_native_path_cannot_be_lost_when_base_checkpoint_declares_quantization():
    from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2 import resolve_wan_transformer_quant_config

    config = DiffusionMXFP4Config(native_checkpoint_path="/quantized/native")
    with pytest.raises(ValueError, match="unquantized BF16"):
        resolve_wan_transformer_quant_config(
            {
                "quantization_config": {
                    "quant_method": "mxfp4",
                    "is_checkpoint_mxfp4_serialized": True,
                    "ignored_layers": ["proj_out"],
                }
            },
            config,
            "transformer",
        )
