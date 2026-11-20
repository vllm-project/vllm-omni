# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import json

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from vllm_omni.diffusion.models.minimax_h3.checkpoint_quantization import (
    FastH3NVFP4Config,
    NVFP4Projection,
    _NVFP4Method,
    _Projection,
    fasth3_fp8_config,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def swizzle_reference(scales):
    rows, groups = scales.shape
    row = torch.arange(rows)[:, None]
    group = torch.arange(groups)[None, :]
    offsets = ((((row // 128) * (groups // 4) + group // 4) * 32 + row % 32) * 4 + row // 32 % 4) * 4 + group % 4
    packed = torch.empty_like(scales).flatten()
    packed[offsets] = scales
    return packed.reshape_as(scales)


@pytest.mark.parametrize("axis,swap", [(0, False), (0, True), (1, False)])
@pytest.mark.parametrize("rank", [0, 1])
def test_nvfp4_tp_preserves_packed_values_and_scale_layout(axis, swap, rank):
    rows, columns = 512, 256
    weight = torch.randint(0, 256, (rows, columns // 2), dtype=torch.uint8)
    scales = torch.randint(0, 127, (rows, columns // 16), dtype=torch.uint8)
    spec = NVFP4Projection("test", rows, columns, {"weight": "w", "scales": "s", "alpha": "a"}, swap)
    local = _Projection(spec, rows // 2 if axis == 0 else rows, columns // 2 if axis == 1 else columns, rank)
    local.weight_loader(local.weight, weight)
    local.weight_loader(local.scales, swizzle_reference(scales))
    expected_weight, expected_scales = weight, scales
    if swap:
        expected_weight = torch.cat(weight.chunk(2)[::-1])
        expected_scales = torch.cat(scales.chunk(2)[::-1])
        indices = torch.cat(
            (torch.arange(rank * 128, (rank + 1) * 128), torch.arange(256 + rank * 128, 256 + (rank + 1) * 128))
        )
        expected_weight, expected_scales = expected_weight[indices], expected_scales[indices]
    else:
        expected_weight, expected_scales = expected_weight.chunk(2, axis)[rank], expected_scales.chunk(2, axis)[rank]
    torch.testing.assert_close(local.weight, expected_weight)
    torch.testing.assert_close(local.scales, swizzle_reference(expected_scales))


def test_nvfp4_scalar_and_shape_validation():
    spec = NVFP4Projection("test", 128, 128, {"weight": "w", "scales": "s", "alpha": "a"}, reciprocal_scale=True)
    layer = _Projection(spec, 128, 128, 0)
    layer.weight_loader(layer.alpha, torch.tensor(4.0))
    assert layer.alpha.item() == 0.25
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="positive scalar"):
            layer.weight_loader(layer.alpha, torch.tensor(bad))
    with pytest.raises(ValueError, match="shape"):
        layer.weight_loader(layer.weight, torch.zeros(128, 128, dtype=torch.uint8))


def test_qad_dit_uses_every_exported_projection_and_keeps_qkv_scale_order(tmp_path, monkeypatch):
    tensors = {}
    for i, part in enumerate(("k", "q", "v"), 1):
        prefix = f"transformer_blocks.0.attn.to_{part}::"
        tensors.update(
            {
                prefix + "_nvfp4_weight": torch.full((256, 64), i, dtype=torch.uint8),
                prefix + "_nvfp4_weight_scale": torch.zeros(256, 8, dtype=torch.uint8),
                prefix + "_nvfp4_alpha": torch.tensor(float(i)),
                prefix + "_weight_global_sf": torch.tensor(1 / i, dtype=torch.bfloat16),
            }
        )
    path = tmp_path / "nvfp4_weights.safetensors"
    save_file(tensors, path)
    config = FastH3NVFP4Config.from_dit(path, rank=1)
    layer = nn.Module()
    method = _NVFP4Method(config.projections["blocks.0.attn.qkv_proj"], rank=1)
    method.create_weights(layer, 128, [128, 128, 128], 128, 768, torch.bfloat16)
    params = dict(layer.named_parameters())
    for name, value in tensors.items():
        parameter = params[config.weight_map[name].removeprefix("blocks.0.attn.qkv_proj.")]
        parameter.weight_loader(parameter, value)
    calls = []

    def gemm(x, weight, scales, alpha, input_scale):
        calls.append(alpha.item())
        return x.new_full((x.shape[0], weight.shape[0]), alpha.item())

    monkeypatch.setattr("vllm_omni.diffusion.models.minimax_h3.checkpoint_quantization._packed_nvfp4_linear", gemm)
    result = method.apply(layer, torch.ones(2, 128))
    assert calls == [2.0, 1.0, 3.0]
    assert result.shape == (2, 384)
    assert all(p.dtype == torch.uint8 for name, p in layer.named_parameters() if name.endswith(("weight", "scales")))
    del tensors["transformer_blocks.0.attn.to_q::_nvfp4_alpha"]
    save_file(tensors, path)
    with pytest.raises(ValueError, match="Incomplete"):
        FastH3NVFP4Config.from_dit(path, rank=0)


def test_qad_encoder_reads_its_own_shards_and_reciprocal_scales(tmp_path):
    config = {
        "quantization_config": {
            "quant_method": "nvfp4",
            "activation_scheme": "dynamic",
            "fmt": "e2m1",
            "group_size": 16,
            "scale_fmt": "e4m3",
            "scale_layout": "128x4",
        }
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    tensors = {}
    for part in ("v", "k", "q"):
        prefix = f"model.language_model.layers.0.self_attn.{part}_proj"
        tensors.update(
            {
                prefix + ".weight_packed": torch.zeros(128, 64, dtype=torch.uint8),
                prefix + ".weight_scale": torch.zeros(128, 8, dtype=torch.uint8),
                prefix + ".weight_global_scale": torch.tensor([2.0]),
            }
        )
    save_file(tensors, tmp_path / "model.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(tensors, "model.safetensors")})
    )
    plan = FastH3NVFP4Config.from_encoder(tmp_path, rank=0)
    specs = plan.projections["text_model.layers.0.self_attn.qkv_proj"]
    assert [p.source.rsplit(".", 1)[-1] for p in specs] == ["q_proj", "k_proj", "v_proj"]
    assert all(p.reciprocal_scale for p in specs)
    assert set(plan.weight_map) == set(tensors)


def test_fp8_scope_comes_from_weights_and_maps_to_native_fused_layers(tmp_path):
    tensors = {"transformer_blocks.0.norm1.weight": torch.ones(128, dtype=torch.bfloat16)}
    for part in ("q", "k", "v"):
        prefix = f"transformer_blocks.0.attn.to_{part}"
        tensors[prefix + ".weight"] = torch.ones(128, 128).to(torch.float8_e4m3fn)
        tensors[prefix + ".weight_scale"] = torch.ones(128)
    save_file(tensors, tmp_path / "weights.safetensors")
    (tmp_path / "diffusion_pytorch_model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(tensors, "weights.safetensors")})
    )
    config = fasth3_fp8_config(tmp_path)
    assert not config.is_layer_excluded("blocks.0.attn.qkv_proj")
    assert config.is_layer_excluded("blocks.0.attn.to_gate_compress")
    assert config.is_layer_excluded("token_refiner.blocks.0.attn.qkv_proj")
    del tensors["transformer_blocks.0.attn.to_q.weight_scale"]
    save_file(tensors, tmp_path / "weights.safetensors")
    (tmp_path / "diffusion_pytorch_model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(tensors, "weights.safetensors")})
    )
    with pytest.raises(ValueError, match="no per-channel scale"):
        fasth3_fp8_config(tmp_path)
