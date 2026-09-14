# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Post-fusion MXFP8 policy for the MiniMax-H3 Ultra projection inventory.

The arithmetic and compile boundaries live in diffusion.layers.mxfp8. This
adapter owns which model layers are converted and the H3 QK/V split geometry.
"""

from __future__ import annotations

import os

import torch
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod

from vllm_omni.diffusion.layers.mxfp8 import (
    _scale_numel,
    mxfp8_linear,
    mxfp8_quantize_project,
    mxfp8_quantize_swizzled,
    mxfp8_scaled_mm,
    silu_mxfp8_linear,
)
from vllm_omni.platforms import current_omni_platform

ENV = "VLLM_OMNI_H3_DIT_MXFP8"
SUFFIXES = ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2")
SHAPES = dict(zip(SUFFIXES, ((21504, 5376), (5376, 7168), (28672, 5376), (5376, 14336))))
_QK_WIDTH = 14336


def requested_mode() -> str | None:
    value = os.getenv(ENV, "0")
    if value not in ("0", "1"):
        raise ValueError(f"{ENV} must be 0 or 1")
    return "mxfp8" if value == "1" else None


class H3MXFP8Method(UnquantizedLinearMethod):
    def apply(self, layer, x, bias=None):
        if bias is not None:
            raise ValueError("H3 MXFP8 main projections must be bias-free")
        return mxfp8_linear(x, layer.weight, layer.mxfp8_weight_scale)


def quantize_layer(layer):
    """Convert one BF16 projection after the student adapter has been fused."""
    weight = layer.weight
    if (
        weight.dtype != torch.bfloat16
        or not weight.is_cuda
        or weight.ndim != 2
        or weight.shape[0] % 16
        or weight.shape[1] % 32
        or layer.bias is not None
        or type(layer.quant_method) is not UnquantizedLinearMethod
    ):
        raise ValueError("MXFP8 source must be a loaded CUDA BF16 unquantized projection")
    if hasattr(layer, "mxfp8_weight_scale"):
        raise ValueError("Duplicate MXFP8 weight quantization")
    q, scale = mxfp8_quantize_swizzled(weight.detach().contiguous())
    layer.weight = torch.nn.Parameter(q, requires_grad=False)
    layer.register_buffer("mxfp8_weight_scale", scale.view(torch.float8_e8m0fnu))
    layer.quant_method = H3MXFP8Method()
    return layer


def install_and_audit(model) -> None:
    if requested_mode() is None or getattr(model, "_h3_mxfp8_installed", False):
        return
    if not getattr(model, "_h3_student_fusion_complete", False):
        raise ValueError("MXFP8 conversion requires completed FastH3 student fusion")
    if getattr(model, "adaln_cache", None) is None or not model.adaln_cache._loaded:
        raise ValueError("H3 Ultra requires the loaded exact AdaLN cache")
    from vllm.distributed import get_tensor_model_parallel_world_size

    from vllm_omni.diffusion.distributed.parallel_state import get_sp_group

    if (
        len(model.blocks) != 50
        or current_omni_platform.get_device_capability(torch.accelerator.current_device_index()) != (12, 0)
        or get_tensor_model_parallel_world_size() != 1
        or get_sp_group().world_size != 8
    ):
        raise ValueError("H3 Ultra MXFP8 is qualified for the 50-block SM120 TP1/SP8 model")
    names = {f"blocks.{i}.{suffix}" for i in range(50) for suffix in SUFFIXES}
    modules = dict(model.named_modules())
    if not names <= modules.keys():
        raise ValueError("Incomplete H3 main projection inventory")
    # Validate the entire inventory before making any in-place conversion.
    for name, layer in modules.items():
        if not isinstance(layer, LinearBase):
            continue
        if type(layer.quant_method) is not UnquantizedLinearMethod:
            raise ValueError("H3 Ultra requires BF16 loading before MXFP8 conversion: " + name)
        if name in names and (
            tuple(layer.weight.shape) != SHAPES[name.split(".", 2)[2]] or layer.bias is not None or layer.tp_size != 1
        ):
            raise ValueError("Unsupported H3 projection shape or parallelism: " + name)
    for block in model.blocks:
        if block.attn.to_gate_compress is None or block.attn.to_gate_compress.weight.dtype != torch.bfloat16:
            raise ValueError("The VSA compression gate must remain BF16")
        if block.adaln_proj is not None:
            raise ValueError("H3 Ultra requires precomputed AdaLN")
    fusion = os.getenv("VLLM_OMNI_H3_SWIGLU_MXFP8_FUSION", "0")
    if fusion not in ("0", "1"):
        raise ValueError("VLLM_OMNI_H3_SWIGLU_MXFP8_FUSION must be 0 or 1")
    for name in sorted(names):
        quantize_layer(modules[name])
    for block in model.blocks:
        block.mlp._h3_swiglu_mxfp8_fused = fusion == "1"
    model._h3_mxfp8_installed = True


def fused_swiglu_fc2(layer, hidden):
    return silu_mxfp8_linear(hidden, layer.weight, layer.mxfp8_weight_scale)


def validate_split_projection(projection) -> None:
    if (
        type(projection.quant_method) is not H3MXFP8Method
        or projection.bias is not None
        or projection.tp_size != 1
        or tuple(projection.weight.shape) != (21504, 5376)
    ):
        raise ValueError("Split QK/V requires the TP1 H3 MXFP8 QKV projection")


def project_split_qk(projection, x):
    cut = _scale_numel(_QK_WIDTH, 5376)
    return mxfp8_quantize_project(x, projection.weight[:_QK_WIDTH], projection.mxfp8_weight_scale[:cut])


def project_split_v(projection, q, activation_scale, qk):
    validate_split_projection(projection)
    cut = _scale_numel(_QK_WIDTH, 5376)
    return mxfp8_scaled_mm(q, projection.weight[_QK_WIDTH:], activation_scale, projection.mxfp8_weight_scale[cut:])
