# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""H3 DiT and VAE weight-conversion policies over shared MXFP8 arithmetic.

DiT conversion follows student fusion; VAE conversion preserves its original
FP32 weight source and FP16 activation boundary. The policies stay distinct.
"""

from __future__ import annotations

import collections
import json
import os

import torch
from torch import nn
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

DIT_MXFP8_ENV = "VLLM_OMNI_H3_DIT_MXFP8"
SUFFIXES = ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2")
SHAPES = dict(zip(SUFFIXES, ((21504, 5376), (5376, 7168), (28672, 5376), (5376, 14336))))
_QK_WIDTH = 14336


def dit_mxfp8_mode() -> str | None:
    value = os.getenv(DIT_MXFP8_ENV, "0")
    if value not in ("0", "1"):
        raise ValueError(f"{DIT_MXFP8_ENV} must be 0 or 1")
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


def install_dit_mxfp8(model) -> None:
    if dit_mxfp8_mode() is None or getattr(model, "_h3_mxfp8_installed", False):
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


VAE_MXFP8_ENV = "VLLM_OMNI_H3_VAE_MXFP8"


def vae_mxfp8_enabled():
    value = os.environ.get(VAE_MXFP8_ENV, "0")
    if value not in ("0", "1"):
        raise ValueError(f"{VAE_MXFP8_ENV} must be 0 or 1")
    if value == "1" and os.environ.get("VLLM_OMNI_H3_VAE_INT8_CONVROT", "0") != "0":
        raise ValueError("Choose exactly one VAE quantization method")
    return value == "1"


class OnlineMXFP8Linear(nn.Module):
    def __init__(self, linear, name):
        super().__init__()
        from comfy_kitchen.backends.cuda import quantize_mxfp8
        from comfy_kitchen.backends.eager.quantization import quantize_mxfp8 as quantize_fp32_weight

        if linear.weight.dtype != torch.float32 or not linear.weight.is_cuda:
            raise RuntimeError("Online MXFP8 requires original FP32 weights on CUDA")
        self.in_features, self.out_features = linear.in_features, linear.out_features
        if self.in_features % 32 or self.out_features % 32:
            raise RuntimeError("Unexpected H3 VAE weight dimensions")
        self.name, self.calls = name, 0
        self._quantize = quantize_mxfp8
        with torch.no_grad(), torch.autocast("cuda", enabled=False):
            # The wheel's CUDA MX producer actually accepts only FP16/BF16.
            # Use its FP32-capable implementation once at load time so weights
            # are not silently rounded to FP16 before the intended quantizer.
            q, scale = quantize_fp32_weight(linear.weight.detach().contiguous(), pad_32x=False)
        assert q.dtype == torch.float8_e4m3fn and scale.dtype == torch.float8_e8m0fnu
        assert q.shape == linear.weight.shape
        self.register_buffer("weight", q)
        self.register_buffer("weight_scale", scale)
        self.register_buffer("bias", None if linear.bias is None else linear.bias.detach().to(torch.float16))

    def forward(self, x):
        if not x.is_cuda or x.shape[-1] != self.in_features:
            raise RuntimeError("Unsupported VAE MXFP8 input")
        if not torch.is_autocast_enabled("cuda") or torch.get_autocast_dtype("cuda") != torch.float16:
            raise RuntimeError("H3 VAE MXFP8 is qualified only under FP16 decode autocast")
        shape = x.shape
        with torch.autocast("cuda", enabled=False):
            x2d = x.to(torch.float16).reshape(-1, self.in_features).contiguous()
            rows = x2d.shape[0]
            q, scale = self._quantize(x2d, pad_32x=True)
            if q.dtype != torch.float8_e4m3fn or scale.dtype != torch.float8_e8m0fnu:
                raise RuntimeError("MXFP8 activation producer returned the wrong format")
            # Call the same native cuBLASLt block-scaled API as the DiT.
            # Unsupported dtype/shape/device raises; no dequantized fallback.
            out = mxfp8_scaled_mm(
                q,
                self.weight,
                scale,
                self.weight_scale,
                bias=self.bias,
                output_dtype=torch.float16,
                use_fast_accum=False,
            )
            if out.dtype != torch.float16 or out.shape != (q.shape[0], self.out_features):
                raise RuntimeError("MXFP8 GEMM returned the wrong output contract")
            self.calls += 1
            return out[:rows].reshape(*shape[:-1], self.out_features)


def install_vae_mxfp8(decoder):
    from importlib.metadata import version

    assert version("comfy-kitchen") == "0.2.33"
    assert len(decoder.transformer_blocks) == 36
    modules = []
    for i, block in enumerate(decoder.transformer_blocks):
        for owner, key, kind in (
            (block.attn, "to_qkv", "qkv"),
            (block.attn, "to_out", "out"),
            (block.ff, "w1", "fc1"),
            (block.ff, "w2", "fc2"),
        ):
            old = getattr(owner, key)
            assert isinstance(old, nn.Linear)
            name = f"transformer_blocks.{i}.{'attn' if kind in ('qkv', 'out') else 'ff'}.{key}"
            module = OnlineMXFP8Linear(old, name)
            module.kind = kind
            setattr(owner, key, module)
            modules.append(module)
    decoder._h3_online_mxfp8_modules = tuple(modules)
    return dict(
        rank=torch.distributed.get_rank() if torch.distributed.is_initialized() else 0,
        layers=144,
        group_size=32,
        weight_source="original_fp32",
        activation_dtype="float8_e4m3fn",
        scale_dtype="float8_e8m0fnu",
        output_dtype="float16",
        backend="torch_scaled_mm_cublaslt",
        weight_quantizer="comfy_kitchen_eager_fp32",
        activation_quantizer="comfy_kitchen_cuda",
        version="0.2.33",
        approximate=True,
        convrot=False,
    )


def audit_vae_mxfp8_calls(decoder, rank, logger):
    modules = getattr(decoder, "_h3_online_mxfp8_modules", ())
    if not modules:
        return
    counts = collections.Counter()
    for module in modules:
        expected = (74 if rank < 4 else 73) if module.kind == "out" else 21
        if module.calls != expected:
            raise RuntimeError(f"MXFP8 dispatch count mismatch: {module.name}: {module.calls} != {expected}")
        counts[module.kind] += module.calls
    logger.info(
        "H3_VAE_MXFP8_COMPLETE %s",
        json.dumps(
            dict(rank=rank, layers=144, calls=dict(counts), group_size=32, fallback=0, status="PASS"), sort_keys=True
        ),
    )
    for module in modules:
        module.calls = 0
