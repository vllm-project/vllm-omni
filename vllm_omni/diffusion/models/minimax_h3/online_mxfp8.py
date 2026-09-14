# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Online FP32 -> MXFP8 E4M3/E8M0 for the same 144 VAE decoder linears.

FP16 activation boundaries, elementwise operators, and scheduling match the
current-best VAE and the INT8 ConvRot arm. No rotation or FP16 GEMM fallback.
"""

import collections
import json
import os

import torch
from torch import nn

from vllm_omni.diffusion.layers.mxfp8 import mxfp8_scaled_mm

FLAG = "VLLM_OMNI_H3_VAE_MXFP8"


def enabled():
    value = os.environ.get(FLAG, "0")
    if value not in ("0", "1"):
        raise ValueError(f"{FLAG} must be 0 or 1")
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


def install(decoder):
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


def complete(decoder, rank, logger):
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
