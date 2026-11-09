# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CANN MXFP8 GEMM, GELU and output quantization for the LingBot FFN."""

import torch


def fused_mxfp8_ffn(hidden, up, down):
    """Run the TP1 FFN with an FP8 intermediate, or decline unsupported layers."""
    if hidden.device.type != "npu" or hidden.dtype != torch.bfloat16 or hidden.numel() == 0:
        return None
    from vllm_omni.quantization.mxfp8_config import NPUMxfp8LinearMethod, NPUMxfp8OnlineLinearMethod

    supported = (NPUMxfp8LinearMethod, NPUMxfp8OnlineLinearMethod)
    if any(
        type(layer.quant_method) not in supported
        or layer.skip_bias_add
        or layer.return_bias
        or layer.weight.dtype != torch.float8_e4m3fn
        or layer.weight_scale.dtype not in (torch.uint8, torch.float8_e8m0fnu)
        or not layer.weight.is_contiguous()
        or not layer.weight_scale.is_contiguous()
        for layer in (up, down)
    ):
        return None
    try:
        from cann_ops_nn.ops.matmul.quant_matmul_activation_quant.quant_matmul_activation_quant import (
            quant_matmul_activation_quant,
        )
    except ImportError:
        return None

    up_method, down_method = up.quant_method, down.quant_method
    x, x_scale = up_method._quantize_activation(hidden.reshape(-1, hidden.shape[-1]))
    # torch_npu may store E8M0 scale bytes in uint8 tensors. The CANN
    # wrapper infers the ACL dtype from the tensor, so reinterpret the bytes.
    intermediate, intermediate_scale = quant_matmul_activation_quant(
        x,
        up.weight,
        up.weight_scale.view(torch.float8_e8m0fnu),
        x1_scale=x_scale.view(torch.float8_e8m0fnu),
        bias=up.bias.float() if up.bias is not None else None,
        group_sizes=[1, 1, 32],
        activation_type="gelu_tanh",
    )
    out = down_method._quant_matmul(intermediate, intermediate_scale, down, down.bias, hidden.dtype)
    return out.reshape(*hidden.shape[:-1], -1)
