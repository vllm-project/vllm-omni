# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import types

import torch
from torch import nn


class _W8A8SwiGLU(nn.Module):
    """Quantized FFN expansion with native FP32 bias before the FP16 SwiGLU."""

    def __init__(self, linear: nn.Linear) -> None:
        super().__init__()
        weight = linear.weight.detach().float()
        scale = weight.abs().amax(dim=1) / 127
        scale = torch.where(scale == 0, 1.0, scale)
        self.register_buffer("weight", (weight / scale[:, None]).round().clamp(-127, 127).to(torch.int8).T.contiguous())
        self.register_buffer("scale", scale)
        self.register_buffer("bias", linear.bias.detach().float())
        self.output_features = linear.out_features // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        import torch_npu

        if torch_npu.get_npu_format(self.weight) != 29:
            # CPU residency stages ND weights. Keep the buffer object held by the stager.
            self.weight.data = torch_npu.npu_format_cast(self.weight.clone(), 29)
        quantized, token_scale = torch_npu.npu_dynamic_quant(x.reshape(-1, x.shape[-1]).to(torch.float16).contiguous())
        output = torch.ops.mindiesd.eagle_quant_batch_matmul_swiglu_vae(
            quantized, self.weight, self.scale, None, self.bias, token_scale, torch.float16
        )
        return output.reshape(*x.shape[:-1], self.output_features)


def _forward_w8a8_swiglu(self, hidden_states: torch.Tensor) -> torch.Tensor:
    return self.w2(self.w1(hidden_states))


def install_video_vae_w8a8_swiglu(decoder: nn.Module) -> None:
    # The VAE-specific SDK entry accepts bias and retains the intermediate FP16 rounding.
    from mindiesd.layers import _custom_ops  # noqa: F401

    if not hasattr(torch.ops.mindiesd, "eagle_quant_batch_matmul_swiglu_vae"):
        raise RuntimeError("Video VAE W8A8 SwiGLU requires the MindIE-SD VAE fused operator")

    installed = 0
    for block in getattr(decoder, "transformer_blocks", ()):
        ff = block.ff
        if ff.use_gated and isinstance(ff.act_fn, nn.SiLU) and isinstance(ff.w1, nn.Linear) and ff.w1.bias is not None:
            ff.w1 = _W8A8SwiGLU(ff.w1)
            ff._forward_impl = types.MethodType(_forward_w8a8_swiglu, ff)
            installed += 1
    if not installed:
        raise ValueError("Video VAE W8A8 SwiGLU requires a native gated SiLU FFN with a bias")
