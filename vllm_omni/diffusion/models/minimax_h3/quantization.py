# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Model-local CUDA MXFP8 scope; checkpoint fusion stays in the normal loader."""

import torch

from vllm_omni.platforms import current_omni_platform
from vllm_omni.quantization.mxfp8_config import DiffusionMXFP8Config


def projection_quant_config(config, prefix):
    """Keep conditioning, AdaLN, VSA gates and token refinement in BF16."""
    if not isinstance(config, DiffusionMXFP8Config) or not current_omni_platform.is_cuda():
        return config
    if prefix.startswith("blocks.") and prefix.endswith((".attn.qkv_proj", ".attn.out_proj", ".mlp.fc1", ".mlp.fc2")):
        return config
    return None


class VideoVAEMXFP8Linear(torch.nn.Module):
    """Online MXFP8 decoder projection; preserve FP32 checkpoint weights until quantization."""

    def __init__(self, linear, device):
        from vllm_omni.diffusion.layers.mxfp8 import mxfp8_quantize_swizzled

        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        home = linear.weight.device
        weight, scale = mxfp8_quantize_swizzled(linear.weight.detach().to(device).contiguous())
        self.register_buffer("weight", weight.to(home))
        self.register_buffer("weight_scale", scale.to(home))
        self.register_buffer("bias", None if linear.bias is None else linear.bias.detach().to(dtype=torch.float16))

    def forward(self, x):
        from vllm_omni.diffusion.layers.mxfp8 import mxfp8_quantize_swizzled, mxfp8_scaled_mm

        # H3 residual and norm state stays FP32. nn.Linear would autocast its
        # activation to FP16, so make the same boundary explicit here.
        if not torch.is_autocast_enabled("cuda") or torch.get_autocast_dtype("cuda") != torch.float16:
            raise RuntimeError("H3 video VAE MXFP8 requires FP16 CUDA autocast")
        activation, scale = mxfp8_quantize_swizzled(x.to(torch.float16).reshape(-1, self.in_features).contiguous())
        output = mxfp8_scaled_mm(activation, self.weight, scale, self.weight_scale, output_dtype=torch.float16)
        output = output.reshape(*x.shape[:-1], self.out_features)
        return output if self.bias is None else output + self.bias


def quantize_video_vae_decoder(decoder, config, device):
    """Convert only the four decoder-block projections, after exact-op installation."""
    from vllm.model_executor.layers.quantization.utils.quant_utils import is_layer_skipped

    from vllm_omni.diffusion.models.minimax_h3.ops.vae import _decoder_block_linears

    if not isinstance(config, DiffusionMXFP8Config) or config.is_checkpoint_mxfp8_serialized:
        raise ValueError("H3 video VAE supports online MXFP8 only")
    capability = current_omni_platform.get_device_capability()
    if device.type != "cuda" or capability is None or capability.major not in (10, 12):
        raise ValueError("H3 video VAE MXFP8 requires Blackwell CUDA")
    if not hasattr(torch.nn.functional, "scaled_mm"):
        raise RuntimeError("H3 video VAE MXFP8 requires PyTorch block-scaled scaled_mm")
    linears = _decoder_block_linears(decoder)
    if linears is None or any(
        x.weight.dtype != torch.float32 or x.in_features % 32 or x.out_features % 16 for x in linears
    ):
        raise ValueError("unsupported H3 video VAE decoder projection contract")
    count = 0
    for index, block in enumerate(decoder.transformer_blocks):
        for parent_name, name in (("attn", "to_qkv"), ("attn", "to_out"), ("ff", "w1"), ("ff", "w2")):
            parent = getattr(block, parent_name)
            prefix = f"video_vae.decoder.transformer_blocks.{index}.{parent_name}.{name}"
            linear = getattr(parent, name)
            if is_layer_skipped(prefix, config.ignored_layers):
                linear.to(dtype=torch.float16)
                continue
            setattr(parent, name, VideoVAEMXFP8Linear(linear, device))
            count += 1
    return count
