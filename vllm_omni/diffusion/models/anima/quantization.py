# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import torch
from torch import nn
from vllm.model_executor.layers.linear import ReplicatedLinear, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.utils.torch_utils import set_default_torch_dtype

from vllm_omni.diffusion.models.anima.anima_transformer import AnimaTransformer3DModel


def prepare_anima_transformer_fp8(transformer: AnimaTransformer3DModel, config: Fp8Config) -> int:
    """Quantize attention and feed-forward projections after loading BF16 weights."""
    blocks = transformer.transformer_blocks
    names = [
        name
        for name, layer in blocks.named_modules()
        if isinstance(layer, nn.Linear) and name.split(".")[1] in {"attn1", "attn2", "ff", "before_proj", "after_proj"}
    ]
    replaced = 0
    for name in names:
        layer = blocks.get_submodule(name)
        dtype = layer.weight.dtype
        if dtype not in (torch.bfloat16, torch.float16):
            raise ValueError("Anima online FP8 requires BF16 or FP16 transformer weights.")
        with torch.device(layer.weight.device), set_default_torch_dtype(dtype):
            replacement = ReplicatedLinear(
                layer.in_features,
                layer.out_features,
                bias=layer.bias is not None,
                params_dtype=dtype,
                quant_config=config,
                prefix=f"transformer.transformer_blocks.{name}",
                return_bias=False,
                disable_tp=True,
            )
        if isinstance(replacement.quant_method, UnquantizedLinearMethod):
            continue
        with torch.no_grad():
            replacement.weight.weight_loader(replacement.weight, layer.weight)
            if layer.bias is not None:
                replacement.bias.weight_loader(replacement.bias, layer.bias)
        replacement.to(layer.weight.device)
        replacement.train(layer.training)
        parent, _, child = name.rpartition(".")
        setattr(blocks.get_submodule(parent), child, replacement)
        replaced += 1
    return replaced
