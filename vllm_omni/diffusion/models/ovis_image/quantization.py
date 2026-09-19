# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import torch
from torch import nn
from transformers import Qwen3Model
from vllm.model_executor.layers.linear import ReplicatedLinear, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.utils.torch_utils import set_default_torch_dtype

from vllm_omni.quantization import ComponentQuantizationConfig


def prepare_ovis_text_encoder_fp8(
    encoder: Qwen3Model, quant_config: QuantizationConfig | None, device: torch.device
) -> int:
    """Prepare Ovis Qwen3 decoder projections for online FP8 loading."""
    if not isinstance(quant_config, ComponentQuantizationConfig):
        return 0
    config = quant_config.component_configs.get("text_encoder")
    if config is None:
        return 0
    if (
        not isinstance(config, Fp8Config)
        or config.is_checkpoint_fp8_serialized
        or config.activation_scheme != "dynamic"
    ):
        raise ValueError("Ovis text_encoder supports dynamic online FP8 from an unquantized checkpoint only.")
    layers = encoder.layers
    replaced = 0
    for name, layer in list(layers.named_modules()):
        if not isinstance(layer, nn.Linear):
            continue
        dtype = layer.weight.dtype
        if dtype not in (torch.bfloat16, torch.float16):
            raise ValueError("Ovis text_encoder FP8 requires BF16 or FP16 weights.")
        # Native online loaders may quantize during weight_loader, including
        # when transformers loaded the checkpoint on CPU. Stage on the worker.
        with torch.device(device), set_default_torch_dtype(dtype):
            replacement = ReplicatedLinear(
                layer.in_features,
                layer.out_features,
                bias=layer.bias is not None,
                params_dtype=dtype,
                quant_config=config,
                prefix=f"text_encoder.layers.{name}",
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
        setattr(layers.get_submodule(parent), child, replacement)
        replaced += 1
    return replaced
