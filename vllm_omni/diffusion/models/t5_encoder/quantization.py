# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Online FP8 for Hugging Face T5 and UMT5 encoders."""

import torch
from torch import nn
from transformers import T5EncoderModel, UMT5EncoderModel
from vllm.model_executor.layers.linear import ReplicatedLinear, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.utils.torch_utils import set_default_torch_dtype

from vllm_omni.quantization import ComponentQuantizationConfig


class _T5Fp8Linear(ReplicatedLinear):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.to(self.params_dtype)).to(x.dtype)


def prepare_t5_fp8(
    encoder: T5EncoderModel | UMT5EncoderModel,
    quant_config: QuantizationConfig | None,
    component: str,
    *,
    quantize_attention: bool = True,
) -> int:
    """Adapt loaded projections; the diffusion loader finalizes FP8 weights.

    Keep FFN output projections in their original precision: HF casts their
    input to wo.weight.dtype, which would otherwise cast activations to FP8
    without a scale. Embeddings, relative position bias and norms stay intact.
    """
    if not isinstance(quant_config, ComponentQuantizationConfig):
        return 0
    config = quant_config.component_configs.get(component)
    if config is None:
        return 0
    if (
        not isinstance(config, Fp8Config)
        or config.is_checkpoint_fp8_serialized
        or config.activation_scheme != "dynamic"
    ):
        raise ValueError(f"{component} supports dynamic online FP8 from an unquantized checkpoint only")

    blocks = encoder.encoder.block
    replaced = 0
    for name, layer in list(blocks.named_modules()):
        if not isinstance(layer, nn.Linear) or name.endswith(".wo"):
            continue
        if not quantize_attention and ".SelfAttention." in name:
            continue
        dtype = layer.weight.dtype if layer.weight.dtype in (torch.float16, torch.bfloat16) else torch.bfloat16
        with torch.device(layer.weight.device), set_default_torch_dtype(dtype):
            replacement = _T5Fp8Linear(
                layer.in_features,
                layer.out_features,
                bias=layer.bias is not None,
                params_dtype=dtype,
                quant_config=config,
                prefix=f"{component}.encoder.block.{name}",
                return_bias=False,
                disable_tp=True,
            )
        if isinstance(replacement.quant_method, UnquantizedLinearMethod):
            continue
        with torch.no_grad():
            replacement.weight.weight_loader(replacement.weight, layer.weight)
            if layer.bias is not None:
                replacement.bias.weight_loader(replacement.bias, layer.bias)
        replacement.train(layer.training)
        parent_name, _, child_name = name.rpartition(".")
        setattr(blocks.get_submodule(parent_name), child_name, replacement)
        replaced += 1
    return replaced
