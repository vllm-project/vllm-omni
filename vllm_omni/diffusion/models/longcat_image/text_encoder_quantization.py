# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Opt-in online FP8 for the LongCat language decoder."""

import torch
from torch import nn
from transformers import Qwen2_5_VLForConditionalGeneration
from vllm.model_executor.layers.linear import ReplicatedLinear, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.utils.torch_utils import set_default_torch_dtype

from vllm_omni.quantization import ComponentQuantizationConfig


class _EncoderFp8Linear(ReplicatedLinear):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # HF can load FP32 weights; the FP8 GEMM requires FP16/BF16 output.
        return super().forward(x.to(self.params_dtype)).to(x.dtype)


def prepare_text_encoder_fp8(
    encoder: Qwen2_5_VLForConditionalGeneration,
    quant_config: QuantizationConfig | None,
) -> int:
    """Adapt loaded decoder linears; the pipeline loader finalizes quantization.

    Require an explicit text_encoder entry. Global quantization retains its
    existing scope. Vision, embeddings, normalization and lm_head stay intact.
    HF still loads the original weights, so the initial load peak is unchanged.
    """
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
        raise ValueError("LongCat text_encoder supports dynamic online FP8 from an unquantized checkpoint only")

    decoder = encoder.get_decoder()
    decoder_prefix = next(name for name, module in encoder.named_modules() if module is decoder)
    replaced = 0
    for name, layer in list(decoder.layers.named_modules()):
        if not isinstance(layer, nn.Linear):
            continue
        dtype = layer.weight.dtype if layer.weight.dtype in (torch.float16, torch.bfloat16) else torch.bfloat16
        with torch.device(layer.weight.device), set_default_torch_dtype(dtype):
            replacement = _EncoderFp8Linear(
                layer.in_features,
                layer.out_features,
                bias=layer.bias is not None,
                params_dtype=dtype,
                quant_config=config,
                prefix=f"text_encoder.{decoder_prefix}.layers.{name}",
                return_bias=False,
                disable_tp=True,
            )
        if isinstance(replacement.quant_method, UnquantizedLinearMethod):
            continue
        with torch.no_grad():
            # Online FP8 parameters may start on meta. Use their native loaders
            # to materialize weights and preserve the backend loading lifecycle.
            replacement.weight.weight_loader(replacement.weight, layer.weight)
            if layer.bias is not None:
                replacement.bias.weight_loader(replacement.bias, layer.bias)
        replacement.train(layer.training)
        parent_name, _, child_name = name.rpartition(".")
        parent = decoder.layers.get_submodule(parent_name)
        setattr(parent, child_name, replacement)
        replaced += 1
    return replaced
