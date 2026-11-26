# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Use the existing int8 Thinker weights with BF16 visual prefill activations."""

import torch
import torch_npu
from vllm_ascend.quantization.configs.compressed_tensors_config import AscendCompressedTensorsConfig
from vllm_ascend.quantization.methods import AscendW8A8DynamicLinearMethod

visual_prefill_active = False


class VisualPrefillA16(AscendW8A8DynamicLinearMethod):
    def prepare_input_for_overlap(self, layer, x):
        if visual_prefill_active:
            return None
        return super().prepare_input_for_overlap(layer, x)

    def apply(self, layer, x, bias=None, tp_rank=0):
        if not visual_prefill_active:
            return super().apply(layer, x, bias, tp_rank)
        if not isinstance(x, torch.Tensor) or x.dtype != torch.bfloat16:
            raise TypeError("Visual prefill requires unquantized BF16 activations")
        return torch_npu.npu_weight_quant_batchmatmul(
            x=x,
            weight=layer.weight,
            antiquant_scale=layer.weight_scale,
            bias=bias,
        )


def enable_visual_prefill_a16(config):
    if not isinstance(config, AscendCompressedTensorsConfig):
        raise TypeError("Visual prefill A16 requires an Ascend compressed-tensors checkpoint")
    original = config._create_scheme_for_layer_type

    def create(weight_quant, input_quant, format, layer_type):
        if (
            layer_type == "linear"
            and weight_quant.symmetric
            and config._detect_quant_type(weight_quant, input_quant, format) == "W8A8_DYNAMIC"
        ):
            return VisualPrefillA16()
        return original(weight_quant, input_quant, format, layer_type)

    config._create_scheme_for_layer_type = create
