# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Optional native quantized Qwen3-TTS text projection component.

The component is not enabled merely by importing this module. Its packed
weights must be selected explicitly by a validated candidate configuration.
"""

from __future__ import annotations

import torch
from torch import nn
from transformers.activations import ACT2FN
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.quantization.modelopt import ModelOptFp8Config, ModelOptNvFp4Config

from .checkpoint import load_tensor_file as load_file


def without_source_projection(weights, reference):
    """Consume and verify the four original tensors before substituting packed ones."""
    expected = {"linear_fc1.weight", "linear_fc1.bias", "linear_fc2.weight", "linear_fc2.bias"}
    if set(reference) != expected:
        raise ValueError("Invalid BF16 text projection reference")
    seen = set()
    for name, value in weights:
        if not name.startswith("talker.text_projection."):
            yield name, value
            continue
        short = name.removeprefix("talker.text_projection.")
        if short not in expected or short in seen:
            raise ValueError(f"Unexpected or duplicate original text projection tensor: {name}")
        original = reference[short]
        if value.dtype != original.dtype or not torch.equal(value.cpu(), original):
            raise ValueError(f"Original text projection differs from calibration source: {name}")
        seen.add(short)
    if seen != expected:
        raise ValueError("Missing original text projection tensors")


class QuantizedTextProjection(nn.Module):
    """Preserve the native two-linear SiLU architecture and BF16 biases."""

    def __init__(self, input_size, intermediate_size, output_size, act, precision):
        super().__init__()
        if precision not in ("fp8", "nvfp4"):
            raise ValueError(f"Unsupported text projection precision: {precision}")
        if precision == "nvfp4" and any(size % 16 for size in (input_size, intermediate_size, output_size)):
            raise ValueError("NVFP4 text projection dimensions must be divisible by16")
        quant = (
            ModelOptFp8Config("FP8", True, None, [])
            if precision == "fp8"
            else ModelOptNvFp4Config(is_checkpoint_nvfp4_serialized=True)
        )
        self.precision = precision
        self.linear_fc1 = ReplicatedLinear(
            input_size,
            intermediate_size,
            bias=True,
            params_dtype=torch.bfloat16,
            quant_config=quant,
            prefix="text_projection.linear_fc1",
            return_bias=False,
            disable_tp=True,
        )
        self.linear_fc2 = ReplicatedLinear(
            intermediate_size,
            output_size,
            bias=True,
            params_dtype=torch.bfloat16,
            quant_config=quant,
            prefix="text_projection.linear_fc2",
            return_bias=False,
            disable_tp=True,
        )
        self.act_fn = ACT2FN[act]

    def forward(self, hidden_state):
        return self.linear_fc2(self.act_fn(self.linear_fc1(hidden_state)))

    def load_packed_weights(self, path):
        """Strict serialized load; the outer vLLM loader processes kernels."""
        state = load_file(path, device="cpu")
        parameters = dict(self.named_parameters())
        if set(state) != set(parameters):
            raise ValueError("Text projection packed tensor names do not match the native module")
        with torch.no_grad():
            for name, parameter in parameters.items():
                value = state[name]
                if value.dtype != parameter.dtype:
                    raise ValueError(f"Text projection dtype mismatch at {name}")
                loader = getattr(parameter, "weight_loader", None)
                if loader is not None:
                    loader(parameter, value)
                else:
                    if value.shape != parameter.shape:
                        raise ValueError(f"Text projection shape mismatch at {name}")
                    parameter.copy_(value)
        return set(parameters)
