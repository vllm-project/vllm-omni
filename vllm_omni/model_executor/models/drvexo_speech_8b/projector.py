# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Explicit low-precision input projection; independent of language quantization."""

from __future__ import annotations

import torch
from torch import nn
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.quantization.modelopt import ModelOptFp8Config, ModelOptNvFp4Config

from .checkpoint import load_tensor_file as load_file


class QuantizedInputProjector(nn.Module):
    """Two native vLLM linears with BF16 normalization, activation and bias."""

    def __init__(self, audio_dim: int, language_dim: int, precision: str):
        super().__init__()
        if precision not in ("fp8", "nvfp4"):
            raise ValueError(f"Unsupported input projector precision: {precision}")
        if precision == "nvfp4" and (audio_dim % 16 or language_dim % 16):
            raise ValueError("NVFP4 input projector dimensions must be divisible by16")
        quant = (
            ModelOptFp8Config("FP8", True, None, [])
            if precision == "fp8"
            else ModelOptNvFp4Config(is_checkpoint_nvfp4_serialized=True)
        )
        self.precision = precision
        self.net = nn.Sequential(
            nn.LayerNorm(audio_dim, dtype=torch.bfloat16),
            ReplicatedLinear(
                audio_dim,
                language_dim,
                params_dtype=torch.bfloat16,
                quant_config=quant,
                prefix="input_projector.net.1",
                return_bias=False,
                disable_tp=True,
            ),
            nn.GELU(),
            ReplicatedLinear(
                language_dim,
                language_dim,
                params_dtype=torch.bfloat16,
                quant_config=quant,
                prefix="input_projector.net.3",
                return_bias=False,
                disable_tp=True,
            ),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[-1] != self.net[0].normalized_shape[0]:
            raise ValueError("Incorrect input projector feature dimension")
        return self.net(features)

    def load_packed_weights(self, path: str) -> set[str]:
        """Load serialized tensors strictly; vLLM processes kernels afterwards."""
        state = load_file(path, device="cpu")
        parameters = dict(self.named_parameters())
        if set(state) != set(parameters):
            raise ValueError(
                f"Input projector tensor mismatch: missing={set(parameters) - set(state)}, "
                f"extra={set(state) - set(parameters)}"
            )
        with torch.no_grad():
            for name, parameter in parameters.items():
                value = state[name]
                if parameter.dtype != value.dtype:
                    raise ValueError(f"Input projector dtype mismatch at{name}: {value.dtype} vs {parameter.dtype}")
                loader = getattr(parameter, "weight_loader", None)
                if loader is not None:
                    loader(parameter, value)
                else:
                    if value.shape != parameter.shape:
                        raise ValueError(f"Input projector shape mismatch at{name}")
                    parameter.copy_(value)
        return set(parameters)
