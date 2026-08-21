# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import torch
import torch.nn as nn

from .base_linear import DiffusionBaseLinearLayerWithLoRA


class DiffusionTorchLinearWithLoRA(DiffusionBaseLinearLayerWithLoRA):
    """LoRA wrapper for replicated ``torch.nn.Linear`` diffusion layers."""

    def __init__(self, base_layer: nn.Linear):
        nn.Module.__init__(self)
        self.base_layer = base_layer
        self.input_size = base_layer.in_features
        self.output_size = base_layer.out_features
        self.output_slices = (base_layer.out_features,)
        self.n_slices = 1
        self.tp_size = 1
        self.tp_rank = 0
        self.device = base_layer.weight.device

    @classmethod
    def can_replace_layer(cls, source_layer, lora_config, packed_modules_list, model_config=None) -> bool:
        del lora_config, model_config
        return isinstance(source_layer, nn.Linear) and not packed_modules_list

    def create_lora_weights(self, max_loras: int, lora_config, model_config=None) -> None:
        del model_config
        self.lora_config = lora_config
        rank = lora_config.max_lora_rank
        factory_kwargs = {
            "device": self.base_layer.weight.device,
            "dtype": lora_config.lora_dtype,
        }
        self.lora_a_stacked = (torch.zeros(max_loras, 1, rank, self.input_size, **factory_kwargs),)
        self.lora_b_stacked = (torch.zeros(max_loras, 1, self.output_size, rank, **factory_kwargs),)
        object.__setattr__(self, "_diffusion_base_layer_ref", self.base_layer)
        self._diffusion_lora_active_slices = (False,)
        self._diffusion_additive_bias = (None,)

    def slice_lora_a(self, lora_a):
        return lora_a

    def slice_lora_b(self, lora_b):
        return lora_b

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.apply(x, self.base_layer.bias)
