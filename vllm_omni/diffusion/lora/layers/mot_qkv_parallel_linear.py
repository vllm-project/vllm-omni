# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import torch
from torch import nn
from transformers import PretrainedConfig
from vllm.config.lora import LoRAConfig

from vllm_omni.diffusion.layers.mot.mot_qkv_parallel_linear import MoTQKVParallelLinear

from .column_parallel_linear import DiffusionMergedQKVParallelLinearWithLoRA


class DiffusionMoTQKVParallelLinearWithLoRA(DiffusionMergedQKVParallelLinearWithLoRA):
    """Apply text-expert QKV adapters without changing MoT image-token routing."""

    @classmethod
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: list[str],
        model_config: PretrainedConfig | None = None,
    ) -> bool:
        return (
            type(source_layer) is MoTQKVParallelLinear
            and len(packed_modules_list) == 3
            and not lora_config.fully_sharded_loras
        )

    def forward(
        self,
        input_: torch.Tensor,
        text_indices: torch.Tensor | None = None,
        vae_indices: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
        if text_indices is None:
            return super().forward(input_)

        result = self.base_layer(input_, text_indices, vae_indices)
        if not any(self._diffusion_lora_active_slices) or text_indices.numel() == 0:
            return result

        output = result[0] if isinstance(result, tuple) else result
        output[text_indices] = self.apply_lora(input_[text_indices], output[text_indices])
        return result
