# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import PretrainedConfig
from vllm.config.lora import LoRAConfig

from vllm_omni.diffusion.layers.linear import MergedRowParallelLinear, to_rank_major
from vllm_omni.lora.layers.row_parallel_linear import DiffusionRowParallelLinearWithLoRA


class DiffusionMergedRowParallelLinearWithLoRA(DiffusionRowParallelLinearWithLoRA):
    """LoRA for ``MergedRowParallelLinear``.

    The base layer keeps ``cat([W_0^r, W_1^r, ...])`` on rank ``r``, so LoRA A (global
    ``[rank, sum(input_sizes)]``) gets the same per-segment re-ordering before the
    contiguous per-rank slice. LoRA B is replicated, as for ``RowParallelLinear``.

    Lives in the diffusion tree rather than next to the other shared wrappers in
    ``vllm_omni/lora/layers`` because its base layer is diffusion-specific.
    """

    def slice_lora_a(self, lora_a: torch.Tensor) -> torch.Tensor:
        base = self.base_layer
        return super().slice_lora_a(to_rank_major(lora_a, 1, base.input_sizes, self.tp_size, base.prefix))

    @classmethod
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: list,
        model_config: PretrainedConfig | None = None,
    ) -> bool:
        return type(source_layer) is MergedRowParallelLinear


__all__ = ["DiffusionMergedRowParallelLinearWithLoRA", "DiffusionRowParallelLinearWithLoRA"]
