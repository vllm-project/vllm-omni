# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Qwen2 adapter for Omni's extra forward metadata.

All attention, logits and weight loading stay in vLLM's native Qwen2 model.
"""

from typing import Any

import torch
from vllm.model_executor.models.qwen2 import Qwen2ForCausalLM


class SparkQwen2ForCausalLM(Qwen2ForCausalLM):
    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        **omni_metadata: Any,
    ) -> torch.Tensor:
        del omni_metadata
        return super().forward(input_ids, positions, intermediate_tensors, inputs_embeds)
