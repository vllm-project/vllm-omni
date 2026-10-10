# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Qwen3 chat model as a generative response judge (one YES/NO token)."""

from __future__ import annotations

from typing import Any

import torch
from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM
from vllm.sequence import IntermediateTensors


class ResponseJudgeQwen3ForCausalLM(Qwen3ForCausalLM):
    """Upstream Qwen3ForCausalLM served as an omni LLM_AR stage.

    The omni AR runner forwards extra bookkeeping kwargs (e.g.
    ``sampling_metadata``); the judge is a plain LM and ignores them.
    """

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | IntermediateTensors:
        return super().forward(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: Any = None,
    ) -> torch.Tensor | None:
        return super().compute_logits(hidden_states)


__all__ = ["ResponseJudgeQwen3ForCausalLM"]
