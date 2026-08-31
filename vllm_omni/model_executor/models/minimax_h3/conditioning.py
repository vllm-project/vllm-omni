# SPDX-License-Identifier: Apache-2.0
"""MiniMax H3 text-conditioning contract."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

MINIMAX_H3_TEXT_HIDDEN_SIZE = 5120
MINIMAX_H3_PRESENTATION_TASK_KEY = "_minimax_h3_presentation_task"
MINIMAX_H3_CONDITION_LABELS_KEY = "_minimax_h3_condition_labels"


@dataclass(frozen=True)
class MiniMaxH3TextConditioning:
    """Layer-50 Qwen3-VL hidden states and their H3 token roles."""

    hidden_states: torch.Tensor
    token_tags: torch.Tensor

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
    ) -> MiniMaxH3TextConditioning:
        hidden_states = payload.get("hidden_states")
        token_tags = payload.get("token_tags")
        if not isinstance(hidden_states, torch.Tensor) or not isinstance(token_tags, torch.Tensor):
            raise ValueError("MiniMax H3 conditioning requires hidden_states and token_tags tensors")
        if hidden_states.ndim != 2 or hidden_states.shape[-1] != MINIMAX_H3_TEXT_HIDDEN_SIZE:
            raise ValueError(
                "MiniMax H3 hidden_states must have shape "
                f"[tokens, {MINIMAX_H3_TEXT_HIDDEN_SIZE}], got {tuple(hidden_states.shape)}"
            )
        if token_tags.ndim != 1 or token_tags.shape[0] != hidden_states.shape[0]:
            raise ValueError(
                "MiniMax H3 token_tags must align with hidden_states, got "
                f"token_tags={tuple(token_tags.shape)} and hidden_states={tuple(hidden_states.shape)}"
            )
        if not torch.all((token_tags == 0) | (token_tags == 1)):
            raise ValueError("MiniMax H3 text-encoder token_tags must contain only 0 and 1")
        return cls(hidden_states=hidden_states, token_tags=token_tags)

    def to_payload(self) -> dict[str, torch.Tensor]:
        return {
            "hidden_states": self.hidden_states,
            "token_tags": self.token_tags,
        }
