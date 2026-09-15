# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MiniMax H3 text-conditioning contract."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

MINIMAX_H3_TEXT_CONDITIONING_SCHEMA = "minimax_h3.text_conditioning/v1"
MINIMAX_H3_TEXT_HIDDEN_SIZE = 5120
MINIMAX_H3_PRESENTATION_TASK_KEY = "_minimax_h3_presentation_task"
MINIMAX_H3_CONDITION_LABELS_KEY = "_minimax_h3_condition_labels"


def _validate_contiguous_strided_layout(name: str, tensor: torch.Tensor) -> None:
    if tensor.layout != torch.strided:
        raise ValueError(
            f"{MINIMAX_H3_TEXT_CONDITIONING_SCHEMA}: {name} must use contiguous strided layout, got {tensor.layout}"
        )
    if not tensor.is_contiguous():
        raise ValueError(
            f"{MINIMAX_H3_TEXT_CONDITIONING_SCHEMA}: {name} must use contiguous "
            f"strided layout, got stride={tuple(tensor.stride())}"
        )


@dataclass(frozen=True)
class MiniMaxH3TextConditioning:
    """``minimax_h3.text_conditioning/v1`` semantic payload."""

    hidden_states: torch.Tensor
    token_tags: torch.Tensor

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
    ) -> MiniMaxH3TextConditioning:
        """Validate the semantic payload consumed by the diffusion stage."""
        hidden_states = payload.get("hidden_states")
        token_tags = payload.get("token_tags")
        if not isinstance(hidden_states, torch.Tensor) or not isinstance(token_tags, torch.Tensor):
            raise ValueError(
                f"{MINIMAX_H3_TEXT_CONDITIONING_SCHEMA}: conditioning requires hidden_states and token_tags tensors"
            )
        if hidden_states.ndim != 2 or hidden_states.shape[-1] != MINIMAX_H3_TEXT_HIDDEN_SIZE:
            raise ValueError(
                f"{MINIMAX_H3_TEXT_CONDITIONING_SCHEMA}: hidden_states must have shape "
                f"[tokens, {MINIMAX_H3_TEXT_HIDDEN_SIZE}], got {tuple(hidden_states.shape)}"
            )
        if hidden_states.dtype != torch.bfloat16:
            raise ValueError(
                f"{MINIMAX_H3_TEXT_CONDITIONING_SCHEMA}: hidden_states must have dtype "
                f"torch.bfloat16, got {hidden_states.dtype}"
            )
        _validate_contiguous_strided_layout("hidden_states", hidden_states)
        if token_tags.ndim != 1 or token_tags.shape[0] != hidden_states.shape[0]:
            raise ValueError(
                f"{MINIMAX_H3_TEXT_CONDITIONING_SCHEMA}: token_tags must align with hidden_states, got "
                f"token_tags={tuple(token_tags.shape)} and hidden_states={tuple(hidden_states.shape)}"
            )
        if token_tags.dtype != torch.int64:
            raise ValueError(
                f"{MINIMAX_H3_TEXT_CONDITIONING_SCHEMA}: token_tags must have dtype torch.int64, got {token_tags.dtype}"
            )
        _validate_contiguous_strided_layout("token_tags", token_tags)
        if not torch.all((token_tags == 0) | (token_tags == 1)):
            raise ValueError(
                f"{MINIMAX_H3_TEXT_CONDITIONING_SCHEMA}: text-encoder token_tags must contain only 0 and 1"
            )
        return cls(hidden_states=hidden_states, token_tags=token_tags)

    @classmethod
    def from_omni_payload(
        cls,
        payload: Mapping[str, Any],
    ) -> MiniMaxH3TextConditioning:
        """Validate and adapt the existing ``OmniPayload`` stage-wire view."""
        hidden_states = payload.get("hidden_states")
        if not isinstance(hidden_states, Mapping):
            raise ValueError(f"{MINIMAX_H3_TEXT_CONDITIONING_SCHEMA}: text encoder returned no hidden_states payload")
        hidden = hidden_states.get("output")
        if not isinstance(hidden, torch.Tensor):
            raise ValueError(
                f"{MINIMAX_H3_TEXT_CONDITIONING_SCHEMA}: text encoder returned no hidden_states.output tensor"
            )

        meta = payload.get("meta")
        if not isinstance(meta, Mapping):
            raise ValueError(f"{MINIMAX_H3_TEXT_CONDITIONING_SCHEMA}: text encoder returned no conditioning metadata")
        token_role_ids = meta.get("token_role_ids")
        if not isinstance(token_role_ids, torch.Tensor):
            raise ValueError(f"{MINIMAX_H3_TEXT_CONDITIONING_SCHEMA}: text encoder returned no token_role_ids tensor")
        if token_role_ids.ndim != 2 or token_role_ids.shape[-1] != 1:
            raise ValueError(
                f"{MINIMAX_H3_TEXT_CONDITIONING_SCHEMA}: stage-wire token_role_ids must have shape "
                f"[tokens, 1], got {tuple(token_role_ids.shape)}"
            )
        if token_role_ids.dtype != torch.int64:
            raise ValueError(
                f"{MINIMAX_H3_TEXT_CONDITIONING_SCHEMA}: stage-wire token_role_ids must have dtype "
                f"torch.int64, got {token_role_ids.dtype}"
            )
        _validate_contiguous_strided_layout("stage-wire token_role_ids", token_role_ids)

        return cls.from_payload(
            {
                "hidden_states": hidden,
                "token_tags": token_role_ids.squeeze(-1),
            }
        )

    def to_payload(self) -> dict[str, torch.Tensor]:
        return {
            "hidden_states": self.hidden_states,
            "token_tags": self.token_tags,
        }
