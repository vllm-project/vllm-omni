"""Shared MammothModa2 AR-to-DiT conditioning selection."""

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class MammothConditioningSpec:
    """Model-defined token categories used to build DiT conditions."""

    gen_vocab_start_index: int
    visual_token_ids: tuple[int, ...]


def conditioning_spec_from_config(config: Any) -> MammothConditioningSpec:
    """Read the condition-selection contract from the outer Mammoth config."""
    llm_config = getattr(config, "llm_config", None)
    if llm_config is None:
        raise ValueError("MammothModa2 config is missing llm_config")

    try:
        return MammothConditioningSpec(
            gen_vocab_start_index=int(llm_config.gen_vocab_start_index),
            visual_token_ids=(
                int(config.image_token_id),
                int(config.video_token_id),
                int(config.vision_start_token_id),
                int(config.vision_end_token_id),
            ),
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("MammothModa2 config is missing conditioning token ids") from exc


def select_ar_conditions(
    full_hidden_states: torch.Tensor,
    full_token_ids: list[int],
    answer_start_index: int,
    spec: MammothConditioningSpec,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select text and image conditioning rows with Mammoth's canonical masks."""
    if full_hidden_states.ndim != 2:
        raise ValueError(f"Expected AR hidden states [T,H], got shape={tuple(full_hidden_states.shape)}")
    if full_hidden_states.shape[0] != len(full_token_ids):
        raise ValueError(
            "AR hidden/token length mismatch while selecting Mammoth conditions: "
            f"hidden={full_hidden_states.shape[0]}, tokens={len(full_token_ids)}"
        )
    if not 0 <= answer_start_index <= len(full_token_ids):
        raise ValueError(f"Invalid Mammoth answer_start_index {answer_start_index} for {len(full_token_ids)} token ids")

    device = full_hidden_states.device
    token_ids = torch.tensor(full_token_ids, dtype=torch.long, device=device)
    positions = torch.arange(token_ids.shape[0], device=device)
    questions_mask = positions < answer_start_index
    generated_token_mask = token_ids >= spec.gen_vocab_start_index
    visual_token_mask = torch.isin(
        token_ids,
        torch.tensor(spec.visual_token_ids, dtype=torch.long, device=device),
    )
    text_mask = questions_mask & ~(visual_token_mask | generated_token_mask)
    image_mask = ~questions_mask & generated_token_mask

    return (
        full_hidden_states[text_mask].contiguous(),
        full_hidden_states[image_mask].contiguous(),
    )
