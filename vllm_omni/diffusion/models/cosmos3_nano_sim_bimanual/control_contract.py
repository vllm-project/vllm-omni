# SPDX-License-Identifier: Apache-2.0
"""Strict conditioning contract for Cosmos3-Nano-Sim-Transfer artifacts."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator, model_validator

from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.action_contract import (
    Cosmos3NanoSimBimanualActionSchema,
    canonical_sha256,
)

TRANSFER_HINTS = ("edge", "blur", "depth", "seg")
TRANSFER_SYSTEM_PROMPT_ID = "cosmos3_transfer_v1"
TRANSFER_CONTROL_ATTENTION_MODE = "causal_control_with_rgb_history"

TransferHint = Literal["edge", "blur", "depth", "seg"]


class Cosmos3NanoSimBimanualActionConditioning(Cosmos3NanoSimBimanualActionSchema):
    """Schema-v1 action-conditioning branch."""

    mode: Literal["action"]


class Cosmos3NanoSimBimanualControlVideoConditioning(BaseModel):
    """Immutable schema-v1 payload pinned to the target Transfer checkpoint."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: Literal["control_video"]
    hints: tuple[TransferHint, ...]
    transfer_control_attention_mode: Literal["causal_control_with_rgb_history"]
    share_vision_temporal_positions: Literal[True]
    system_prompt_id: Literal["cosmos3_transfer_v1"]
    emphasize_control_in_prompt: Literal[True]
    no_eviction: Literal[True]

    @field_validator(
        "share_vision_temporal_positions",
        "emphasize_control_in_prompt",
        "no_eviction",
        mode="before",
    )
    @classmethod
    def validate_literal_true(cls, value: object) -> object:
        if value is not True:
            raise ValueError("Cosmos3-Nano-Sim-Transfer contract flags must be JSON boolean true.")
        return value

    @model_validator(mode="after")
    def validate_target_contract(self) -> Cosmos3NanoSimBimanualControlVideoConditioning:
        if self.hints != TRANSFER_HINTS:
            raise ValueError(
                f"Cosmos3-Nano-Sim-Transfer hints must exactly match {list(TRANSFER_HINTS)}, got {list(self.hints)}."
            )
        return self

    @property
    def digest(self) -> str:
        return canonical_sha256(self.model_dump(mode="json"))


Cosmos3NanoSimBimanualConditioning = Annotated[
    Cosmos3NanoSimBimanualActionConditioning | Cosmos3NanoSimBimanualControlVideoConditioning,
    Field(discriminator="mode"),
]
_CONDITIONING_ADAPTER = TypeAdapter(Cosmos3NanoSimBimanualConditioning)


def parse_cosmos3_nano_sim_bimanual_conditioning(value: object) -> Cosmos3NanoSimBimanualConditioning:
    """Parse schema-v1 conditioning by its required ``mode`` discriminator."""

    return _CONDITIONING_ADAPTER.validate_python(value)


__all__ = [
    "Cosmos3NanoSimBimanualActionConditioning",
    "Cosmos3NanoSimBimanualControlVideoConditioning",
    "Cosmos3NanoSimBimanualConditioning",
    "TRANSFER_CONTROL_ATTENTION_MODE",
    "TRANSFER_HINTS",
    "TRANSFER_SYSTEM_PROMPT_ID",
    "TransferHint",
    "parse_cosmos3_nano_sim_bimanual_conditioning",
]
