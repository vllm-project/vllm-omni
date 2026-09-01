# SPDX-License-Identifier: Apache-2.0
"""Typed AR-Diffusion control adapter for Cosmos-Dreams AgiBot ticks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from vllm_omni.diffusion.models.cosmos_dreams.action_contract import (
    AGIBOT_DOMAIN_ID,
    AGIBOT_RAW_ACTION_DIM,
)
from vllm_omni.experimental.ar_diffusion.tick_protocol import (
    ARDiffusionControlInput,
    ARDiffusionTickRequest,
)

COSMOS_DREAMS_ACTION_TRACK = "robot_action"
COSMOS_DREAMS_ACTION_SCHEMA = "robot_action.v1"
COSMOS_DREAMS_LATENT_FRAMES_PER_TICK = 4
COSMOS_DREAMS_ACTION_STEPS_PER_TICK = 16
COSMOS_DREAMS_EMBODIMENT = "agibotworld"


@dataclass(frozen=True, slots=True)
class CosmosDreamsTickInputs:
    """Validated model inputs reconstructed from one typed control payload."""

    action: torch.Tensor
    frame_idx: int
    num_latent_frames: int
    domain_name: str
    domain_id: int
    measure_tick_latency: bool


def build_cosmos_dreams_action_control(
    action: torch.Tensor,
    *,
    measure_tick_latency: bool = False,
) -> ARDiffusionControlInput:
    """Serialize a raw AgiBot chunk into the model-neutral tick contract."""

    tensor = torch.as_tensor(action, dtype=torch.float32).detach().cpu().contiguous()
    expected = (COSMOS_DREAMS_ACTION_STEPS_PER_TICK, AGIBOT_RAW_ACTION_DIM)
    if tuple(tensor.shape) != expected:
        raise ValueError(f"Cosmos-Dreams ticks require raw action shape {expected}, got {tuple(tensor.shape)}.")
    if not torch.isfinite(tensor).all():
        raise ValueError("Cosmos-Dreams raw action contains NaN or Inf values.")
    data: dict[str, Any] = {"values": tensor.tolist()}
    if measure_tick_latency:
        data["measure_tick_latency"] = True
    return ARDiffusionControlInput(
        track=COSMOS_DREAMS_ACTION_TRACK,
        schema=COSMOS_DREAMS_ACTION_SCHEMA,
        data=data,
    )


def parse_cosmos_dreams_tick(tick: ARDiffusionTickRequest) -> CosmosDreamsTickInputs:
    """Validate the schema-tagged action and reconstruct its float32 tensor."""

    controls = [control for control in tick.controls if control.track == COSMOS_DREAMS_ACTION_TRACK]
    if len(controls) != 1:
        raise ValueError("Cosmos-Dreams typed ticks require exactly one robot_action control.")
    control = controls[0]
    if control.schema != COSMOS_DREAMS_ACTION_SCHEMA:
        raise ValueError(
            f"Cosmos-Dreams robot_action schema must be {COSMOS_DREAMS_ACTION_SCHEMA!r}, got {control.schema!r}."
        )
    data = control.data
    unexpected_fields = sorted(set(data) - {"values", "measure_tick_latency"})
    if unexpected_fields:
        raise ValueError(
            f"{COSMOS_DREAMS_ACTION_SCHEMA} contains unsupported fields: {unexpected_fields}. "
            "Frame geometry and embodiment are derived from chunk_index and the schema."
        )
    frame_idx = 0 if tick.chunk_index == 0 else 1 + tick.chunk_index * COSMOS_DREAMS_LATENT_FRAMES_PER_TICK
    measure_tick_latency = data.get("measure_tick_latency", False)
    if not isinstance(measure_tick_latency, bool):
        raise ValueError(f"{COSMOS_DREAMS_ACTION_SCHEMA}.measure_tick_latency must be a boolean.")
    try:
        action = torch.tensor(data["values"], dtype=torch.float32)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{COSMOS_DREAMS_ACTION_SCHEMA}.values must contain numeric action rows.") from exc
    expected_shape = (COSMOS_DREAMS_ACTION_STEPS_PER_TICK, AGIBOT_RAW_ACTION_DIM)
    if tuple(action.shape) != expected_shape:
        raise ValueError(
            f"{COSMOS_DREAMS_ACTION_SCHEMA}.values must have shape {expected_shape}, got {tuple(action.shape)}."
        )
    if not torch.isfinite(action).all():
        raise ValueError(f"{COSMOS_DREAMS_ACTION_SCHEMA}.values contains NaN or Inf values.")
    return CosmosDreamsTickInputs(
        action=action.contiguous(),
        frame_idx=frame_idx,
        num_latent_frames=COSMOS_DREAMS_LATENT_FRAMES_PER_TICK,
        domain_name=COSMOS_DREAMS_EMBODIMENT,
        domain_id=AGIBOT_DOMAIN_ID,
        measure_tick_latency=measure_tick_latency,
    )


__all__ = [
    "COSMOS_DREAMS_ACTION_SCHEMA",
    "COSMOS_DREAMS_ACTION_TRACK",
    "COSMOS_DREAMS_LATENT_FRAMES_PER_TICK",
    "CosmosDreamsTickInputs",
    "build_cosmos_dreams_action_control",
    "parse_cosmos_dreams_tick",
]
