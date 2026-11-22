# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Typed AR-Diffusion control adapter for Cosmos3-Nano-Sim-Bimanual AgiBot ticks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.action_inputs import validate_action_values
from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.action_schema import UNIFIED_ACTION_DIM
from vllm_omni.experimental.ar_diffusion.tick_protocol import (
    ARDiffusionControlInput,
    ARDiffusionTickRequest,
)

COSMOS3_NANO_SIM_BIMANUAL_ACTION_TRACK = "robot_action"
COSMOS3_NANO_SIM_BIMANUAL_ACTION_SCHEMA = "robot_action.v2"
COSMOS3_NANO_SIM_BIMANUAL_LATENT_FRAMES_PER_TICK = 2
COSMOS3_NANO_SIM_BIMANUAL_ACTION_STEPS_PER_TICK = 8
COSMOS3_NANO_SIM_BIMANUAL_EMBODIMENT = "agibotworld"


@dataclass(frozen=True, slots=True)
class Cosmos3NanoSimBimanualTickInputs:
    """Validated model inputs reconstructed from one typed control payload."""

    action: torch.Tensor
    frame_idx: int
    num_latent_frames: int
    domain_name: str
    measure_tick_latency: bool


def build_cosmos3_nano_sim_bimanual_action_control(
    action: torch.Tensor,
    *,
    measure_tick_latency: bool = False,
) -> ARDiffusionControlInput:
    """Serialize a normalized unified action chunk into the model-neutral tick contract."""

    tensor = validate_action_values(action, width=UNIFIED_ACTION_DIM).cpu().contiguous()
    expected = (COSMOS3_NANO_SIM_BIMANUAL_ACTION_STEPS_PER_TICK, UNIFIED_ACTION_DIM)
    if tuple(tensor.shape) != expected:
        raise ValueError(f"Cosmos3-Nano-Sim-Bimanual ticks require action shape {expected}, got {tuple(tensor.shape)}.")
    data: dict[str, Any] = {"values": tensor.tolist()}
    if measure_tick_latency:
        data["measure_tick_latency"] = True
    return ARDiffusionControlInput(
        track=COSMOS3_NANO_SIM_BIMANUAL_ACTION_TRACK,
        schema=COSMOS3_NANO_SIM_BIMANUAL_ACTION_SCHEMA,
        data=data,
    )


def parse_cosmos3_nano_sim_bimanual_tick(tick: ARDiffusionTickRequest) -> Cosmos3NanoSimBimanualTickInputs:
    """Validate the schema-tagged action and reconstruct its float32 tensor."""

    controls = [control for control in tick.controls if control.track == COSMOS3_NANO_SIM_BIMANUAL_ACTION_TRACK]
    if len(controls) != 1:
        raise ValueError("Cosmos3-Nano-Sim-Bimanual typed ticks require exactly one robot_action control.")
    control = controls[0]
    if control.schema != COSMOS3_NANO_SIM_BIMANUAL_ACTION_SCHEMA:
        raise ValueError(
            "Cosmos3-Nano-Sim-Bimanual robot_action schema must be "
            f"{COSMOS3_NANO_SIM_BIMANUAL_ACTION_SCHEMA!r}, got {control.schema!r}."
        )
    data = control.data
    unexpected_fields = sorted(set(data) - {"values", "measure_tick_latency"})
    if unexpected_fields:
        raise ValueError(
            f"{COSMOS3_NANO_SIM_BIMANUAL_ACTION_SCHEMA} contains unsupported fields: {unexpected_fields}. "
            "Frame geometry and embodiment are derived from chunk_index and the schema."
        )
    frame_idx = 0 if tick.chunk_index == 0 else 1 + tick.chunk_index * COSMOS3_NANO_SIM_BIMANUAL_LATENT_FRAMES_PER_TICK
    measure_tick_latency = data.get("measure_tick_latency", False)
    if not isinstance(measure_tick_latency, bool):
        raise ValueError(f"{COSMOS3_NANO_SIM_BIMANUAL_ACTION_SCHEMA}.measure_tick_latency must be a boolean.")
    try:
        action = validate_action_values(data["values"], width=UNIFIED_ACTION_DIM)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{COSMOS3_NANO_SIM_BIMANUAL_ACTION_SCHEMA}.values must contain numeric action rows.") from exc
    expected_shape = (COSMOS3_NANO_SIM_BIMANUAL_ACTION_STEPS_PER_TICK, UNIFIED_ACTION_DIM)
    if tuple(action.shape) != expected_shape:
        raise ValueError(
            f"{COSMOS3_NANO_SIM_BIMANUAL_ACTION_SCHEMA}.values must have shape {expected_shape}, "
            f"got {tuple(action.shape)}."
        )
    return Cosmos3NanoSimBimanualTickInputs(
        action=action.contiguous(),
        frame_idx=frame_idx,
        num_latent_frames=COSMOS3_NANO_SIM_BIMANUAL_LATENT_FRAMES_PER_TICK,
        domain_name=COSMOS3_NANO_SIM_BIMANUAL_EMBODIMENT,
        measure_tick_latency=measure_tick_latency,
    )


__all__ = [
    "COSMOS3_NANO_SIM_BIMANUAL_ACTION_SCHEMA",
    "COSMOS3_NANO_SIM_BIMANUAL_ACTION_TRACK",
    "COSMOS3_NANO_SIM_BIMANUAL_LATENT_FRAMES_PER_TICK",
    "Cosmos3NanoSimBimanualTickInputs",
    "build_cosmos3_nano_sim_bimanual_action_control",
    "parse_cosmos3_nano_sim_bimanual_tick",
]
