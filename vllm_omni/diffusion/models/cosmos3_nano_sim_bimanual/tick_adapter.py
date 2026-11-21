# SPDX-License-Identifier: Apache-2.0
"""Typed AR-Diffusion control adapter for Cosmos3-Nano-Sim-Bimanual AgiBot ticks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.action_contract import (
    AGIBOT_DOMAIN_ID,
    AGIBOT_RAW_ACTION_DIM,
)
from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.action_inputs import validate_action_values
from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.action_schema import UNIFIED_ACTION_DIM
from vllm_omni.experimental.ar_diffusion.tick_protocol import (
    ARDiffusionControlInput,
    ARDiffusionTickRequest,
)

COSMOS3_NANO_SIM_BIMANUAL_ACTION_TRACK = "robot_action"
COSMOS3_NANO_SIM_BIMANUAL_ACTION_SCHEMA = "robot_action.v1"
COSMOS3_NANO_SIM_BIMANUAL_UNIFIED_ACTION_SCHEMA = "robot_action.v2"
_ACTION_SPACES = {
    COSMOS3_NANO_SIM_BIMANUAL_ACTION_SCHEMA: ("raw", AGIBOT_RAW_ACTION_DIM, 4),
    COSMOS3_NANO_SIM_BIMANUAL_UNIFIED_ACTION_SCHEMA: ("normalized_unified_v1", UNIFIED_ACTION_DIM, 2),
}
COSMOS3_NANO_SIM_BIMANUAL_LATENT_FRAMES_PER_TICK = _ACTION_SPACES[COSMOS3_NANO_SIM_BIMANUAL_ACTION_SCHEMA][2]
COSMOS3_NANO_SIM_BIMANUAL_ACTION_STEPS_PER_TICK = COSMOS3_NANO_SIM_BIMANUAL_LATENT_FRAMES_PER_TICK * 4
COSMOS3_NANO_SIM_BIMANUAL_EMBODIMENT = "agibotworld"


@dataclass(frozen=True, slots=True)
class Cosmos3NanoSimBimanualTickInputs:
    """Validated model inputs reconstructed from one typed control payload."""

    action: torch.Tensor
    frame_idx: int
    num_latent_frames: int
    domain_name: str
    domain_id: int | None
    measure_tick_latency: bool
    action_space: str


def build_cosmos3_nano_sim_bimanual_action_control(
    action: torch.Tensor,
    *,
    measure_tick_latency: bool = False,
    schema: str | None = None,
    chunk_size: int | None = None,
) -> ARDiffusionControlInput:
    """Serialize raw legacy or normalized unified AgiBot actions."""

    if schema is None:
        shape = getattr(action, "shape", None)
        if shape is None:
            shape = torch.as_tensor(action).shape
        schema = (
            COSMOS3_NANO_SIM_BIMANUAL_UNIFIED_ACTION_SCHEMA
            if len(shape) >= 2 and shape[-1] == UNIFIED_ACTION_DIM
            else COSMOS3_NANO_SIM_BIMANUAL_ACTION_SCHEMA
        )
    _, width, default_chunk = _tick_contract(schema)
    frames = default_chunk if chunk_size is None else chunk_size
    tensor = validate_action_values(action, width=width).cpu().contiguous()
    _validate_tick_shape(tensor, schema, frames, width)
    data: dict[str, Any] = {"values": tensor.tolist()}
    if measure_tick_latency:
        data["measure_tick_latency"] = True
    return ARDiffusionControlInput(
        track=COSMOS3_NANO_SIM_BIMANUAL_ACTION_TRACK,
        schema=schema,
        data=data,
    )


def _tick_contract(schema: str) -> tuple[str, int, int]:
    try:
        return _ACTION_SPACES[schema]
    except KeyError as exc:
        raise ValueError(f"Unsupported Cosmos3-Nano-Sim-Bimanual robot_action schema {schema!r}.") from exc


def _validate_tick_shape(action: torch.Tensor, schema: str, frames: int, width: int) -> None:
    if type(frames) is not int or frames <= 0:
        raise ValueError("Robot action chunk_size must be a positive integer.")
    expected = (frames * 4, width)
    if action.shape != expected:
        raise ValueError(f"{schema}.values must have shape {expected}, got {tuple(action.shape)}.")


def parse_cosmos3_nano_sim_bimanual_tick(
    tick: ARDiffusionTickRequest,
    *,
    schema: str | None = None,
    chunk_size: int | None = None,
) -> Cosmos3NanoSimBimanualTickInputs:
    """Validate the schema-tagged action and reconstruct its float32 tensor."""

    controls = [control for control in tick.controls if control.track == COSMOS3_NANO_SIM_BIMANUAL_ACTION_TRACK]
    if len(controls) != 1:
        raise ValueError("Cosmos3-Nano-Sim-Bimanual typed ticks require exactly one robot_action control.")
    control = controls[0]
    if schema is not None and control.schema != schema:
        raise ValueError(f"Robot action schema must be {schema!r}, got {control.schema!r}.")
    schema = control.schema
    action_space, width, default_chunk = _tick_contract(schema)
    frames = default_chunk if chunk_size is None else chunk_size
    data = control.data
    unexpected_fields = sorted(set(data) - {"values", "measure_tick_latency"})
    if unexpected_fields:
        raise ValueError(f"{schema} contains unsupported fields: {unexpected_fields}.")
    frame_idx = 0 if tick.chunk_index == 0 else 1 + tick.chunk_index * frames
    measure_tick_latency = data.get("measure_tick_latency", False)
    if not isinstance(measure_tick_latency, bool):
        raise ValueError(f"{schema}.measure_tick_latency must be a boolean.")
    try:
        action = validate_action_values(data["values"], width=width)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{schema}.values must contain numeric action rows. {exc}") from exc
    _validate_tick_shape(action, schema, frames, width)
    return Cosmos3NanoSimBimanualTickInputs(
        action=action.contiguous(),
        frame_idx=frame_idx,
        num_latent_frames=frames,
        domain_name=COSMOS3_NANO_SIM_BIMANUAL_EMBODIMENT,
        domain_id=AGIBOT_DOMAIN_ID if schema == COSMOS3_NANO_SIM_BIMANUAL_ACTION_SCHEMA else None,
        measure_tick_latency=measure_tick_latency,
        action_space=action_space,
    )


__all__ = [
    "COSMOS3_NANO_SIM_BIMANUAL_ACTION_SCHEMA",
    "COSMOS3_NANO_SIM_BIMANUAL_ACTION_TRACK",
    "COSMOS3_NANO_SIM_BIMANUAL_UNIFIED_ACTION_SCHEMA",
    "COSMOS3_NANO_SIM_BIMANUAL_LATENT_FRAMES_PER_TICK",
    "Cosmos3NanoSimBimanualTickInputs",
    "build_cosmos3_nano_sim_bimanual_action_control",
    "parse_cosmos3_nano_sim_bimanual_tick",
]
