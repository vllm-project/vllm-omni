# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Resolve a stage's robot adapter for the OpenPI endpoint.

Every robot policy served so far has been a diffusion pipeline, which owns its
own observation transforms and returns `multimodal_output["actions"]`. A policy
whose actions are generated *tokens* has no such pipeline object, so the three
model-specific pieces — handshake values, observation → prompt, tokens → action
array — live under `vllm_omni/model_executor/models/<name>/` and the stage
declares the dotted path in its `StagePipelineConfig.robot_adapter`, the same
way it declares `custom_process_input_func` or `scheduler_cls`.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass(frozen=True)
class RobotARRequest:
    """The three fields `ServingRealtimeRobotOpenPI.infer` consumes."""

    prompt: Any
    sampling_params: Any
    request_id: str


def _load(path: str) -> Any:
    module_path, attr = path.rsplit(".", 1)
    return getattr(importlib.import_module(module_path), attr)


def resolve_robot_ar_adapter(engine_client: Any) -> Any | None:
    """A bound adapter if some stage declares one, else None."""
    for stage_config in getattr(engine_client, "stage_configs", []) or []:
        engine_args = getattr(stage_config, "engine_args", None)
        path = getattr(engine_args, "robot_adapter", None)
        if not path:
            continue
        adapter = _load(str(path)).from_engine_client(engine_client)
        logger.info("Robot OpenPI serving using the AR adapter %s", path)
        return adapter
    return None
