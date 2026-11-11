# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""ABot-World components, loaded lazily so the interaction registry stays light."""

from __future__ import annotations

from typing import Any

__all__ = [
    "ABotCameraInteractionHandler",
    "ABotWorldCausalPipeline",
    "get_abot_world_post_process_func",
    "get_abot_world_pre_process_func",
]


def __getattr__(name: str) -> Any:
    if name == "ABotCameraInteractionHandler":
        from .interaction import ABotCameraInteractionHandler

        return ABotCameraInteractionHandler
    if name in (
        "ABotWorldCausalPipeline",
        "get_abot_world_post_process_func",
        "get_abot_world_pre_process_func",
    ):
        from .pipeline_abot_world import (
            ABotWorldCausalPipeline,
            get_abot_world_post_process_func,
            get_abot_world_pre_process_func,
        )

        return {
            "ABotWorldCausalPipeline": ABotWorldCausalPipeline,
            "get_abot_world_post_process_func": get_abot_world_post_process_func,
            "get_abot_world_pre_process_func": get_abot_world_pre_process_func,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
