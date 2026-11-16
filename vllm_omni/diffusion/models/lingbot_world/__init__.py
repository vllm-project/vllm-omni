# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""LingBot World components, loaded lazily for shared VAE/communication hooks."""

from __future__ import annotations

from typing import Any

__all__ = [
    "CausalLingBotWorldTransformer3DModel",
    "LingBotCameraControlReducer",
    "LingBotWorldCausalDMDPipeline",
    "get_lingbot_world_post_process_func",
    "get_lingbot_world_pre_process_func",
]


def __getattr__(name: str) -> Any:
    # Shared VAE and communication code can import the lightweight output and
    # ownership helpers before the LingBot pipeline itself is initialized.
    if name == "CausalLingBotWorldTransformer3DModel":
        from .transformer import CausalLingBotWorldTransformer3DModel

        return CausalLingBotWorldTransformer3DModel
    if name == "LingBotCameraControlReducer":
        from .actions import LingBotCameraControlReducer

        return LingBotCameraControlReducer
    if name in (
        "LingBotWorldCausalDMDPipeline",
        "get_lingbot_world_post_process_func",
        "get_lingbot_world_pre_process_func",
    ):
        from .pipeline import (
            LingBotWorldCausalDMDPipeline,
            get_lingbot_world_post_process_func,
            get_lingbot_world_pre_process_func,
        )

        return {
            "LingBotWorldCausalDMDPipeline": LingBotWorldCausalDMDPipeline,
            "get_lingbot_world_post_process_func": get_lingbot_world_post_process_func,
            "get_lingbot_world_pre_process_func": get_lingbot_world_pre_process_func,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
