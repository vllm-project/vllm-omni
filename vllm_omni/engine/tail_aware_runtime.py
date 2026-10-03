# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Validate head-side admission before launching unchanged stage processes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from vllm_omni.config.omni_config import BaseVllmOmniStageConfig, VllmOmniDiffusionStageConfig


def prepare_tail_aware_stages(
    stages: list[BaseVllmOmniStageConfig],
    settings: dict[str, Any] | None,
    *,
    distributed: bool = False,
    async_chunk: bool = False,
    session_mode: str = "turn",
    api_client_count: int = 1,
) -> bool:
    """Validate typed stages for centralized FIFO admission.

    Admission state has one local owner; reject unsupported layouts before
    launching any stage process.
    """
    if not settings or not settings.get("enabled", False):
        return False
    if distributed or api_client_count != 1:
        raise ValueError("Tail-aware scheduling requires one local head with locally managed replicas")
    if len(stages) != 1 or async_chunk or session_mode != "turn":
        raise ValueError("Tail-aware scheduling currently supports one non-streaming diffusion stage")
    if stages[0].stage_type != "diffusion":
        raise ValueError("Tail-aware scheduling can only be enabled for a diffusion stage")
    stage = cast("VllmOmniDiffusionStageConfig", stages[0])
    if stage.diffusion_config.streaming_output:
        raise ValueError("Tail-aware scheduling does not support streaming diffusion output")
    return True
