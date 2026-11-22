# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Cosmos3-Nano-Sim-Auto: causal rolling multiview (RGB + optional LiDAR) diffusion model family."""

from vllm_omni.diffusion.models.cosmos3_nano_sim_auto.geometry import (
    Chunk,
    SensorGrid,
    build_chunk_schedule,
    lidar_grid,
    rgb_grid,
)

__all__ = ["Chunk", "SensorGrid", "build_chunk_schedule", "lidar_grid", "rgb_grid"]
