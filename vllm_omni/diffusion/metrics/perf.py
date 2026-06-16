# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from typing import TypeAlias

from vllm.v1.metrics.perf import PerfStats

from vllm_omni.diffusion.metrics.common import PatchSize3D
from vllm_omni.diffusion.metrics.estimators import (
    DiTFlopsBreakdown,
    DiTForwardStats,
    FluxDiTForwardStats,
    HunyuanVideoDiTForwardStats,
    WanDiTForwardStats,
    collect_flux_dit_forward_stats,
    estimate_dit_flops_per_gpu,
    estimate_dit_forward_flops_per_gpu,
    estimate_flux_dit_forward_flops_per_gpu,
    estimate_hunyuan_video_dit_forward_flops_per_gpu,
    estimate_wan_dit_forward_flops_per_gpu,
)

DiffusionForwardStats: TypeAlias = (
    DiTForwardStats | FluxDiTForwardStats | WanDiTForwardStats | HunyuanVideoDiTForwardStats
)


def estimate_diffusion_forward_flops_per_gpu(stats: DiffusionForwardStats) -> int:
    """Dispatch estimated/theoretical diffusion FLOPs by stats type."""

    if isinstance(stats, DiTForwardStats):
        return estimate_dit_forward_flops_per_gpu(stats)
    if isinstance(stats, FluxDiTForwardStats):
        return estimate_flux_dit_forward_flops_per_gpu(stats)
    if isinstance(stats, WanDiTForwardStats):
        return estimate_wan_dit_forward_flops_per_gpu(stats)
    if isinstance(stats, HunyuanVideoDiTForwardStats):
        return estimate_hunyuan_video_dit_forward_flops_per_gpu(stats)
    raise TypeError(f"Unsupported diffusion FLOPs stats type: {type(stats)!r}")


def to_perf_stats(breakdown: DiTFlopsBreakdown) -> PerfStats:
    """Convert a generic DiT estimate breakdown into upstream vLLM PerfStats."""

    return PerfStats(num_flops_per_gpu=breakdown.total_flops_per_gpu)


def estimate_diffusion_perf_stats(stats: DiffusionForwardStats) -> PerfStats:
    """Convert estimated diffusion FLOPs into upstream vLLM ``PerfStats``."""

    return PerfStats(
        num_flops_per_gpu=estimate_diffusion_forward_flops_per_gpu(stats),
        num_read_bytes_per_gpu=0,
        num_write_bytes_per_gpu=0,
    )


def estimate_dit_perf_stats(stats: DiffusionForwardStats) -> PerfStats:
    """Backward-compatible wrapper for diffusion estimated FLOPs stats."""

    return estimate_diffusion_perf_stats(stats)


__all__ = [
    "DiTFlopsBreakdown",
    "DiTForwardStats",
    "DiffusionForwardStats",
    "FluxDiTForwardStats",
    "HunyuanVideoDiTForwardStats",
    "PatchSize3D",
    "WanDiTForwardStats",
    "collect_flux_dit_forward_stats",
    "estimate_diffusion_forward_flops_per_gpu",
    "estimate_diffusion_perf_stats",
    "estimate_dit_flops_per_gpu",
    "estimate_dit_forward_flops_per_gpu",
    "estimate_dit_perf_stats",
    "estimate_flux_dit_forward_flops_per_gpu",
    "estimate_hunyuan_video_dit_forward_flops_per_gpu",
    "estimate_wan_dit_forward_flops_per_gpu",
    "to_perf_stats",
]
