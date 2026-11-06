# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Noisy PP cell schedule, executor, and versioned KV pools.

Graduated from ``experimental.ar_diffusion`` for request-path CausVid / Physis
cell PP. AR-Diffusion engine/runner/capability remain experimental and import
these symbols from here.
"""

from vllm_omni.diffusion.distributed.noisy_pipeline_parallel.noisy_pp_executor import (
    ARDiffusionCellContext,
    CellAdapter,
    CellRunSpec,
    CellTopology,
    resolve_pp_rank_and_group,
    run_cell_pipeline,
)
from vllm_omni.diffusion.distributed.noisy_pipeline_parallel.noisy_pp_plan import (
    CellPlan,
    CellSchedule,
    Ordering,
    build_cell_plan,
    cfg_step_replicas,
)
from vllm_omni.diffusion.distributed.noisy_pipeline_parallel.noisy import (
    ARDiffusionNoisyKVSpec,
    NoisyKVCache,
    NoisyKVState,
)

__all__ = [
    "ARDiffusionCellContext",
    "ARDiffusionNoisyKVSpec",
    "CellAdapter",
    "CellPlan",
    "CellRunSpec",
    "CellSchedule",
    "CellTopology",
    "NoisyKVCache",
    "NoisyKVState",
    "Ordering",
    "build_cell_plan",
    "cfg_step_replicas",
    "resolve_pp_rank_and_group",
    "run_cell_pipeline",
]
