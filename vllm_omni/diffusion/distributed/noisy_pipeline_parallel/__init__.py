# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Noisy PP plan and executor (layer / vertical cell pipeline parallel).

Mainline depends only on ``SupportsNoisyPPKV``. Concrete NoisyKV stays under
``vllm_omni.experimental.ar_diffusion.kv_cache.noisy`` (wired by AR-Diffusion
runner), same pattern as stepwise ``SupportsStepExecution``.
"""

from vllm_omni.diffusion.distributed.noisy_pipeline_parallel.noisy_pp_executor import (
    ARDiffusionCellContext,
    CellAdapter,
    CellRunSpec,
    CellTopology,
    SupportsNoisyPPKV,
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

__all__ = [
    "ARDiffusionCellContext",
    "CellAdapter",
    "CellPlan",
    "CellRunSpec",
    "CellSchedule",
    "CellTopology",
    "Ordering",
    "SupportsNoisyPPKV",
    "build_cell_plan",
    "cfg_step_replicas",
    "resolve_pp_rank_and_group",
    "run_cell_pipeline",
]
