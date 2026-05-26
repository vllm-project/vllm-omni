# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from abc import ABC, abstractmethod

from vllm.logger import init_logger

from vllm_omni.diffusion.data import DiffusionBatchingConfig
from vllm_omni.diffusion.sched.interface import DiffusionRequestState

logger = init_logger(__name__)


class DiffusionBatchingPolicy(ABC):
    """Policy plugin that derives a request-count batch cap."""

    def on_schedule_start(self) -> None:
        """Hook called once at the beginning of a scheduler cycle."""

    @abstractmethod
    def get_max_running_reqs(
        self,
        hard_cap: int,
        candidate_state: DiffusionRequestState,
    ) -> int:
        """Return the effective request cap for admitting ``candidate_state``."""


class FixedDiffusionBatchingPolicy(DiffusionBatchingPolicy):
    """Preserve the existing fixed ``max_num_seqs`` behavior."""

    def get_max_running_reqs(
        self,
        hard_cap: int,
        candidate_state: DiffusionRequestState,
    ) -> int:
        del candidate_state
        return max(1, int(hard_cap))


class ComputeBudgetDiffusionBatchingPolicy(DiffusionBatchingPolicy):
    """Map a profiled reference workload to per-request-type batch caps."""

    def __init__(self, config: DiffusionBatchingConfig) -> None:
        self.config = config

    def get_max_running_reqs(
        self,
        hard_cap: int,
        candidate_state: DiffusionRequestState,
    ) -> int:
        hard_cap = max(1, int(hard_cap))
        request_units = self.config.request_compute_units(
            candidate_state.req.sampling_params,
            candidate_state.req.prompts,
        )
        budget = self.config.effective_compute_unit_budget()
        dynamic_cap = (budget + request_units - 1) // request_units
        capped = max(1, min(hard_cap, dynamic_cap))

        if self.config.log_stats:
            logger.info(
                "[DiffusionBatching] request_compute_units=%s compute_unit_budget=%s "
                "dynamic_max_num_seqs=%s hard_max_num_seqs=%s effective_max_num_seqs=%s",
                request_units,
                budget,
                dynamic_cap,
                hard_cap,
                capped,
            )
        return capped


class DiffusionBatchProfilerPolicy(DiffusionBatchingPolicy):
    """Gradually increases batch cap so users can collect sweet-spot logs."""

    def __init__(self, config: DiffusionBatchingConfig) -> None:
        self.config = config
        self._schedule_count = 0
        self._current_cap = 1
        logger.info(
            "DiffusionBatchProfiler enabled (profiler_interval=%s)",
            self.config.profiler_interval,
        )

    def on_schedule_start(self) -> None:
        interval = max(1, int(self.config.profiler_interval))
        self._schedule_count += 1
        self._current_cap = self._schedule_count // interval + 1

    def get_max_running_reqs(
        self,
        hard_cap: int,
        candidate_state: DiffusionRequestState,
    ) -> int:
        del candidate_state
        hard_cap = max(1, int(hard_cap))
        cap = min(hard_cap, self._current_cap)
        logger.info("[DiffusionBatchProfiler] max_num_seqs=%s", cap)
        return cap


def create_diffusion_batching_policy(config: DiffusionBatchingConfig) -> DiffusionBatchingPolicy:
    if config.uses_compute_budget:
        logger.info(
            "ComputeBudgetDiffusionBatchingPolicy enabled (compute_unit_budget=%s)",
            config.effective_compute_unit_budget(),
        )
        return ComputeBudgetDiffusionBatchingPolicy(config)
    if config.uses_profiler:
        return DiffusionBatchProfilerPolicy(config)
    return FixedDiffusionBatchingPolicy()
