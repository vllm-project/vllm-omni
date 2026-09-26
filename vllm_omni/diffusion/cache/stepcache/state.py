# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Mutable state for step-level DiT velocity caching."""

from __future__ import annotations

import torch

from vllm_omni.diffusion.cache.stepcache.config import StepCacheConfig


class StepCacheState:
    """Per-generation mutable state (skip countdown)."""

    def __init__(self, config: StepCacheConfig) -> None:
        self.config = config
        self.skip_countdown = 0

    def reset(self) -> None:
        self.skip_countdown = 0

    def should_run_step(self, prev_predictions: list[tuple[torch.Tensor, ...]]) -> bool:
        """Return True when the DiT forward should execute at this scheduler step."""
        if not self.config.enabled:
            return True

        if len(prev_predictions) < self.config.min_history_steps:
            return True

        if self.skip_countdown > 1:
            self.skip_countdown -= 1
            return False
        if self.skip_countdown == 1:
            self.skip_countdown = 0
            return True

        v_last = prev_predictions[-1][0].flatten(1).float()
        v_prev = prev_predictions[-2][0].flatten(1).float()
        sim = torch.nn.functional.cosine_similarity(v_last, v_prev, dim=1).mean()

        for threshold, countdown in zip(self.config.sim_thresholds, self.config.skip_countdowns):
            if sim > threshold:
                self.skip_countdown = countdown
                return False

        return True

    def trim_history(self, prev_predictions: list[tuple[torch.Tensor, ...]]) -> None:
        while len(prev_predictions) > self.config.max_history:
            prev_predictions.pop(0)
