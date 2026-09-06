# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Per-branch cache state for SeaCache, mirroring TeaCacheState."""

from __future__ import annotations

import torch


class SeaCacheState:
    """Caching state for one CFG branch of a generation.

    Attributes:
        cnt: Number of forwards seen by this branch (0-based step index).
        accumulated_rel_l1_distance: Accumulated filtered relative-L1 distance
            since the last refresh.
        previous_modulated_input: Block-0 modulated input of the previous step
            (unfiltered on force-computed steps, filtered otherwise).
        previous_residual: Cached block-stack output residual of the last
            computed step.
        real_steps: Steps where the transformer blocks ran.
        skipped_steps: Steps where the cached residual was reused.
    """

    def __init__(self):
        self.cnt = 0
        self.accumulated_rel_l1_distance = 0.0
        self.previous_modulated_input: torch.Tensor | None = None
        self.previous_residual: torch.Tensor | None = None
        self.real_steps = 0
        self.skipped_steps = 0
