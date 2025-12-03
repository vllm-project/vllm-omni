# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Optional

import torch


class TeaCacheState:
    """
    State management for TeaCache.

    Tracks caching state across diffusion timesteps, managing counters, accumulated
    distances, and cached values needed for the TeaCache algorithm.

    Attributes:
        cnt: Current timestep counter, incremented with each forward pass.
        num_steps: Total number of inference steps for the current run.
        accumulated_rel_l1_distance: Running accumulator for rescaled L1 distances.
        previous_modulated_input: Modulated input from previous timestep.
        previous_residual: Cached residual (output - input) from previous timestep.
    """

    def __init__(self):
        self.cnt = 0
        self.num_steps = 0
        self.accumulated_rel_l1_distance = 0.0
        self.previous_modulated_input: Optional[torch.Tensor] = None
        self.previous_residual: Optional[torch.Tensor] = None
        self.previous_encoder_residual: Optional[torch.Tensor] = None

    def reset(self):
        """Reset all state variables for a new inference run."""
        self.cnt = 0
        self.num_steps = 0
        self.accumulated_rel_l1_distance = 0.0
        self.previous_modulated_input = None
        self.previous_residual = None
        self.previous_encoder_residual = None
