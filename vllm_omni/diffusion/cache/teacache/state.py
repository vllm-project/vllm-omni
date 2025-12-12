# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
TeaCache state management.

This module manages the state for TeaCache hooks across diffusion timesteps.
"""

from typing import Optional

import torch


class TeaCacheState:
    """
    State management for TeaCache hook.

    Tracks caching state across diffusion timesteps, managing counters,
    accumulated distances, and cached residuals for the TeaCache algorithm.
    """

    def __init__(self):
        """Initialize empty TeaCache state."""
        # Timestep tracking
        self.cnt = 0

        # Caching state
        self.accumulated_rel_l1_distance = 0.0
        self.previous_modulated_input: Optional[torch.Tensor] = None
        self.previous_residual: Optional[torch.Tensor] = None
        self.previous_residual_encoder: Optional[torch.Tensor] = None

    def reset(self) -> None:
        """Reset all state variables for a new inference run."""
        self.cnt = 0
        self.accumulated_rel_l1_distance = 0.0
        self.previous_modulated_input = None
        self.previous_residual = None
        self.previous_residual_encoder = None
