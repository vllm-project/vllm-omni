# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
MagCache state management.

This module contains the MagCacheState class which manages the internal state
for MagCache caching logic, including residuals, accumulated metrics, and step tracking.
"""

import torch


class MagCacheState:
    """State management for MagCache caching logic."""

    def __init__(self) -> None:
        """Initialize empty MagCache state."""
        self.previous_residual: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None = None
        self.head_block_input: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None = None
        self.should_compute: bool = True
        self.accumulated_ratio: float = 1.0
        self.accumulated_err: float = 0.0
        self.accumulated_steps: int = 0
        self.step_index: int = 0
        self.calibration_ratios: list[float] = []
        self.norm_ratios: list[float] = []
        self.norm_stds: list[float] = []
        self.cos_dises: list[float] = []
        self._is_first_step: bool = True

    def reset(self) -> None:
        """Reset all state variables for a new inference run."""
        self.previous_residual = None
        self.head_block_input = None
        self.should_compute = True
        self.accumulated_ratio = 1.0
        self.accumulated_err = 0.0
        self.accumulated_steps = 0
        self.step_index = 0
        self.calibration_ratios = []
        self.norm_ratios = []
        self.norm_stds = []
        self.cos_dises = []
        self._is_first_step = True
