# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
TeaCache state management (legacy).

Note: The current CFG-aware implementation manages state directly in the transformer
using branch-specific dictionaries (_qwen_teacache_states). This module is kept for
backward compatibility and potential future use.
"""

from typing import Optional

import torch


class TeaCacheState:
    """
    Legacy state management for TeaCache.

    Note: Not currently used by the CFG-aware Qwen implementation, which manages
    state directly with separate positive/negative branches.

    This class is kept for backward compatibility and non-CFG implementations.
    """

    def __init__(self):
        # Timestep tracking
        self.cnt = 0
        self.num_steps = 0

        # Caching state
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
