# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Callable, Optional

import numpy as np
import torch

from vllm_omni.diffusion.teacache.config import TeaCacheConfig
from vllm_omni.diffusion.teacache.extractors import get_extractor
from vllm_omni.diffusion.teacache.state import TeaCacheState


class TeaCacheWrapper:
    """
    Wrapper that manages TeaCache state and caching decisions.

    This class is used by the transformer's forward method to implement
    adaptive caching based on timestep embedding similarity.
    """

    def __init__(self, config: TeaCacheConfig):
        self.config = config
        self.state = TeaCacheState()
        self.state.num_steps = config.num_inference_steps

        # Initialize polynomial rescaling function
        self.rescale_func = np.poly1d(config.coefficients)

        # Get model-specific extractor
        self.extractor_fn = get_extractor(config.model_type)

    def should_compute_full_transformer(
        self, module, hidden_states: torch.Tensor, temb: torch.Tensor
    ) -> bool:
        """
        Determine whether to compute full transformer or reuse cached residual.

        Always computes first and last timesteps. For intermediate timesteps,
        compares accumulated L1 distance to threshold.

        Args:
            module: Transformer module instance
            hidden_states: Current hidden states tensor
            temb: Timestep embedding tensor

        Returns:
            True to compute full transformer, False to reuse cache
        """
        # Reset if we've completed all steps (new inference run)
        if self.state.cnt == self.state.num_steps and self.state.num_steps > 0:
            self.state.cnt = 0
            self.state.accumulated_rel_l1_distance = 0.0
            self.state.previous_modulated_input = None
            self.state.previous_residual = None
            self.state.previous_encoder_residual = None

        # Extract modulated input
        modulated_inp = self.extractor_fn(module, hidden_states, temb)

        # Always compute first timestep
        if self.state.cnt == 0:
            self.state.accumulated_rel_l1_distance = 0.0
            self.state.previous_modulated_input = modulated_inp
            return True

        # Always compute last timestep
        if self.state.num_steps > 0 and self.state.cnt == self.state.num_steps - 1:
            self.state.accumulated_rel_l1_distance = 0.0
            return True

        # Need previous modulated input for comparison
        if self.state.previous_modulated_input is None:
            self.state.previous_modulated_input = modulated_inp
            return True

        # Compute relative L1 distance
        rel_distance = (
            (modulated_inp - self.state.previous_modulated_input)
            .abs()
            .mean()
            .cpu()
            .item()
            / self.state.previous_modulated_input.abs().mean().cpu().item()
        )

        # Apply polynomial rescaling
        rescaled_distance = self.rescale_func(rel_distance)
        self.state.accumulated_rel_l1_distance += rescaled_distance

        # Make decision based on accumulated threshold
        should_compute = self.state.accumulated_rel_l1_distance >= self.config.rel_l1_thresh
        if should_compute:
            self.state.accumulated_rel_l1_distance = 0.0  # Reset accumulator

        # Store current modulated input for next iteration
        self.state.previous_modulated_input = modulated_inp

        return should_compute

    def cache_residual(self, residual: torch.Tensor, encoder_residual: Optional[torch.Tensor] = None):
        """Cache the residual from transformer computation."""
        self.state.previous_residual = residual
        if encoder_residual is not None:
            self.state.previous_encoder_residual = encoder_residual

    def get_cached_residual(self) -> torch.Tensor:
        """Get the cached residual."""
        return self.state.previous_residual

    def get_cached_encoder_residual(self) -> Optional[torch.Tensor]:
        """Get the cached encoder residual."""
        return self.state.previous_encoder_residual

    def increment_counter(self):
        """Increment timestep counter."""
        self.state.cnt += 1

    def reset(self):
        """Reset state for a new inference run."""
        self.state.reset()
        self.state.num_steps = self.config.num_inference_steps


def apply_teacache(module, config: TeaCacheConfig) -> TeaCacheWrapper:
    """
    Apply TeaCache to a transformer module.

    This function stores the TeaCache wrapper on the module for use in forward pass.

    Args:
        module: Transformer module to apply TeaCache to
        config: TeaCache configuration

    Returns:
        TeaCacheWrapper instance
    """
    wrapper = TeaCacheWrapper(config)
    module._teacache_wrapper = wrapper
    return wrapper
