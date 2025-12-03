# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
TeaCache core module - model-agnostic caching logic.

TeaCache speeds up diffusion model inference by caching transformer block computations
when consecutive timestep embeddings are similar, avoiding redundant forward passes.
"""

import numpy as np
import torch
from typing import Optional, Dict, Tuple

from vllm_omni.diffusion.teacache.config import TeaCacheConfig
from vllm_omni.diffusion.teacache.extractors import get_extractor


class TeaCacheWrapper:
    """
    Model-agnostic wrapper that manages TeaCache state and caching decisions.

    This class handles the core caching logic and can be used with any transformer model.
    It supports CFG-aware caching with separate states for positive/negative branches.

    Attributes:
        config: TeaCache configuration with thresholds and model type
        rescale_func: Polynomial function to rescale L1 distances
        extractor_fn: Model-specific function to extract modulated input
        cfg_states: Dictionary holding separate states for CFG branches
    """

    def __init__(self, config: TeaCacheConfig):
        self.config = config

        # Polynomial rescaling function maps raw L1 distances to scaled values
        # This accounts for model-specific characteristics in how embeddings change
        self.rescale_func = np.poly1d(config.coefficients)

        # Get the model-specific extractor function
        # This knows how to extract the modulated input from the first transformer block
        self.extractor_fn = get_extractor(config.model_type)

        # CFG-aware state management
        # Separate states for positive and negative branches prevent cache corruption
        self.cfg_states: Dict[str, Dict] = {}

    def _get_or_create_state(self, branch: str) -> Dict:
        """Get or create state dictionary for a specific CFG branch."""
        if branch not in self.cfg_states:
            self.cfg_states[branch] = {
                "accumulated_rel_l1_distance": 0.0,
                "previous_modulated_input": None,
                "previous_hidden_residual": None,
                "previous_encoder_residual": None,
            }
        return self.cfg_states[branch]

    def should_compute(
        self,
        module,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        branch: str = "default"
    ) -> Tuple[bool, torch.Tensor]:
        """
        Decide whether to compute full transformer or reuse cached results.

        This is the core TeaCache decision logic, model-agnostic and CFG-aware.

        Args:
            module: Transformer module instance
            hidden_states: Current hidden states tensor
            temb: Timestep embedding tensor
            branch: CFG branch identifier ("positive", "negative", or "default")

        Returns:
            Tuple of (should_compute, modulated_input):
                - should_compute: True to run full computation, False to use cache
                - modulated_input: Extracted modulated input (for state tracking)
        """
        state = self._get_or_create_state(branch)

        # Extract modulated input from first transformer block
        # This captures how the timestep embedding affects the input
        modulated_inp = self.extractor_fn(module, hidden_states, temb)

        # First timestep for this branch - must compute
        if state["previous_modulated_input"] is None:
            state["accumulated_rel_l1_distance"] = 0.0
            state["previous_modulated_input"] = modulated_inp.detach()
            return True, modulated_inp

        # Compare current input to previous input
        # rel_distance measures how much the input has changed
        prev_inp = state["previous_modulated_input"]
        rel_distance = (
            (modulated_inp - prev_inp).abs().mean().cpu().item()
            / (prev_inp.abs().mean().cpu().item() + 1e-8)  # Avoid division by zero
        )

        # Apply model-specific polynomial scaling
        # Different models have different change patterns, so we rescale accordingly
        rescaled_distance = abs(float(self.rescale_func(rel_distance)))
        state["accumulated_rel_l1_distance"] += rescaled_distance

        # Decision: If accumulated change exceeds threshold, recompute. Otherwise, use cache.
        if state["accumulated_rel_l1_distance"] >= self.config.rel_l1_thresh:
            should_compute = True
            state["accumulated_rel_l1_distance"] = 0.0  # Reset accumulator
        else:
            should_compute = False  # Use cached results

        # Save current input for next timestep comparison
        state["previous_modulated_input"] = modulated_inp.detach()

        return should_compute, modulated_inp

    def get_cached_residuals(
        self, branch: str = "default"
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Get cached residuals for a specific CFG branch.

        Args:
            branch: CFG branch identifier

        Returns:
            Tuple of (hidden_residual, encoder_residual)
        """
        state = self._get_or_create_state(branch)
        return (
            state["previous_hidden_residual"],
            state["previous_encoder_residual"]
        )

    def cache_residuals(
        self,
        hidden_residual: torch.Tensor,
        encoder_residual: Optional[torch.Tensor] = None,
        branch: str = "default"
    ):
        """
        Cache residuals for future reuse.

        Args:
            hidden_residual: Residual (output - input) for hidden states
            encoder_residual: Residual for encoder hidden states (if applicable)
            branch: CFG branch identifier
        """
        state = self._get_or_create_state(branch)
        state["previous_hidden_residual"] = hidden_residual.detach()
        if encoder_residual is not None:
            state["previous_encoder_residual"] = encoder_residual.detach()

    def reset(self):
        """Reset all cached states (call at start of new generation)."""
        self.cfg_states.clear()


def apply_teacache(module, config: TeaCacheConfig) -> TeaCacheWrapper:
    """
    Apply TeaCache to a transformer module.

    This attaches a TeaCacheWrapper instance to the module, which the transformer's
    forward pass can then use for model-agnostic caching logic.

    Args:
        module: Transformer module to apply TeaCache to
        config: TeaCache configuration

    Returns:
        TeaCacheWrapper instance attached to the module

    Example:
        >>> config = TeaCacheConfig(rel_l1_thresh=0.2, model_type="Qwen")
        >>> wrapper = apply_teacache(transformer, config)
        >>> # In transformer forward:
        >>> should_compute, _ = wrapper.should_compute(self, hidden_states, temb, branch="positive")
    """
    wrapper = TeaCacheWrapper(config)
    module._teacache_wrapper = wrapper
    return wrapper
