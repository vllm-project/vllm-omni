# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Hook-based TeaCache implementation for vLLM-Omni.

This module implements a diffusers-style hook system that completely intercepts
the transformer forward pass, eliminating the need for any TeaCache-specific
code in model definitions. Model developers only need to add an extractor function
to support new models.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from vllm_omni.diffusion.cache.teacache.config import TeaCacheConfig
from vllm_omni.diffusion.cache.teacache.extractors import get_extractor
from vllm_omni.diffusion.cache.teacache.state import TeaCacheState
from vllm_omni.diffusion.distributed.parallel_state import (
    get_classifier_free_guidance_rank,
    get_classifier_free_guidance_world_size,
)
from vllm_omni.diffusion.hooks import HookRegistry, ModelHook, StateManager


class TeaCacheHook(ModelHook):
    """
    ModelHook implementing TeaCache for transformer models.

    This hook completely intercepts the transformer's forward pass and implements
    adaptive caching based on timestep embedding similarity. It's model-agnostic
    and supports multiple model types through extractor functions.

    Key features:
    - Zero changes to model code
    - CFG-aware with separate states for positive/negative branches
    - CFG-parallel compatible: properly detects branch identity across ranks
    - Model-specific polynomial rescaling
    - Auto-detection of model types

    Attributes:
        config: TeaCache configuration with thresholds and callbacks
        rescale_func: Polynomial function for rescaling L1 distances
        state_manager: Manages TeaCacheState across forward passes
        extractor_fn: Model-specific function to extract modulated input
    """

    _HOOK_NAME = "teacache"

    def __init__(self, config: TeaCacheConfig):
        """
        Initialize TeaCacheHook.

        Args:
            config: TeaCache configuration object.
        """
        super().__init__()
        self.config = config
        self.rescale_func = np.poly1d(config.coefficients)
        self.state_manager = StateManager(TeaCacheState)
        self.extractor_fn = None
        self._forward_cnt = 0

    def initialize_hook(self, module: torch.nn.Module) -> torch.nn.Module:
        """
        Initialize hook with extractor from config transformer model type.

        Args:
            module: The module to initialize the hook for.

        Returns:
            The initialized module.
        """
        # Get extractor function based on transformer_type from config
        # transformer_type is the transformer class name (e.g., "QwenImageTransformer2DModel")
        self.extractor_fn = get_extractor(self.config.transformer_type)

        # Set default context
        self.state_manager.set_context("teacache")

        return module

    def new_forward(self, module: torch.nn.Module, *args: Any, **kwargs: Any) -> Any:
        """
        Generic forward handler that works for ANY model.

        This method is completely model-agnostic. All model-specific logic
        is encapsulated in the extractor function that returns a CacheContext.

        The extractor does:
        - Model-specific preprocessing
        - Extraction of modulated input for cache decision
        - Providing transformer execution callable
        - Providing postprocessing callable

        This hook does:
        - CFG-aware state management
        - Cache decision logic (generic)
        - Residual caching and reuse

        Args:
            module: Transformer module (any architecture)
            *args: Positional arguments for model forward
            **kwargs: Keyword arguments for model forward

        Returns:
            Model output (format depends on model)
        """
        # Get model-specific context from extractor
        # The extractor encapsulates ALL model-specific logic
        ctx = self.extractor_fn(module, *args, **kwargs)

        # ============================================================================
        # GENERIC CACHING LOGIC (works for all models)
        # ============================================================================
        # Set context based on CFG branch for separate state tracking
        # With CFG-parallel, each rank processes only one branch:
        #   - cfg_rank 0: positive branch
        #   - cfg_rank > 0: negative branch
        # Without CFG-parallel, branches alternate within a single rank
        if getattr(module, "do_true_cfg", False):
            cfg_parallel_size = get_classifier_free_guidance_world_size()
            if cfg_parallel_size > 1:
                cfg_rank = get_classifier_free_guidance_rank()
                cache_branch = "negative" if cfg_rank > 0 else "positive"
            else:
                # No CFG-parallel: use forward counter to alternate branches
                cache_branch = "negative" if self._forward_cnt % 2 == 1 else "positive"
        else:
            cache_branch = "positive"

        context_name = f"teacache_{cache_branch}"
        self.state_manager.set_context(context_name)
        state = self.state_manager.get_state()

        # Decide whether to compute or cache based on modulated input similarity
        should_compute = self._should_compute_full_transformer(state, ctx.modulated_input)

        if not should_compute and state.previous_residual is not None:
            # ============================================================================
            # FAST PATH: Reuse cached residuals
            # ============================================================================
            ctx.hidden_states = ctx.hidden_states + state.previous_residual
            if state.previous_residual_encoder is not None and ctx.encoder_hidden_states is not None:
                ctx.encoder_hidden_states = ctx.encoder_hidden_states + state.previous_residual_encoder
            output = ctx.hidden_states
        else:
            # ============================================================================
            # SLOW PATH: Full transformer computation
            # ============================================================================
            ori_hidden_states = ctx.hidden_states.clone()
            ori_encoder_hidden_states = (
                ctx.encoder_hidden_states.clone() if ctx.encoder_hidden_states is not None else None
            )

            # Run transformer blocks using model-specific callable
            outputs = ctx.run_transformer_blocks()

            # Update context with outputs
            ctx.hidden_states = outputs[0]
            if len(outputs) > 1 and ctx.encoder_hidden_states is not None:
                ctx.encoder_hidden_states = outputs[1]

            # Cache residuals for next timestep
            state.previous_residual = (ctx.hidden_states - ori_hidden_states).detach()
            if ori_encoder_hidden_states is not None:
                state.previous_residual_encoder = (ctx.encoder_hidden_states - ori_encoder_hidden_states).detach()

            output = ctx.hidden_states

        # Update state
        state.previous_modulated_input = ctx.modulated_input.detach()
        state.cnt += 1
        self._forward_cnt += 1

        # ============================================================================
        # POSTPROCESSING (model-specific, via callable)
        # ============================================================================
        return ctx.postprocess(output)

    def _should_compute_full_transformer(self, state: TeaCacheState, modulated_inp: torch.Tensor) -> bool:
        """
        Determine whether to compute full transformer or reuse cached residual.

        This implements the core TeaCache algorithm:
        1. Always compute first timestep
        2. For intermediate steps:
           - Compute relative L1 distance between current and previous modulated inputs
           - Apply polynomial rescaling with model-specific coefficients
           - Accumulate rescaled distances
           - Compare to threshold: below = cache, above = compute

        Args:
            state: Current TeaCacheState containing counters and cached values
            modulated_inp: Modulated input extracted from first transformer block

        Returns:
            True to compute full transformer, False to reuse cached residual
        """
        # First timestep: always compute
        if state.cnt == 0:
            state.accumulated_rel_l1_distance = 0.0
            return True

        # Need previous input for comparison
        if state.previous_modulated_input is None:
            return True

        # Compute relative L1 distance between consecutive modulated inputs
        rel_distance = (
            (
                (modulated_inp - state.previous_modulated_input).abs().mean()
                / (state.previous_modulated_input.abs().mean() + 1e-8)
            )
            .cpu()
            .item()
        )

        # Apply model-specific polynomial rescaling
        rescaled_distance = float(self.rescale_func(rel_distance))
        state.accumulated_rel_l1_distance += abs(rescaled_distance)

        # Decision: below threshold = cache, above = compute
        if state.accumulated_rel_l1_distance < self.config.rel_l1_thresh:
            return False  # Use cache
        else:
            state.accumulated_rel_l1_distance = 0.0  # Reset accumulator
            return True  # Compute

    def reset_state(self, module: torch.nn.Module) -> torch.nn.Module:
        """
        Reset all cached states for a new inference run.

        Args:
            module: The module to reset state for.

        Returns:
            The module with reset state.
        """
        self.state_manager.reset()
        self._forward_cnt = 0
        return module


def apply_teacache_hook(module: torch.nn.Module, config: TeaCacheConfig) -> None:
    """
    Apply TeaCache optimization to a transformer module.

    This function registers a TeaCacheHook that completely intercepts the
    module's forward pass, implementing adaptive caching without any changes
    to the model code.

    Args:
        module: Transformer model to optimize (e.g., QwenImageTransformer2DModel)
        config: TeaCacheConfig specifying caching parameters

    Example:
        >>> config = TeaCacheConfig(
        ...     rel_l1_thresh=0.2,
        ...     transformer_type="QwenImageTransformer2DModel"
        ... )
        >>> apply_teacache_hook(transformer, config)
        >>> # Transformer bound to the pipeline now uses TeaCache automatically,
        ... # no code changes needed!
    """
    registry = HookRegistry.get_or_create(module)
    hook = TeaCacheHook(config)
    registry.register_hook(TeaCacheHook._HOOK_NAME, hook)
