# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

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
    get_sp_group,
    model_parallel_is_initialized,
)
from vllm_omni.diffusion.hooks import HookRegistry, ModelHook, StateManager


def _average_l1_stats_across_sp(mean_diff: torch.Tensor, mean_prev: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Average TeaCache L1 stats across sequence-parallel ranks.

    Skip vs compute must be identical on every rank that participates in
    transformer SP collectives. Sequence shards otherwise produce different
    local means and deadlock. TP is not included: the residual TeaCache reads
    is replicated across TP ranks. CFG/DP ranks are also excluded: they do
    not share those collectives.
    """
    if not model_parallel_is_initialized():
        return mean_diff, mean_prev

    sp_group = get_sp_group()
    if sp_group.world_size <= 1:
        return mean_diff, mean_prev

    stats = torch.stack((mean_diff.detach(), mean_prev.detach()))
    stats = sp_group.all_reduce(stats) / sp_group.world_size
    return stats[0], stats[1]


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
        # Set context based on CFG branch for separate state tracking.
        # Explicit sources win, most specific first:
        #   1. the extractor's per-call hint, ctx.extra_states["teacache_branch"]
        #      (MammothModa2 passes it through its forward kwargs);
        #   2. module.cfg_branch, stamped by CFGParallelMixin on every call it
        #      makes, which stays correct when non-CFG forwards are interleaved
        #      (step execution).
        # Callers that provide neither fall back to the inferred branch:
        #   - CFG-parallel: cfg_rank 0 is positive, cfg_rank > 0 negative
        #   - otherwise branches are assumed to alternate on this rank
        cache_branch = self._explicit_cfg_branch(module, ctx)
        if cache_branch is None:
            cache_branch = self._infer_cfg_branch(module)

        context_name = f"teacache_{cache_branch}"
        self.state_manager.set_context(context_name)
        state = self.state_manager.get_state()

        # Decide whether to compute or cache based on modulated input similarity
        local_should_compute = self._should_compute_full_transformer(state, ctx.modulated_input)
        sync_cache_decision = (ctx.extra_states or {}).get("synchronize_cache_decision")
        if sync_cache_decision is not None:
            should_compute = sync_cache_decision(local_should_compute)
            # A rank that locally chose the cache path must reset its counter
            # when another SP rank requires a full collective block execution.
            if should_compute and not local_should_compute:
                state.accumulated_rel_l1_distance = 0.0
        else:
            should_compute = local_should_compute

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

            # Handle models with additional blocks (e.g., Flux2 single_transformer_blocks)
            if getattr(ctx, "extra_states", None) and "run_flux2_full_transformer_with_single" in ctx.extra_states:
                run_full = ctx.extra_states["run_flux2_full_transformer_with_single"]
                ctx.hidden_states, ctx.encoder_hidden_states = run_full(ori_hidden_states, ori_encoder_hidden_states)
                output = ctx.hidden_states
                state.previous_residual = (ctx.hidden_states - ori_hidden_states).detach()
            else:
                # Run transformer blocks using model-specific callable
                outputs = ctx.run_transformer_blocks()
                # Update context with outputs
                ctx.hidden_states = outputs[0]
                if len(outputs) > 1 and ctx.encoder_hidden_states is not None:
                    ctx.encoder_hidden_states = outputs[1]

                output = ctx.hidden_states

                # Cache residuals for next timestep
                state.previous_residual = (ctx.hidden_states - ori_hidden_states).detach()
                if ori_encoder_hidden_states is not None:
                    state.previous_residual_encoder = (ctx.encoder_hidden_states - ori_encoder_hidden_states).detach()

        # Update state
        state.previous_modulated_input = ctx.modulated_input.detach()
        state.cnt += 1
        self._forward_cnt += 1

        # ============================================================================
        # POSTPROCESSING (model-specific, via callable)
        # ============================================================================
        return ctx.postprocess(output)

    @staticmethod
    def _explicit_cfg_branch(module: torch.nn.Module, ctx: Any) -> str | None:
        """Branch named by the extractor hint or the caller's stamp, if any."""
        extra_states = getattr(ctx, "extra_states", None) or {}
        source, branch = "teacache_branch", extra_states.get("teacache_branch")
        if branch is None:
            source, branch = "cfg_branch", getattr(module, "cfg_branch", None)
        if branch is not None and branch not in ("positive", "negative"):
            raise ValueError(f"Invalid {source}={branch!r}; expected 'positive' or 'negative'.")
        return branch

    def _infer_cfg_branch(self, module: torch.nn.Module) -> str:
        """Fallback branch for callers that don't stamp ``module.cfg_branch``."""
        if not getattr(module, "do_true_cfg", False):
            return "positive"
        if get_classifier_free_guidance_world_size() > 1:
            return "negative" if get_classifier_free_guidance_rank() > 0 else "positive"
        # Sequential CFG without a stamp: assume calls alternate on this rank.
        return "negative" if self._forward_cnt % 2 == 1 else "positive"

    def _should_compute_full_transformer(self, state: TeaCacheState, modulated_inp: torch.Tensor) -> bool:
        """
        Determine whether to compute full transformer or reuse cached residual.

        This implements the core TeaCache algorithm:
        1. Always compute first timestep
        2. For intermediate steps:
           - Compute relative L1 distance between current and previous modulated inputs
           - Average that distance across SP so all ranks share the skip decision
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

        # Compute relative L1 distance between consecutive modulated inputs.
        # Reduce across SP before the threshold so every rank takes the same
        # skip/compute path (otherwise a cache hit on one rank deadlocks
        # waiting on collectives the miss path never enters).
        mean_diff = (modulated_inp - state.previous_modulated_input).abs().mean()
        mean_prev = state.previous_modulated_input.abs().mean()
        mean_diff, mean_prev = _average_l1_stats_across_sp(mean_diff, mean_prev)
        rel_distance = (mean_diff / (mean_prev + 1e-8)).item()

        # Apply model-specific polynomial rescaling
        rescaled_distance = float(self.rescale_func(rel_distance))
        state.accumulated_rel_l1_distance += abs(rescaled_distance)

        # Decision: below threshold = cache, above = compute
        rel_l1_thresh = self.config.rel_l1_thresh
        assert rel_l1_thresh is not None
        if state.accumulated_rel_l1_distance < rel_l1_thresh:
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
