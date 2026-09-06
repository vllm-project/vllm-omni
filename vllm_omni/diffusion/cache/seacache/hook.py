# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Hook-based SeaCache implementation for vLLM-Omni.

SeaCache (https://arxiv.org/abs/2602.18993, CVPR 2026) keeps TeaCache's
accumulate-and-refresh schedule but measures the step-to-step distance after
a timestep-dependent Wiener filter, so the distance tracks content change
rather than noise; there are no per-checkpoint fitted coefficients.

The hook mirrors ``TeaCacheHook``: it intercepts the transformer forward and
delegates all model-specific work to the TeaCache extractor registry. A model
supports SeaCache when its extractor provides ``sigma`` and ``grid_hw`` on
the ``CacheContext``.
"""

from __future__ import annotations

from typing import Any

import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.cache.seacache.config import SeaCacheConfig
from vllm_omni.diffusion.cache.seacache.filter import (
    _POWER_EXP_IMAGE,
    ab_from_sigma,
    apply_sea_filter,
    rel_l1,
)
from vllm_omni.diffusion.cache.seacache.state import SeaCacheState
from vllm_omni.diffusion.cache.teacache.extractors import get_extractor
from vllm_omni.diffusion.distributed.parallel_state import (
    get_classifier_free_guidance_rank,
    get_classifier_free_guidance_world_size,
    get_sequence_parallel_world_size,
)
from vllm_omni.diffusion.hooks import HookRegistry, ModelHook, StateManager

logger = init_logger(__name__)


class SeaCacheHook(ModelHook):
    """Intercepts the transformer forward like TeaCacheHook; all model-specific
    logic lives in the extractor. Adds the SEA filter before the distance and
    drops TeaCache's polynomial rescaling.
    """

    _HOOK_NAME = "seacache"

    def __init__(self, config: SeaCacheConfig):
        super().__init__()
        self.config = config
        self.state_manager = StateManager(SeaCacheState)
        self.extractor_fn = None
        self._forward_cnt = 0
        # Set by the backend's refresh() before each generation; needed to
        # force the last step to run. Kept across reset_state().
        self.num_inference_steps: int | None = None
        self._sp_warned = False

    def initialize_hook(self, module: torch.nn.Module) -> torch.nn.Module:
        self.extractor_fn = get_extractor(self.config.transformer_type)
        self.state_manager.set_context("seacache")
        return module

    def new_forward(self, module: torch.nn.Module, *args: Any, **kwargs: Any) -> Any:
        ctx = self.extractor_fn(module, *args, **kwargs)

        if self._sequence_parallel_active():
            return self._run_uncached(ctx)

        # Same CFG branch selection as TeaCacheHook: CFG-parallel gives each
        # rank one branch; otherwise branches alternate (positive first).
        if getattr(module, "do_true_cfg", False):
            cfg_parallel_size = get_classifier_free_guidance_world_size()
            if cfg_parallel_size > 1:
                cfg_rank = get_classifier_free_guidance_rank()
                cache_branch = "negative" if cfg_rank > 0 else "positive"
            else:
                cache_branch = "negative" if self._forward_cnt % 2 == 1 else "positive"
        else:
            cache_branch = "positive"

        self.state_manager.set_context(f"seacache_{cache_branch}")
        state = self.state_manager.get_state()

        should_compute = self._should_compute_full_transformer(state, ctx)

        if not should_compute and state.previous_residual is not None:
            # FAST PATH: reuse the cached block-stack residual.
            ctx.hidden_states = ctx.hidden_states + state.previous_residual
            output = ctx.hidden_states
            state.skipped_steps += 1
        else:
            # SLOW PATH: full transformer computation; cache the residual delta.
            ori_hidden_states = ctx.hidden_states.clone()
            self._run_full_stack(ctx)
            output = ctx.hidden_states
            state.previous_residual = (ctx.hidden_states - ori_hidden_states).detach()
            state.real_steps += 1

        state.cnt += 1
        self._forward_cnt += 1
        self._maybe_log_summary(state, cache_branch)
        return ctx.postprocess(output)

    def _should_compute_full_transformer(self, state: SeaCacheState, ctx: Any) -> bool:
        """Decide from the accumulated SEA-filtered distance; the first and
        last steps always run.
        """
        is_last_step = self.num_inference_steps is not None and state.cnt == self.num_inference_steps - 1
        if state.cnt == 0 or is_last_step or state.previous_modulated_input is None:
            # The reference stores the unfiltered feature on force-computed
            # steps and the filtered one elsewhere; kept as-is, it is the
            # schedule the published numbers come from.
            state.accumulated_rel_l1_distance = 0.0
            state.previous_modulated_input = self._decision_feature(ctx).detach()
            return True

        sigma = getattr(ctx, "sigma", None)
        grid_hw = getattr(ctx, "grid_hw", None)
        if sigma is None or grid_hw is None:
            # Extractor did not provide the SEA inputs; run uncached.
            return True

        feature = self._decision_feature(ctx)
        a, b = ab_from_sigma(sigma)
        height, width = grid_hw
        filtered = apply_sea_filter(
            feature.reshape(feature.shape[0], height, width, -1),
            a=a,
            b=b,
            power_exp=_POWER_EXP_IMAGE,
            norm_mode=self.config.sea_norm_mode,
        ).reshape(feature.shape)

        state.accumulated_rel_l1_distance += rel_l1(filtered, state.previous_modulated_input)
        state.previous_modulated_input = filtered.detach()

        should_compute = state.accumulated_rel_l1_distance >= self.config.sea_thresh
        if should_compute:
            state.accumulated_rel_l1_distance = 0.0
        return should_compute

    @staticmethod
    def _decision_feature(ctx: Any) -> torch.Tensor:
        """The modulated input sliced to the leading ``grid_seq_len`` tokens
        when the model appends tokens after the grid (e.g. Qwen-Image-Edit's
        step-constant condition tokens): only that segment forms the grid.
        """
        feature = ctx.modulated_input
        grid_seq_len = getattr(ctx, "grid_seq_len", None)
        if grid_seq_len is not None and 0 < grid_seq_len < feature.shape[1]:
            feature = feature[:, :grid_seq_len]
        return feature

    @staticmethod
    def _run_full_stack(ctx: Any) -> None:
        """Run the full block stack in place. Klein's full runner (dual- plus
        single-stream stages) comes from ``extra_states`` because its
        ``run_transformer_blocks`` covers only the dual-stream blocks.
        """
        extra_states = getattr(ctx, "extra_states", None)
        if extra_states and "run_flux2_full_transformer_with_single" in extra_states:
            ctx.hidden_states, ctx.encoder_hidden_states = extra_states["run_flux2_full_transformer_with_single"](
                ctx.hidden_states, ctx.encoder_hidden_states
            )
            return
        outputs = ctx.run_transformer_blocks()
        ctx.hidden_states = outputs[0]
        if len(outputs) > 1 and ctx.encoder_hidden_states is not None:
            ctx.encoder_hidden_states = outputs[1]

    def _run_uncached(self, ctx: Any) -> Any:
        self._run_full_stack(ctx)
        self._forward_cnt += 1
        return ctx.postprocess(ctx.hidden_states)

    def _sequence_parallel_active(self) -> bool:
        try:
            if get_sequence_parallel_world_size() <= 1:
                return False
        except Exception:
            return False
        if not self._sp_warned:
            self._sp_warned = True
            logger.warning(
                "SeaCache is disabled under sequence parallelism: its 2-D filter needs "
                "the full latent grid, but each rank holds a slice of rows. Running uncached."
            )
        return True

    def _maybe_log_summary(self, state: SeaCacheState, cache_branch: str) -> None:
        if self.num_inference_steps is None or state.cnt != self.num_inference_steps:
            return
        total = state.real_steps + state.skipped_steps
        if total == 0:
            return
        logger.info(
            "[SeaCache] %s branch: %d/%d steps refreshed (refresh ratio %.2f), sea_thresh=%.3f",
            cache_branch,
            state.real_steps,
            total,
            state.real_steps / total,
            self.config.sea_thresh,
        )

    def reset_state(self, module: torch.nn.Module) -> torch.nn.Module:
        self.state_manager.reset()
        self._forward_cnt = 0
        return module


def apply_seacache_hook(module: torch.nn.Module, config: SeaCacheConfig) -> None:
    """Register a SeaCacheHook that intercepts the module's forward pass."""
    registry = HookRegistry.get_or_create(module)
    hook = SeaCacheHook(config)
    registry.register_hook(SeaCacheHook._HOOK_NAME, hook)
