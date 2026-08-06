# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
cache-dit integration backend for vllm-omni.

This module provides a CacheDiTBackend class to enable cache-dit acceleration on diffusion
pipelines in vllm-omni, supporting both single and dual-transformer architectures.
"""

import functools
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any, Optional, TypeAlias

import cache_dit
import torch
from cache_dit import (
    BlockAdapter,
    CalibratorConfig,
    DBCacheConfig,
    ForwardPattern,
    ParamsModifier,
    TaylorSeerCalibratorConfig,
)
from cache_dit.caching.block_adapters import FakeDiffusionPipeline
from cache_dit.caching.cache_adapters.cache_adapter import CachedAdapter
from cache_dit.caching.cache_blocks.pattern_0_1_2 import CachedBlocks_Pattern_0_1_2
from cache_dit.caching.cache_blocks.pattern_3_4_5 import CachedBlocks_Pattern_3_4_5
from cache_dit.caching.cache_contexts import BasicCacheConfig
from cache_dit.caching.cache_contexts.cache_manager import CachedContextManager
from vllm.logger import init_logger

from vllm_omni.diffusion.cache.base import CacheBackend
from vllm_omni.diffusion.data import DiffusionCacheConfig, OmniDiffusionConfig

logger = init_logger(__name__)

RefreshCacheContextFunc: TypeAlias = Callable[[Any, int, bool], None]


@dataclass
class CacheDiTAdapterConfig:
    """Config for creating a Cache DiT's block adapter; to enable CacheDiT,
    most models just need to define an instance of this class as a class
    var in the DiT.
    """

    block_forward_patterns: dict[str, ForwardPattern]
    has_separate_cfg: bool = False
    cached_adapter_cls: type[CachedAdapter] | None = None
    check_forward_pattern: bool = True


# Registry of custom cache-dit enablers for specific models
# Maps pipeline names to their cache-dit enablement functions
CUSTOM_DIT_ENABLERS: dict[str, Callable] = {}


# Small helper to centralize cache-dit summaries.
def cache_summary(pipeline: Any, details: bool = True) -> None:
    if hasattr(pipeline, "transformer"):
        cache_dit.summary(pipeline.transformer, details=details)
    if hasattr(pipeline, "transformer_2"):
        cache_dit.summary(pipeline.transformer_2, details=details)
    if not hasattr(pipeline, "transformer") and not hasattr(pipeline, "transformer_2"):
        logger.warning("CacheDiT summary failed; this pipeline has no defined transformer attribute")


def default_get_pipeline_transformer(pipeline: Any) -> Any:
    return pipeline.transformer


def build_cache_context_refresh(
    cache_config: DiffusionCacheConfig,
    get_pipeline_transformer: Callable[[Any], Any] = default_get_pipeline_transformer,
) -> RefreshCacheContextFunc:
    """Build the cache context refresh func for a single Transformer."""

    def refresh_cache_context(pipeline: Any, num_inference_steps: int, verbose: bool = True) -> None:
        """Refresh cache context for the transformer with new num_inference_steps.

        Args:
            pipeline: The pipeline instance.
            num_inference_steps: New number of inference steps.
        """
        transformer = get_pipeline_transformer(pipeline)

        # Bypass SCM for step counts that don't support predefined masks (e.g., vLLM's 1-step dummy run)
        scm_supported_steps = num_inference_steps >= 8 or num_inference_steps in (4, 6)

        if cache_config.scm_steps_mask_policy is None or not scm_supported_steps:
            cache_dit.refresh_context(transformer, num_inference_steps=num_inference_steps, verbose=verbose)
        else:
            cache_dit.refresh_context(
                transformer,
                cache_config=DBCacheConfig().reset(
                    num_inference_steps=num_inference_steps,
                    steps_computation_mask=cache_dit.steps_mask(
                        mask_policy=cache_config.scm_steps_mask_policy,
                        total_steps=num_inference_steps,
                    ),
                    steps_computation_policy=cache_config.scm_steps_policy,
                ),
                verbose=verbose,
            )

    return refresh_cache_context


def _resolve_calibrator_config(cache_config: DiffusionCacheConfig) -> CalibratorConfig | None:
    """Resolves the Calibrator subconfig For DiT cache."""
    calibrator = None
    if cache_config.enable_taylorseer:
        taylorseer_order = cache_config.taylorseer_order
        calibrator = TaylorSeerCalibratorConfig(taylorseer_order=taylorseer_order)
        logger.info(f"TaylorSeer enabled with order={taylorseer_order}")
    # In the future, more calibrators will likely be added to DiT Cache,
    # e.g., focal; handle them generically here.
    return calibrator


def _build_db_cache_config(cache_config: DiffusionCacheConfig) -> DBCacheConfig:
    """Build DBCacheConfig with optional SCM (Step Computation Masking) support.

    Args:
        cache_config: DiffusionCacheConfig instance.

    Returns:
        DBCacheConfig instance with SCM support if configured.
    """
    return DBCacheConfig(
        # we will refresh the context when we get num_inference_steps in the first inference request
        num_inference_steps=None,
        Fn_compute_blocks=cache_config.Fn_compute_blocks,
        Bn_compute_blocks=cache_config.Bn_compute_blocks,
        max_warmup_steps=cache_config.max_warmup_steps,
        max_cached_steps=cache_config.max_cached_steps,
        max_continuous_cached_steps=cache_config.max_continuous_cached_steps,
        residual_diff_threshold=cache_config.residual_diff_threshold,
        force_refresh_step_hint=cache_config.force_refresh_step_hint,
        force_refresh_step_policy=cache_config.force_refresh_step_policy,
    )


def enable_cache_for_dit(
    pipeline: Any,
    cache_config: Any,
    block_adapter: BlockAdapter | None = None,
    adapter_cls: type[CachedAdapter] | None = None,
) -> RefreshCacheContextFunc:
    """Enable cache-dit for regular single-transformer DiT models.

    Args:
        pipeline: The diffusion pipeline instance.
        cache_config: DiffusionCacheConfig instance with cache configuration.
        block_adapter: Custom block adapters for specific model architectures.
        adapter_cls: Custom cached adapter class for specific model architectures.

    Returns:
        A refresh function that can be called to update cache context with new num_inference_steps.
    """
    # Build DBCacheConfig with optional SCM support
    db_cache_config = _build_db_cache_config(cache_config)

    # Build calibrator config if TaylorSeer is enabled
    calibrator_config = _resolve_calibrator_config(cache_config)

    logger.info(
        f"Enabling cache-dit on transformer: "
        f"Fn={db_cache_config.Fn_compute_blocks}, "
        f"Bn={db_cache_config.Bn_compute_blocks}, "
        f"W={db_cache_config.max_warmup_steps}, "
    )

    # Enable cache-dit on the transformer
    transformer = default_get_pipeline_transformer(pipeline)

    # If we have a custom cached adapter subclass, call apply directly
    if adapter_cls is not None:
        adapter_cls.apply(
            transformer if block_adapter is None else block_adapter,
            cache_config=db_cache_config,
            calibrator_config=calibrator_config,
        )
    else:
        # Otherwise, call enable cache, which will call CachedAdapter.apply for us
        cache_dit.enable_cache(
            transformer if block_adapter is None else block_adapter,
            cache_config=db_cache_config,
            calibrator_config=calibrator_config,
        )

    return build_cache_context_refresh(cache_config)


### Complex / custom enablers for DiT cache
# NOTE (Alex): This case is rare; you should only really need to do this if you have a dual transformer
# architecture, since it hasn't been handled generically yet, or if the model class has unique attributes
# that it sets during Cache DiT enablement.
#
# For the vast majority of models, you should only need to add a _cache_dit_adapter_config attribute
# to the Transformer class, which controls the forward pattern, whether or not we have separate CFG,
# and so on.


# from https://github.com/vipshop/cache-dit/pull/542
def _split_wan22_inference_steps(pipeline, num_inference_steps: int) -> tuple[int, int]:
    """Split inference steps into high-noise and low-noise steps for Wan2.2.

    This is an internal helper function specific to Wan2.2's dual-transformer
    architecture that uses boundary_ratio to determine the split point.

    Args:
        num_inference_steps: Total number of inference steps.

    Returns:
        A tuple of (num_high_noise_steps, num_low_noise_steps).
    """
    if pipeline.boundary_ratio is not None:
        boundary_timestep = pipeline.boundary_ratio * pipeline.scheduler.config.num_train_timesteps
    else:
        boundary_timestep = None

    # Set timesteps to calculate the split
    device = next(pipeline.transformer.parameters()).device
    pipeline.scheduler.set_timesteps(num_inference_steps, device=device)

    timesteps = pipeline.scheduler.timesteps
    num_high_noise_steps = 0  # high-noise steps for transformer
    for t in timesteps:
        if boundary_timestep is None or t >= boundary_timestep:
            num_high_noise_steps += 1
    # low-noise steps for transformer_2
    num_low_noise_steps = num_inference_steps - num_high_noise_steps
    return num_high_noise_steps, num_low_noise_steps


def enable_cache_for_wan22(pipeline: Any, cache_config: Any) -> RefreshCacheContextFunc:
    """Enable cache-dit for Wan2.2 single or dual-transformer architecture.

    Wan2.2 can use single or dual transformers (transformer and transformer_2) that need
    to be enabled using BlockAdapter.

    Args:
        pipeline: The Wan2.2 pipeline instance.
        cache_config: DiffusionCacheConfig instance with cache configuration.

    Returns:
        A refresh function that can be called to update cache context with new num_inference_steps.
    """
    # Build DBCacheConfig with optional SCM support
    db_cache_config = _build_db_cache_config(cache_config)
    calibrator_config = _resolve_calibrator_config(cache_config)

    if getattr(pipeline, "transformer_2", None) is None:
        logger.info("transformer_2 not found, enabling cache-dit for single transformer mode")
        cache_dit.enable_cache(
            BlockAdapter(
                transformer=pipeline.transformer,
                # For VACE, cache only the main denoising blocks. The
                # conditioning branch (vace_blocks) has a different forward
                # contract and produces per-step hints from the current latent
                # plus vace_context; keeping it outside CacheDiT preserves the
                # control signal while still accelerating the repeated backbone.
                blocks=[pipeline.transformer.blocks],
                forward_pattern=[ForwardPattern.Pattern_2],
                params_modifiers=[
                    ParamsModifier(cache_config=db_cache_config, calibrator_config=calibrator_config),
                ],
                has_separate_cfg=True,
            ),
            cache_config=db_cache_config,
            calibrator_config=calibrator_config,
        )
        return build_cache_context_refresh(cache_config)

    cache_dit.enable_cache(
        BlockAdapter(
            transformer=[
                pipeline.transformer,
                pipeline.transformer_2,
            ],
            blocks=[
                # See the single-transformer branch above: VACE conditioning
                # blocks are intentionally recomputed each step and are not
                # wrapped by CacheDiT's main-block Pattern_2 adapter.
                pipeline.transformer.blocks,
                pipeline.transformer_2.blocks,
            ],
            forward_pattern=[
                ForwardPattern.Pattern_2,
                ForwardPattern.Pattern_2,
            ],
            params_modifiers=[
                # high-noise transformer only have 30% steps
                ParamsModifier(
                    cache_config=DBCacheConfig().reset(
                        max_warmup_steps=cache_config.max_warmup_steps,
                        max_cached_steps=cache_config.max_cached_steps,
                    ),
                    calibrator_config=calibrator_config,
                ),
                ParamsModifier(
                    cache_config=DBCacheConfig().reset(
                        max_warmup_steps=2,
                        max_cached_steps=20,
                    ),
                    calibrator_config=calibrator_config,
                ),
            ],
            has_separate_cfg=True,
        ),
        cache_config=db_cache_config,
        calibrator_config=calibrator_config,
    )

    refresh_trans_one = build_cache_context_refresh(cache_config)
    refresh_trans_two = build_cache_context_refresh(cache_config, lambda pipeline: pipeline.transformer_2)

    def refresh_cache_context(pipeline: Any, num_inference_steps: int, verbose: bool = True) -> None:
        """Refresh cache context for both transformers with new num_inference_steps.

        Args:
            pipeline: The Wan2.2 pipeline instance.
            num_inference_steps: New number of inference steps.
        """
        num_high_noise_steps, num_low_noise_steps = _split_wan22_inference_steps(pipeline, num_inference_steps)
        refresh_trans_one(pipeline, num_high_noise_steps, verbose)
        refresh_trans_two(pipeline, num_low_noise_steps, verbose)

    return refresh_cache_context


def enable_cache_for_wan22_s2v(pipeline: Any, cache_config: Any) -> RefreshCacheContextFunc:
    """Enable cache-dit for Wan2.2 S2V.

    S2V uses a single transformer, but unlike the other Wan2.2 variants its
    block loop calls each block as ``block(hidden_states, **kwargs)`` and keeps
    the timestep modulation state in ``e`` rather than a second positional
    tensor. CacheDiT Pattern_3 matches that contract: cache hidden states only
    and pass the remaining conditioning through kwargs unchanged.

    The S2V transformer has an ``after_transformer_block`` method that injects
    audio embeddings after specific layers. The cached blocks wrapper
    (Wan22S2VCachedBlocks._run_block) calls the original internally, so we
    permanently replace it with a no-op on the transformer to prevent double
    injection from the main forward loop.
    """
    db_cache_config = _build_db_cache_config(cache_config)
    calibrator_config = None
    if cache_config.enable_taylorseer:
        taylorseer_order = cache_config.taylorseer_order
        calibrator_config = TaylorSeerCalibratorConfig(taylorseer_order=taylorseer_order)
        logger.info(f"TaylorSeer enabled with order={taylorseer_order}")

    # Save the original after_transformer_block before cache-dit wrapping
    transformer = pipeline.transformer
    if hasattr(transformer, "after_transformer_block"):
        transformer._cache_dit_original_after_transformer_block = transformer.after_transformer_block

    Wan22S2VCachedAdapter.apply(
        BlockAdapter(
            transformer=transformer,
            blocks=[transformer.blocks],
            forward_pattern=[ForwardPattern.Pattern_3],
            params_modifiers=[
                ParamsModifier(cache_config=db_cache_config, calibrator_config=calibrator_config),
            ],
            has_separate_cfg=True,
        ),
        cache_config=db_cache_config,
        calibrator_config=calibrator_config,
    )

    # Permanently replace after_transformer_block with a no-op.
    # The cached blocks wrapper (Wan22S2VCachedBlocks._run_block) already calls
    # the original via _cache_dit_original_after_transformer_block.
    def _noop_after_transformer_block(block_idx: int, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states

    transformer.after_transformer_block = _noop_after_transformer_block

    def refresh_cache_context(pipeline: Any, num_inference_steps: int, verbose: bool = True) -> None:
        """Refresh cache context for the S2V transformer."""
        if cache_config.scm_steps_mask_policy is None:
            cache_dit.refresh_context(
                pipeline.transformer,
                num_inference_steps=num_inference_steps,
                verbose=verbose,
            )
        else:
            cache_dit.refresh_context(
                pipeline.transformer,
                cache_config=DBCacheConfig().reset(
                    num_inference_steps=num_inference_steps,
                    steps_computation_mask=cache_dit.steps_mask(
                        mask_policy=cache_config.scm_steps_mask_policy,
                        total_steps=num_inference_steps,
                    ),
                    steps_computation_policy=cache_config.scm_steps_policy,
                ),
                verbose=verbose,
            )

    return refresh_cache_context


class BagelCachedContextManager(CachedContextManager):
    """
    Custom CachedContextManager for Bagel that safely handles NaiveCache objects
    (mapped to encoder_hidden_states) by skipping tensor operations on them.
    """

    @torch.compiler.disable
    def apply_cache(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        prefix: str = "Bn",
        encoder_prefix: str = "Bn_encoder",
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # Allow Bn and Fn prefix to be used for residual cache.
        if "Bn" in prefix:
            hidden_states_prev = self.get_Bn_buffer(prefix)
        else:
            hidden_states_prev = self.get_Fn_buffer(prefix)

        assert hidden_states_prev is not None, f"{prefix}_buffer must be set before"

        if self.is_cache_residual():
            hidden_states = hidden_states_prev + hidden_states
        else:
            # If cache is not residual, we use the hidden states directly
            hidden_states = hidden_states_prev

        hidden_states = hidden_states.contiguous()

        if encoder_hidden_states is not None:
            if "Bn" in encoder_prefix:
                encoder_hidden_states_prev = self.get_Bn_encoder_buffer(encoder_prefix)
            else:
                encoder_hidden_states_prev = self.get_Fn_encoder_buffer(encoder_prefix)

            if encoder_hidden_states_prev is not None:
                if self.is_encoder_cache_residual():
                    # FIX: Check if encoder_hidden_states is a tensor before adding
                    if isinstance(encoder_hidden_states, torch.Tensor) and isinstance(
                        encoder_hidden_states_prev, torch.Tensor
                    ):
                        encoder_hidden_states = encoder_hidden_states_prev + encoder_hidden_states
                else:
                    # If encoder cache is not residual, we use the encoder hidden states directly
                    encoder_hidden_states = encoder_hidden_states_prev

            # FIX: Check if encoder_hidden_states is a tensor before calling contiguous
            if isinstance(encoder_hidden_states, torch.Tensor):
                encoder_hidden_states = encoder_hidden_states.contiguous()

        return hidden_states, encoder_hidden_states


class BagelCachedBlocks(CachedBlocks_Pattern_0_1_2):
    """
    Custom CachedBlocks for Bagel that safely handles NaiveCache objects
    by adding isinstance checks in call_Mn_blocks and compute_or_prune.
    """

    def call_Mn_blocks(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        *args,
        **kwargs,
    ):
        original_hidden_states = hidden_states
        original_encoder_hidden_states = encoder_hidden_states
        for block in self._Mn_blocks():
            hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                *args,
                **kwargs,
            )
            hidden_states, encoder_hidden_states = self._process_block_outputs(hidden_states, encoder_hidden_states)

        # compute hidden_states residual
        hidden_states = hidden_states.contiguous()

        hidden_states_residual = hidden_states - original_hidden_states

        if (
            encoder_hidden_states is not None
            and original_encoder_hidden_states is not None
            and isinstance(encoder_hidden_states, torch.Tensor)  # FIX: Added Check
        ):
            encoder_hidden_states = encoder_hidden_states.contiguous()
            encoder_hidden_states_residual = encoder_hidden_states - original_encoder_hidden_states
        else:
            encoder_hidden_states_residual = None

        return (
            hidden_states,
            encoder_hidden_states,
            hidden_states_residual,
            encoder_hidden_states_residual,
        )

    def compute_or_prune(
        self,
        block_id: int,  # Block index in the transformer blocks
        # Below are the inputs to the block
        block,  # The transformer block to be executed
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        *args,
        **kwargs,
    ):
        # NOTE: Although Bagel likely won't use pruning, implementing safe version just in case.
        # Copy-pasted from original but adding checks.

        original_hidden_states = hidden_states
        original_encoder_hidden_states = encoder_hidden_states

        can_use_prune = self._maybe_prune(
            block_id,
            hidden_states,
            prefix=f"{self.cache_prefix}_{block_id}_Fn_original",
        )

        torch._dynamo.graph_break()
        if can_use_prune:
            self.context_manager.add_pruned_step()
            hidden_states, encoder_hidden_states = self.context_manager.apply_prune(
                hidden_states,
                encoder_hidden_states,
                prefix=(
                    f"{self.cache_prefix}_{block_id}_Bn_residual"
                    if self.context_manager.is_cache_residual()
                    else f"{self.cache_prefix}_Bn_hidden_states"
                ),
                encoder_prefix=(
                    f"{self.cache_prefix}_{block_id}_Bn_encoder_residual"
                    if self.context_manager.is_encoder_cache_residual()
                    else f"{self.cache_prefix}_{block_id}_Bn_encoder_hidden_states"
                ),
            )
            torch._dynamo.graph_break()
        else:
            # Normal steps: Compute the block and cache the residuals.
            hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                *args,
                **kwargs,
            )
            hidden_states, encoder_hidden_states = self._process_block_outputs(hidden_states, encoder_hidden_states)
            if not self._skip_prune(block_id):
                hidden_states = hidden_states.contiguous()
                hidden_states_residual = hidden_states - original_hidden_states

                if (
                    encoder_hidden_states is not None
                    and original_encoder_hidden_states is not None
                    and isinstance(encoder_hidden_states, torch.Tensor)  # FIX: Added Check
                ):
                    encoder_hidden_states = encoder_hidden_states.contiguous()
                    encoder_hidden_states_residual = encoder_hidden_states - original_encoder_hidden_states
                else:
                    encoder_hidden_states_residual = None

                self.context_manager.set_Fn_buffer(
                    original_hidden_states,
                    prefix=f"{self.cache_prefix}_{block_id}_Fn_original",
                )
                if self.context_manager.is_cache_residual():
                    self.context_manager.set_Bn_buffer(
                        hidden_states_residual,
                        prefix=f"{self.cache_prefix}_{block_id}_Bn_residual",
                    )
                else:
                    self.context_manager.set_Bn_buffer(
                        hidden_states,
                        prefix=f"{self.cache_prefix}_{block_id}_Bn_hidden_states",
                    )
                if encoder_hidden_states_residual is not None:
                    if self.context_manager.is_encoder_cache_residual():
                        self.context_manager.set_Bn_encoder_buffer(
                            encoder_hidden_states_residual,
                            prefix=f"{self.cache_prefix}_{block_id}_Bn_encoder_residual",
                        )
                    else:
                        self.context_manager.set_Bn_encoder_buffer(
                            encoder_hidden_states_residual,
                            prefix=f"{self.cache_prefix}_{block_id}_Bn_encoder_hidden_states",
                        )
            torch._dynamo.graph_break()

        return hidden_states, encoder_hidden_states


class Wan22S2VCachedBlocks(CachedBlocks_Pattern_3_4_5):
    """CacheDiT blocks wrapper that preserves S2V per-layer audio injection."""

    def _run_block(self, block_id: int, block: torch.nn.Module, hidden_states: torch.Tensor, *args, **kwargs):
        hidden_states = block(hidden_states, *args, **kwargs)
        hidden_states, new_encoder_hidden_states = self._process_block_outputs(hidden_states)
        original_after_transformer_block = getattr(
            self.transformer,
            "_cache_dit_original_after_transformer_block",
            getattr(self.transformer, "after_transformer_block", None),
        )
        if original_after_transformer_block is not None:
            hidden_states = original_after_transformer_block(block_id, hidden_states)
        return hidden_states, new_encoder_hidden_states

    def call_blocks(
        self,
        hidden_states: torch.Tensor,
        *args,
        **kwargs,
    ):
        new_encoder_hidden_states = None
        for block_id, block in enumerate(self.transformer_blocks):
            hidden_states, new_encoder_hidden_states = self._run_block(
                block_id,
                block,
                hidden_states,
                *args,
                **kwargs,
            )
        return hidden_states, new_encoder_hidden_states

    def call_Fn_blocks(
        self,
        hidden_states: torch.Tensor,
        *args,
        **kwargs,
    ):
        new_encoder_hidden_states = None
        for block_id, block in enumerate(self._Fn_blocks()):
            hidden_states, new_encoder_hidden_states = self._run_block(
                block_id,
                block,
                hidden_states,
                *args,
                **kwargs,
            )
        return hidden_states, new_encoder_hidden_states

    def call_Mn_blocks(
        self,
        hidden_states: torch.Tensor,
        *args,
        **kwargs,
    ):
        original_hidden_states = hidden_states
        new_encoder_hidden_states = None
        start_idx = self.context_manager.Fn_compute_blocks()
        for block_id, block in enumerate(self._Mn_blocks(), start=start_idx):
            hidden_states, new_encoder_hidden_states = self._run_block(
                block_id,
                block,
                hidden_states,
                *args,
                **kwargs,
            )

        hidden_states = hidden_states.contiguous()
        hidden_states_residual = hidden_states - original_hidden_states.to(hidden_states.device)

        return (
            hidden_states,
            new_encoder_hidden_states,
            hidden_states_residual,
        )

    def call_Bn_blocks(
        self,
        hidden_states: torch.Tensor,
        *args,
        **kwargs,
    ):
        new_encoder_hidden_states = None
        if self.context_manager.Bn_compute_blocks() == 0:
            return hidden_states, new_encoder_hidden_states

        start_idx = len(self.transformer_blocks) - self.context_manager.Bn_compute_blocks()
        for block_id, block in enumerate(self._Bn_blocks(), start=start_idx):
            hidden_states, new_encoder_hidden_states = self._run_block(
                block_id,
                block,
                hidden_states,
                *args,
                **kwargs,
            )

        return hidden_states, new_encoder_hidden_states


class Wan22S2VCachedAdapter(CachedAdapter):
    """CacheDiT adapter that uses Wan22S2VCachedBlocks for S2V audio injection.

    Only overrides collect_unified_blocks to use Wan22S2VCachedBlocks (which
    calls after_transformer_block per-layer internally). The base class
    mock_transformer handles the forward wrapping — after_transformer_block is
    permanently replaced with a no-op in enable_cache_for_wan22_s2v() to prevent
    double injection.
    """

    @classmethod
    def collect_unified_blocks(
        cls,
        block_adapter: BlockAdapter,
        contexts_kwargs: list[dict],
    ) -> list[dict[str, torch.nn.ModuleList]]:
        BlockAdapter.assert_normalized(block_adapter)

        total_cached_blocks: list[dict[str, torch.nn.ModuleList]] = []
        assert hasattr(block_adapter.pipe, "_context_manager")

        for i in range(len(block_adapter.transformer)):
            unified_blocks_bind_context = {}
            for j in range(len(block_adapter.blocks[i])):
                cache_config: BasicCacheConfig = contexts_kwargs[i * len(block_adapter.blocks[i]) + j]["cache_config"]
                unified_blocks_bind_context[block_adapter.unique_blocks_name[i][j]] = torch.nn.ModuleList(
                    [
                        Wan22S2VCachedBlocks(
                            block_adapter.blocks[i][j],
                            transformer=block_adapter.transformer[i],
                            forward_pattern=block_adapter.forward_pattern[i][j],
                            check_forward_pattern=block_adapter.check_forward_pattern,
                            check_num_outputs=block_adapter.check_num_outputs,
                            cache_prefix=block_adapter.blocks_name[i][j],
                            cache_context=block_adapter.unique_blocks_name[i][j],
                            context_manager=block_adapter.pipe._context_manager,
                            cache_type=cache_config.cache_type,
                        )
                    ]
                )

            total_cached_blocks.append(unified_blocks_bind_context)

        return total_cached_blocks


class BagelCachedAdapter(CachedAdapter):
    """
    Custom CachedAdapter for Bagel that uses BagelCachedContextManager and BagelCachedBlocks.
    """

    @classmethod
    def create_context(
        cls,
        block_adapter: BlockAdapter,
        **context_kwargs,
    ) -> tuple[list[str], list[dict[str, Any]]]:
        # Override to use BagelCachedContextManager

        BlockAdapter.assert_normalized(block_adapter)

        if BlockAdapter.is_cached(block_adapter.pipe):
            return block_adapter.pipe

        # Check context_kwargs
        context_kwargs = cls.check_context_kwargs(block_adapter, **context_kwargs)

        # Each Pipeline should have it's own context manager instance.
        cache_config: BasicCacheConfig = context_kwargs.get("cache_config", None)
        assert cache_config is not None, "cache_config can not be None."

        # Apply cache on pipeline: wrap cache context
        pipe_cls_name = block_adapter.pipe.__class__.__name__

        # USE CUSTOM CONTEXT MANAGER
        context_manager = BagelCachedContextManager(
            name=f"{pipe_cls_name}_{hash(id(block_adapter.pipe))}",
            persistent_context=isinstance(block_adapter.pipe, FakeDiffusionPipeline),
        )

        flatten_contexts, contexts_kwargs = cls.modify_context_params(block_adapter, **context_kwargs)

        block_adapter.pipe._context_manager = context_manager  # instance level

        if not context_manager.persistent_context:
            original_call = block_adapter.pipe.__class__.__call__

            @functools.wraps(original_call)
            def new_call(self, *args, **kwargs):
                with ExitStack() as stack:
                    # cache context will be reset for each pipe inference
                    for context_name, context_kwargs in zip(flatten_contexts, contexts_kwargs):
                        stack.enter_context(
                            context_manager.enter_context(
                                context_manager.reset_context(
                                    context_name,
                                    **context_kwargs,
                                ),
                            )
                        )
                    outputs = original_call(self, *args, **kwargs)
                    cls.apply_stats_hooks(block_adapter)
                    return outputs

            block_adapter.pipe.__class__.__call__ = new_call
            block_adapter.pipe.__class__._original_call = original_call

        else:
            # Init persistent cache context for transformer
            for context_name, context_kwargs in zip(flatten_contexts, contexts_kwargs):
                context_manager.reset_context(
                    context_name,
                    **context_kwargs,
                )

        block_adapter.pipe.__class__._is_cached = True

        cls.apply_params_hooks(block_adapter, contexts_kwargs)

        return flatten_contexts, contexts_kwargs

    @classmethod
    def collect_unified_blocks(
        cls,
        block_adapter: BlockAdapter,
        contexts_kwargs: list[dict],
    ) -> list[dict[str, torch.nn.ModuleList]]:
        # Override to use BagelCachedBlocks

        BlockAdapter.assert_normalized(block_adapter)

        total_cached_blocks: list[dict[str, torch.nn.ModuleList]] = []
        assert hasattr(block_adapter.pipe, "_context_manager")
        # Skipping isinstance check for ContextManager._supported_managers to avoid import issues

        for i in range(len(block_adapter.transformer)):
            unified_blocks_bind_context = {}
            for j in range(len(block_adapter.blocks[i])):
                cache_config: BasicCacheConfig = contexts_kwargs[i * len(block_adapter.blocks[i]) + j]["cache_config"]

                # Directly instantiate BagelCachedBlocks
                unified_blocks_bind_context[block_adapter.unique_blocks_name[i][j]] = torch.nn.ModuleList(
                    [
                        BagelCachedBlocks(
                            # 0. Transformer blocks configuration
                            block_adapter.blocks[i][j],
                            transformer=block_adapter.transformer[i],
                            forward_pattern=block_adapter.forward_pattern[i][j],
                            check_forward_pattern=block_adapter.check_forward_pattern,
                            check_num_outputs=block_adapter.check_num_outputs,
                            # 1. Cache/Prune context configuration
                            cache_prefix=block_adapter.blocks_name[i][j],
                            cache_context=block_adapter.unique_blocks_name[i][j],
                            context_manager=block_adapter.pipe._context_manager,
                            cache_type=cache_config.cache_type,
                        )
                    ]
                )

            total_cached_blocks.append(unified_blocks_bind_context)

        return total_cached_blocks


class SensenovaCachedBlocks(CachedBlocks_Pattern_3_4_5):
    """
    Custom CachedBlocks for SenseNova-U1 that only caches image-token hidden
    states during denoising.
    """

    @classmethod
    def _is_denoising_call(cls, kwargs: dict[str, Any]) -> bool:
        if kwargs.get("cache_dit_skip", False):
            return False

        # Prefix/text forwards either omit image_gen_indicators or update the
        # DynamicCache. Denoising forwards are gen-only and use update_cache=False.
        if kwargs.get("update_cache", True):
            return False

        exist_gen = kwargs.get("exist_gen")
        exist_und = kwargs.get("exist_und")
        if exist_gen is None or exist_und is None:
            image_gen_indicators = kwargs.get("image_gen_indicators")
            if image_gen_indicators is None:
                return False
            exist_gen = image_gen_indicators.any().item()
            exist_und = (~image_gen_indicators).any().item()

        return exist_gen and not exist_und

    @staticmethod
    def _strip_cache_only_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
        kwargs = dict(kwargs)
        kwargs.pop("cache_dit_skip", None)
        return kwargs

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs):
        block_kwargs = self._strip_cache_only_kwargs(kwargs)
        if not self._is_denoising_call(kwargs):
            hidden_states, new_encoder_hidden_states = self.call_blocks(
                hidden_states,
                *args,
                **block_kwargs,
            )
            return self._process_forward_outputs(hidden_states, new_encoder_hidden_states)

        return super().forward(hidden_states, *args, **block_kwargs)


class SensenovaCachedAdapter(CachedAdapter):
    """Custom CachedAdapter for SenseNova-U1 that uses SensenovaCachedBlocks."""

    @classmethod
    def collect_unified_blocks(
        cls,
        block_adapter: BlockAdapter,
        contexts_kwargs: list[dict],
    ) -> list[dict[str, torch.nn.ModuleList]]:
        BlockAdapter.assert_normalized(block_adapter)

        total_cached_blocks: list[dict[str, torch.nn.ModuleList]] = []
        assert hasattr(block_adapter.pipe, "_context_manager")

        for i in range(len(block_adapter.transformer)):
            unified_blocks_bind_context = {}
            for j in range(len(block_adapter.blocks[i])):
                cache_config: BasicCacheConfig = contexts_kwargs[i * len(block_adapter.blocks[i]) + j]["cache_config"]
                unified_blocks_bind_context[block_adapter.unique_blocks_name[i][j]] = torch.nn.ModuleList(
                    [
                        SensenovaCachedBlocks(
                            # 0. Transformer blocks configuration
                            block_adapter.blocks[i][j],
                            transformer=block_adapter.transformer[i],
                            forward_pattern=block_adapter.forward_pattern[i][j],
                            check_forward_pattern=block_adapter.check_forward_pattern,
                            check_num_outputs=block_adapter.check_num_outputs,
                            # 1. Cache/Prune context configuration
                            cache_prefix=block_adapter.blocks_name[i][j],
                            cache_context=block_adapter.unique_blocks_name[i][j],
                            context_manager=block_adapter.pipe._context_manager,
                            cache_type=cache_config.cache_type,
                        )
                    ]
                )

            total_cached_blocks.append(unified_blocks_bind_context)

        return total_cached_blocks


def enable_cache_for_cosmos3(pipeline: Any, cache_config: Any) -> RefreshCacheContextFunc:
    """Enable cache-dit for Cosmos3.

    Cosmos3 has a dual-pathway architecture (UND + GEN) but only the GEN
    pathway (``gen_layers``) runs at every denoising step.  The UND pathway
    computes once and its K/V are cached by the pipeline itself; no cache-dit
    needed there.  We wrap only ``gen_layers`` via ``BlockAdapter``.

    Args:
        pipeline: The Cosmos3 pipeline instance.
        cache_config: DiffusionCacheConfig instance with cache configuration.

    Returns:
        A refresh function that can be called to update cache context with new num_inference_steps.
    """
    # The T2I denoising loop skips the unconditional forward outside the
    # guidance interval as a speed optimization. cache-dit distinguishes the
    # conditional vs unconditional passes purely by transformer-forward parity
    # (has_separate_cfg=True above), so that skip would desync its per-generation
    # step accounting. Still do both cond/uncond CFG steps when cache-dit is active.
    # CFG is instead neutralized via scale=1.0 outside the interval.
    pipeline._cache_dit_requires_paired_cfg = True
    block_adapter = CacheDiTBackend.maybe_build_block_adapter(pipeline)
    return enable_cache_for_dit(pipeline, cache_config, block_adapter)


# Register custom cache-dit enablers after function definitions
CUSTOM_DIT_ENABLERS.update(
    {
        "Wan22Pipeline": enable_cache_for_wan22,
        "Wan22I2VPipeline": enable_cache_for_wan22,
        "Wan22TI2VPipeline": enable_cache_for_wan22,
        "Wan22VACEPipeline": enable_cache_for_wan22,
        "Wan22S2VPipeline": enable_cache_for_wan22_s2v,
        "Cosmos3OmniDiffusersPipeline": enable_cache_for_cosmos3,
    }
)


class CacheDiTBackend(CacheBackend):
    """Backend class for cache-dit acceleration on diffusion pipelines.

    This class implements cache-dit acceleration (DBCache, SCM, TaylorSeer) using
    the cache-dit library. It inherits from CacheBackend and provides a unified
    interface for managing cache-dit acceleration on diffusion models.

    Attributes:
        config: Cache configuration (DiffusionCacheConfig instance), inherited from CacheBackend.
        enabled: Whether cache-dit is enabled on this pipeline, inherited from CacheBackend.
        _refresh_func: Internal refresh function for updating cache context.
        _last_num_inference_steps: Last num_inference_steps used for refresh optimization.
    """

    def __init__(self, cache_config: Any = None):
        """Initialize the cache-dit backend.

        Args:
            cache_config: Cache configuration (DiffusionCacheConfig instance, dict, or None).
                         If None or empty, uses default DiffusionCacheConfig().
        """
        # Use default config if cache_config is not provided or is empty
        if cache_config is None:
            config = DiffusionCacheConfig()
        elif isinstance(cache_config, dict):
            # Convert dict to DiffusionCacheConfig, using defaults for missing keys
            config = DiffusionCacheConfig.from_dict(cache_config)
        else:
            config = cache_config

        # Initialize base class with normalized config
        super().__init__(config)

        # Cache-dit specific attributes
        self._refresh_func: Callable[[Any, int, bool], None] | None = None
        self._last_num_inference_steps: int | None = None

    @staticmethod
    def maybe_build_block_adapter(pipeline) -> BlockAdapter | None:
        """If a module defines `_cache_dit_adapter_config`, build the corresponding
        block adapter.
        """
        transformer = default_get_pipeline_transformer(pipeline)

        adapter_cfg: CacheDiTAdapterConfig | None = getattr(transformer, "_cache_dit_adapter_config", None)
        if adapter_cfg is None:
            return None

        block_attrs, forward_pattern = zip(*(adapter_cfg.block_forward_patterns).items())
        missing_attrs = [block_attr for block_attr in block_attrs if not hasattr(transformer, block_attr)]

        if missing_attrs:
            logger.warning("Missing Cache DiT block attributes: %s", missing_attrs)

        block_adapter = BlockAdapter(
            transformer=transformer,
            blocks=[getattr(transformer, block_attr) for block_attr in block_attrs],
            forward_pattern=list(forward_pattern),
            has_separate_cfg=adapter_cfg.has_separate_cfg,
            check_forward_pattern=adapter_cfg.check_forward_pattern,
        )
        return block_adapter

    @staticmethod
    def maybe_get_cached_adapter_cls(pipeline) -> type[CachedAdapter] | None:
        """If a module has a custom cached adapter type registered, e.g., SenseNova, retrieve it
        from the transformer's CacheDiTAdapterConfig."""
        transformer = default_get_pipeline_transformer(pipeline)

        adapter_cfg: CacheDiTAdapterConfig | None = getattr(transformer, "_cache_dit_adapter_config", None)
        if adapter_cfg is None:
            return None
        return adapter_cfg.cached_adapter_cls

    def enable(self, pipeline: Any) -> None:
        """Enable cache-dit on the pipeline if configured.

        This method applies cache-dit acceleration to the appropriate transformer(s)
        in the pipeline. It handles both single-transformer and dual-transformer
        architectures (e.g., Wan2.2).

        Args:
            pipeline: The diffusion pipeline instance.
        """

        # Extract pipeline name from pipeline
        pipeline_name = pipeline.__class__.__name__
        # Check if this model has a custom cache-dit enabler
        if pipeline_name in CUSTOM_DIT_ENABLERS:
            logger.info(f"Using custom cache-dit enabler for model: {pipeline_name}")
            self._refresh_func = CUSTOM_DIT_ENABLERS[pipeline_name](pipeline, self.config)
        else:
            # Common case; either the model doesn't explicitly support dit cache yet,
            # Or it defines its _cache_dit_adapter_config, which describes how we should
            # create its block adapter.
            block_adapter = self.maybe_build_block_adapter(pipeline)
            adapter_cls = self.maybe_get_cached_adapter_cls(pipeline)
            self._refresh_func = enable_cache_for_dit(pipeline, self.config, block_adapter, adapter_cls)

        self.enabled = True
        logger.info(f"Cache-dit enabled successfully on {pipeline_name}")

    def refresh(self, pipeline: Any, num_inference_steps: int, verbose: bool = True) -> None:
        """Refresh cache context with new num_inference_steps.

        This method updates the cache context when num_inference_steps changes
        during inference. For dual-transformer models (e.g., Wan2.2), it automatically
        splits the steps based on boundary_ratio.

        Args:
            pipeline: The diffusion pipeline instance.
            num_inference_steps: New number of inference steps.
            verbose: Whether to log refresh operations.
        """
        if not self.enabled or self._refresh_func is None:
            logger.warning("Cache-dit is not enabled. Cannot refresh cache context.")
            return

        # Only refresh if num_inference_steps has changed
        if self._last_num_inference_steps is None or num_inference_steps != self._last_num_inference_steps:
            if verbose:
                logger.info(f"Refreshing cache context for transformer with num_inference_steps: {num_inference_steps}")
            self._refresh_func(pipeline, num_inference_steps, verbose)
            self._last_num_inference_steps = num_inference_steps

    def is_enabled(self) -> bool:
        """Check if cache-dit is enabled on this pipeline.

        Returns:
            True if cache-dit is enabled, False otherwise.
        """
        return self.enabled


def may_enable_cache_dit(pipeline: Any, od_config: OmniDiffusionConfig) -> Optional["CacheDiTBackend"]:
    """Enable cache-dit on the pipeline if configured (convenience function).

    This is a convenience function that creates and enables a CacheDiTBackend.
    For new code, consider using CacheDiTBackend directly.

    Args:
        pipeline: The diffusion pipeline instance.
        od_config: OmniDiffusionConfig with cache configuration.

    Returns:
        A CacheDiTBackend instance if cache-dit is enabled, None otherwise.
    """
    if od_config.cache_backend != "cache-dit" or not od_config.cache_config:
        return None

    backend = CacheDiTBackend(od_config.cache_config)
    backend.enable(pipeline)
    return backend if backend.is_enabled() else None
