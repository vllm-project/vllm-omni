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
from typing import Any, Optional

import cache_dit
import torch
from cache_dit import BlockAdapter, DBCacheConfig, ForwardPattern, ParamsModifier, TaylorSeerCalibratorConfig
from cache_dit.caching.block_adapters import FakeDiffusionPipeline
from cache_dit.caching.cache_adapters.cache_adapter import CachedAdapter
from cache_dit.caching.cache_blocks.pattern_0_1_2 import CachedBlocks_Pattern_0_1_2
from cache_dit.caching.cache_contexts import BasicCacheConfig
from cache_dit.caching.cache_contexts.cache_manager import CachedContextManager
from vllm.logger import init_logger

from vllm_omni.diffusion.cache.base import CacheBackend
from vllm_omni.diffusion.data import DiffusionCacheConfig, OmniDiffusionConfig

logger = init_logger(__name__)


# Registry of custom cache-dit enablers for specific models
# Maps pipeline names to their cache-dit enablement functions
# Models in this registry require custom handling (e.g., dual-transformer architectures)
# Will be populated after function definitions
CUSTOM_DIT_ENABLERS: dict[str, Callable] = {}


def _build_db_cache_config(cache_config: Any) -> DBCacheConfig:
    """Build DBCacheConfig with optional SCM (Step Computation Masking) support.

    Args:
        cache_config: DiffusionCacheConfig instance.

    Returns:
        DBCacheConfig instance with SCM support if configured.
    """

    return DBCacheConfig(
        # we will refresh the context when gets num_inference_steps in the first inference request
        num_inference_steps=None,
        Fn_compute_blocks=cache_config.Fn_compute_blocks,
        Bn_compute_blocks=cache_config.Bn_compute_blocks,
        max_warmup_steps=cache_config.max_warmup_steps,
        max_cached_steps=cache_config.max_cached_steps,
        max_continuous_cached_steps=cache_config.max_continuous_cached_steps,
        residual_diff_threshold=cache_config.residual_diff_threshold,
    )


def enable_cache_for_wan22(pipeline: Any, cache_config: Any) -> Callable[[int], None]:
    """Enable cache-dit for Wan2.2 dual-transformer architecture.

    Wan2.2 uses two transformers (transformer and transformer_2) that need
    to be enabled together using BlockAdapter.

    Args:
        pipeline: The Wan2.2 pipeline instance.
        cache_config: DiffusionCacheConfig instance with cache configuration.

    Returns:
        A refresh function that can be called to update cache context with new num_inference_steps.
    """

    # Build DBCacheConfig for primary transformer
    primary_cache_config = _build_db_cache_config(cache_config)

    # FIXME: secondary cache shares the same config with primary cache for now,
    # but we should support different config for secondary transformer in the future
    # Build DBCacheConfig for secondary transformer (can use same or different config)
    secondary_cache_config = _build_db_cache_config(cache_config)

    # Build calibrator configs if TaylorSeer is enabled
    primary_calibrator = None
    secondary_calibrator = None
    if cache_config.enable_taylorseer:
        taylorseer_order = cache_config.taylorseer_order
        primary_calibrator = TaylorSeerCalibratorConfig(taylorseer_order=taylorseer_order)
        secondary_calibrator = TaylorSeerCalibratorConfig(taylorseer_order=taylorseer_order)
        logger.info(f"TaylorSeer enabled with order={taylorseer_order}")

    # Build ParamsModifier for each transformer
    primary_modifier = ParamsModifier(
        cache_config=primary_cache_config,
        calibrator_config=primary_calibrator,
    )
    secondary_modifier = ParamsModifier(
        cache_config=secondary_cache_config,
        calibrator_config=secondary_calibrator,
    )

    logger.info(
        "Enabling cache-dit on Wan2.2 dual transformers with BlockAdapter: "
        f"Fn={primary_cache_config.Fn_compute_blocks}, "
        f"Bn={primary_cache_config.Bn_compute_blocks}, "
        f"W={primary_cache_config.max_warmup_steps}, "
    )

    transformer = pipeline.transformer
    transformer_2 = pipeline.transformer_2
    transformer_blocks = transformer.blocks
    transformer_2_blocks = transformer_2.blocks
    # Enable cache-dit using BlockAdapter for both transformers simultaneously
    cache_dit.enable_cache(
        BlockAdapter(
            transformer=[transformer, transformer_2],
            blocks=[transformer_blocks, transformer_2_blocks],
            forward_pattern=[ForwardPattern.Pattern_2, ForwardPattern.Pattern_2],
            params_modifiers=[primary_modifier, secondary_modifier],
            has_separate_cfg=True,
        ),
    )

    # from https://github.com/vipshop/cache-dit/pull/542
    def _split_inference_steps(num_inference_steps: int) -> tuple[int, int]:
        """Split inference steps into high-noise and low-noise steps for Wan2.2.

        This is an internal helper function specific to Wan2.2's dual-transformer
        architecture that uses boundary_ratio to determine the split point.

        Args:
            num_inference_steps: Total number of inference steps.

        Returns:
            A tuple of (num_high_noise_steps, num_low_noise_steps).
        """
        if pipeline.config.boundary_ratio is not None:
            boundary_timestep = pipeline.config.boundary_ratio * pipeline.scheduler.config.num_train_timesteps
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

    def refresh_cache_context(pipeline: Any, num_inference_steps: int, verbose: bool = True) -> None:
        """Refresh cache context for both transformers with new num_inference_steps.

        Args:
            pipeline: The Wan2.2 pipeline instance.
            num_inference_steps: New number of inference steps.
        """
        num_high_noise_steps, num_low_noise_steps = _split_inference_steps(num_inference_steps)
        # Refresh context for high-noise transformer
        if cache_config.scm_steps_mask_policy is None:
            cache_dit.refresh_context(pipeline.transformer, num_inference_steps=num_high_noise_steps, verbose=verbose)
        else:
            cache_dit.refresh_context(
                pipeline.transformer,
                cache_config=DBCacheConfig().reset(
                    num_inference_steps=num_high_noise_steps,
                    steps_computation_mask=cache_dit.steps_mask(
                        mask_policy=cache_config.scm_steps_mask_policy, total_steps=num_high_noise_steps
                    ),
                    steps_computation_policy=cache_config.scm_steps_policy,
                ),
                verbose=verbose,
            )

        # Refresh context for low-noise transformer
        if cache_config.scm_steps_mask_policy is None:
            cache_dit.refresh_context(pipeline.transformer_2, num_inference_steps=num_low_noise_steps, verbose=verbose)
        else:
            cache_dit.refresh_context(
                pipeline.transformer_2,
                cache_config=DBCacheConfig().reset(
                    num_inference_steps=num_low_noise_steps,
                    steps_computation_mask=cache_dit.steps_mask(
                        mask_policy=cache_config.scm_steps_mask_policy, total_steps=num_low_noise_steps
                    ),
                    steps_computation_policy=cache_config.scm_steps_policy,
                ),
                verbose=verbose,
            )

    return refresh_cache_context


def enable_cache_for_longcat_image(pipeline: Any, cache_config: Any) -> Callable[[int], None]:
    """Enable cache-dit for LongCatImage pipeline.

    Args:
        pipeline: The LongCatImage pipeline instance.
        cache_config: DiffusionCacheConfig instance with cache configuration.
    """
    # Build DBCacheConfig for transformer
    db_cache_config = _build_db_cache_config(cache_config)

    calibrator = None
    if cache_config.enable_taylorseer:
        taylorseer_order = cache_config.taylorseer_order
        calibrator = TaylorSeerCalibratorConfig(taylorseer_order=taylorseer_order)
        logger.info(f"TaylorSeer enabled with order={taylorseer_order}")

    # Build ParamsModifier for transformer
    modifier = ParamsModifier(
        cache_config=db_cache_config,
        calibrator_config=calibrator,
    )

    logger.info(
        f"Enabling cache-dit on LongCatImage transformer with BlockAdapter: "
        f"Fn={db_cache_config.Fn_compute_blocks}, "
        f"Bn={db_cache_config.Bn_compute_blocks}, "
        f"W={db_cache_config.max_warmup_steps}, "
    )

    # Enable cache-dit using BlockAdapter for transformer
    cache_dit.enable_cache(
        (
            BlockAdapter(
                transformer=pipeline.transformer,
                blocks=[
                    pipeline.transformer.transformer_blocks,
                    pipeline.transformer.single_transformer_blocks,
                ],
                forward_pattern=[ForwardPattern.Pattern_1, ForwardPattern.Pattern_1],
                params_modifiers=[modifier],
            )
        ),
        cache_config=db_cache_config,
    )

    def refresh_cache_context(pipeline: Any, num_inference_steps: int, verbose: bool = True) -> None:
        """Refresh cache context for the transformer with new num_inference_steps.

        Args:
            pipeline: The LongCatImage pipeline instance.
            num_inference_steps: New number of inference steps.
        """
        if cache_config.scm_steps_mask_policy is None:
            cache_dit.refresh_context(pipeline.transformer, num_inference_steps=num_inference_steps, verbose=verbose)
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


def enable_cache_for_flux(pipeline: Any, cache_config: Any) -> Callable[[int], None]:
    """Enable cache-dit for Flux dual-transformer architecture.

    Flux uses two transformers (transformer and transformer_2) that need
    to be enabled together using BlockAdapter.

    Args:
        pipeline: The Flux pipeline instance.
        cache_config: DiffusionCacheConfig instance with cache configuration.

    Returns:
        A refresh function that can be called to update cache context with new num_inference_steps.
    """
    raise NotImplementedError("cache-dit is not implemented for Flux pipeline.")


def enable_cache_for_sd3(pipeline: Any, cache_config: Any) -> Callable[[int], None]:
    """Enable cache-dit for StableDiffusion3Pipeline.

    Args:
        pipeline: The StableDiffusion3 pipeline instance.
        cache_config: DiffusionCacheConfig instance with cache configuration.
    """
    # Build DBCacheConfig for transformer
    db_cache_config = _build_db_cache_config(cache_config)

    calibrator = None
    if cache_config.enable_taylorseer:
        taylorseer_order = cache_config.taylorseer_order
        calibrator = TaylorSeerCalibratorConfig(taylorseer_order=taylorseer_order)
        logger.info(f"TaylorSeer enabled with order={taylorseer_order}")

    # Build ParamsModifier for transformer
    modifier = ParamsModifier(
        cache_config=db_cache_config,
        calibrator_config=calibrator,
    )

    logger.info(
        f"Enabling cache-dit on StableDiffusion3 transformer with BlockAdapter: "
        f"Fn={db_cache_config.Fn_compute_blocks}, "
        f"Bn={db_cache_config.Bn_compute_blocks}, "
        f"W={db_cache_config.max_warmup_steps}, "
    )

    # Enable cache-dit using BlockAdapter for transformer
    cache_dit.enable_cache(
        (
            BlockAdapter(
                transformer=pipeline.transformer,
                blocks=pipeline.transformer.transformer_blocks,
                forward_pattern=ForwardPattern.Pattern_1,
                params_modifiers=[modifier],
            )
        ),
        cache_config=db_cache_config,
    )

    def refresh_cache_context(pipeline: Any, num_inference_steps: int, verbose: bool = True) -> None:
        """Refresh cache context for the transformer with new num_inference_steps.

        Args:
            pipeline: The LongCatImage pipeline instance.
            num_inference_steps: New number of inference steps.
        """
        if cache_config.scm_steps_mask_policy is None:
            cache_dit.refresh_context(pipeline.transformer, num_inference_steps=num_inference_steps, verbose=verbose)
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


def enable_cache_for_dit(pipeline: Any, cache_config: Any) -> Callable[[int], None]:
    """Enable cache-dit for regular single-transformer DiT models.

    Args:
        pipeline: The diffusion pipeline instance.
        cache_config: DiffusionCacheConfig instance with cache configuration.

    Returns:
        A refresh function that can be called to update cache context with new num_inference_steps.
    """
    # Build DBCacheConfig with optional SCM support
    db_cache_config = _build_db_cache_config(cache_config)

    # Build calibrator config if TaylorSeer is enabled
    calibrator_config = None
    if cache_config.enable_taylorseer:
        taylorseer_order = cache_config.taylorseer_order
        calibrator_config = TaylorSeerCalibratorConfig(taylorseer_order=taylorseer_order)
        logger.info(f"TaylorSeer enabled with order={taylorseer_order}")

    logger.info(
        f"Enabling cache-dit on transformer: "
        f"Fn={db_cache_config.Fn_compute_blocks}, "
        f"Bn={db_cache_config.Bn_compute_blocks}, "
        f"W={db_cache_config.max_warmup_steps}, "
    )

    # Enable cache-dit on the transformer
    cache_dit.enable_cache(
        pipeline.transformer,
        cache_config=db_cache_config,
        calibrator_config=calibrator_config,
    )

    def refresh_cache_context(pipeline: Any, num_inference_steps: int, verbose: bool = True) -> None:
        """Refresh cache context for the transformer with new num_inference_steps.

        Args:
            pipeline: The diffusion pipeline instance.
            num_inference_steps: New number of inference steps.
        """
        if cache_config.scm_steps_mask_policy is None:
            cache_dit.refresh_context(pipeline.transformer, num_inference_steps=num_inference_steps, verbose=verbose)
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


def enable_cache_for_bagel(pipeline: Any, cache_config: Any) -> Callable[[int], None]:
    """Enable cache-dit for Bagel model (via OmniDiffusion pipeline).

    Args:
        pipeline: The OmniDiffusion pipeline instance.
        cache_config: DiffusionCacheConfig instance with cache configuration.

    Returns:
        A refresh function that can be called to update cache context with new num_inference_steps.
    """
    # Build DBCacheConfig
    db_cache_config = _build_db_cache_config(cache_config)

    # Build calibrator config if TaylorSeer is enabled
    calibrator_config = None
    if cache_config.enable_taylorseer:
        taylorseer_order = cache_config.taylorseer_order
        calibrator_config = TaylorSeerCalibratorConfig(taylorseer_order=taylorseer_order)
        logger.info(f"TaylorSeer enabled with order={taylorseer_order}")

    # Access the transformer: BagelPipeline -> Qwen2MoTForCausalLM -> Qwen2MoTModel
    # BagelPipeline has self.language_model which is Qwen2MoTForCausalLM
    # Qwen2MoTForCausalLM has self.model which is Qwen2MoTModel
    transformer = pipeline.language_model.model

    logger.info(
        f"Enabling cache-dit on Bagel transformer: "
        f"Fn={db_cache_config.Fn_compute_blocks}, "
        f"Bn={db_cache_config.Bn_compute_blocks}, "
        f"W={db_cache_config.max_warmup_steps}, "
    )

    # Enable cache-dit on the transformer
    # Pattern_0 corresponds to (hidden_states, encoder_hidden_states) input, output
    # Custom adapter for Bagel to handle NaiveCache correctly
    # from vllm_omni.diffusion.cache.bagel_cache_adapter import BagelCachedAdapter # No longer needed
    BagelCachedAdapter.apply(
        BlockAdapter(
            transformer=transformer,
            blocks=transformer.layers,
            forward_pattern=ForwardPattern.Pattern_0,
        ),
        cache_config=db_cache_config,
        calibrator_config=calibrator_config,
    )

    def refresh_cache_context(pipeline: Any, num_inference_steps: int, verbose: bool = True) -> None:
        transformer = pipeline.language_model.model
        if cache_config.scm_steps_mask_policy is None:
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


# Register custom cache-dit enablers after function definitions
CUSTOM_DIT_ENABLERS.update(
    {
        "WanPipeline": enable_cache_for_wan22,
        "FluxPipeline": enable_cache_for_flux,
        "LongCatImagePipeline": enable_cache_for_longcat_image,
        "LongCatImageEditPipeline": enable_cache_for_longcat_image,
        "StableDiffusion3Pipeline": enable_cache_for_sd3,
        "BagelPipeline": enable_cache_for_bagel,
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
            # For regular single-transformer models
            self._refresh_func = enable_cache_for_dit(pipeline, self.config)

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
