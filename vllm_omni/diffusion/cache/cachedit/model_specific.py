# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Model-specific Cache-DiT adapters and enablers."""

import functools
from contextlib import ExitStack
from typing import Any

import cache_dit
import torch
from cache_dit import (
    BlockAdapter,
    DBCacheConfig,
    ForwardPattern,
    ParamsModifier,
)
from cache_dit.caching.block_adapters import FakeDiffusionPipeline
from cache_dit.caching.cache_adapters.cache_adapter import CachedAdapter
from cache_dit.caching.cache_blocks.pattern_0_1_2 import CachedBlocks_Pattern_0_1_2
from cache_dit.caching.cache_blocks.pattern_3_4_5 import CachedBlocks_Pattern_3_4_5
from cache_dit.caching.cache_contexts import BasicCacheConfig
from cache_dit.caching.cache_contexts.cache_manager import CachedContextManager
from vllm.logger import init_logger

from vllm_omni.diffusion.cache.cachedit.backend import (
    CUSTOM_DIT_ENABLERS,
    RefreshCacheContextFunc,
    _build_cache_context_refresh,
    _default_get_pipeline_transformer,
    _maybe_build_block_adapter,
    enable_cache_for_dit,
)
from vllm_omni.diffusion.cache.cachedit.config import CacheDiTConfig

logger = init_logger(__name__)


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
    projected_config = CacheDiTConfig.from_diffusion_config(cache_config)
    db_cache_config = projected_config.to_db_cache_config()
    calibrator_config = projected_config.to_calibrator_config()

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
        return _build_cache_context_refresh(cache_config)

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
                        max_warmup_steps=projected_config.max_warmup_steps,
                        max_cached_steps=projected_config.max_cached_steps,
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

    refresh_trans_one = _build_cache_context_refresh(cache_config)
    refresh_trans_two = _build_cache_context_refresh(cache_config, lambda pipeline: pipeline.transformer_2)

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
    projected_config = CacheDiTConfig.from_diffusion_config(cache_config)
    db_cache_config = projected_config.to_db_cache_config()
    calibrator_config = projected_config.to_calibrator_config()

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
        if projected_config.scm_steps_mask_policy is None:
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
                        mask_policy=projected_config.scm_steps_mask_policy,
                        total_steps=num_inference_steps,
                    ),
                    steps_computation_policy=projected_config.scm_steps_policy,
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
    block_adapter = _maybe_build_block_adapter(pipeline)
    return enable_cache_for_dit(pipeline, cache_config, block_adapter)


def enable_cache_for_krea2(pipeline: Any, cache_config: Any) -> RefreshCacheContextFunc:
    """Enable cache-dit for Krea 2.

    Krea 2 is a single-stream MMDiT: each ``Krea2TransformerBlock`` takes and returns only
    ``hidden_states`` (text is fused into the token stream), so the blocks follow
    ``ForwardPattern.Pattern_3``.

    ``has_separate_cfg`` is checkpoint-dependent, which is why this needs a custom enabler
    rather than a static ``_cache_dit_adapter_config``: the distilled Turbo checkpoint runs
    no-CFG (a single transformer forward per denoise step), while the Raw checkpoint runs CFG
    as two separate forwards. cache-dit tells cond/uncond apart purely by transformer-forward
    parity, so the flag must match the actual per-step forward count — ``True`` only for the
    CFG (Raw) path. The pipeline exposes this via ``is_distilled`` (read from ``model_index.json``).

    Args:
        pipeline: The Krea2Pipeline instance.
        cache_config: DiffusionCacheConfig instance with cache configuration.

    Returns:
        A refresh function that can be called to update cache context with new num_inference_steps.
    """
    transformer = _default_get_pipeline_transformer(pipeline)
    block_adapter = BlockAdapter(
        transformer=transformer,
        blocks=[transformer.transformer_blocks],
        forward_pattern=[ForwardPattern.Pattern_3],
        has_separate_cfg=not pipeline.is_distilled,
        check_forward_pattern=False,
    )
    return enable_cache_for_dit(pipeline, cache_config, block_adapter)


def register_custom_dit_enablers() -> None:
    """Register model-specific Cache-DiT enablers.

    This is called explicitly by the package initializer so registration does
    not depend on unrelated model-specific symbols being imported for their
    side effects.
    """
    CUSTOM_DIT_ENABLERS.update(
        {
            "Wan22Pipeline": enable_cache_for_wan22,
            "Wan22I2VPipeline": enable_cache_for_wan22,
            "Wan22TI2VPipeline": enable_cache_for_wan22,
            "Wan22VACEPipeline": enable_cache_for_wan22,
            "Wan22S2VPipeline": enable_cache_for_wan22_s2v,
            "Cosmos3OmniDiffusersPipeline": enable_cache_for_cosmos3,
            "Cosmos3OmniPipeline": enable_cache_for_cosmos3,
            "Krea2Pipeline": enable_cache_for_krea2,
        }
    )


__all__ = [
    "BagelCachedAdapter",
    "SensenovaCachedAdapter",
]
