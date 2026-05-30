# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Hook-based MagCache implementation for vLLM-Omni.

This module implements a diffusers-style hook system for MagCache (Magnitude-based Cache),
providing adaptive caching for diffusion model inference.

MagCache speeds up inference by skipping transformer block computations when the accumulated
magnitude error is below a threshold, reusing cached residuals instead.

Based on: https://github.com/Zehong-Ma/MagCache
Reference: diffusers/src/diffusers/hooks/mag_cache.py

Architecture:
- MagCacheStrategy: Model-specific strategy for preprocessing/postprocessing
- MagCacheState: Per-step state tracking residuals and accumulated error
- MagCacheHeadHook: Decides whether to skip based on accumulated error
- MagCacheBlockHook: Computes and stores residuals at tail block
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from diffusers.hooks._helpers import TransformerBlockMetadata, TransformerBlockRegistry
from diffusers.utils.torch_utils import unwrap_module

from vllm_omni.diffusion.cache.magcache.config import MagCacheConfig
from vllm_omni.diffusion.cache.magcache.state import MagCacheState
from vllm_omni.diffusion.cache.magcache.strategy import MagCacheStrategy, MagCacheStrategyRegistry
from vllm_omni.diffusion.hooks.base import HookRegistry, ModelHook, StateManager
from vllm_omni.logger import init_logger

logger = init_logger(__name__)

_MAG_CACHE_LEADER_BLOCK_HOOK = "mag_cache_leader_block_hook"
_MAG_CACHE_BLOCK_HOOK = "mag_cache_block_hook"


class MagCacheHeadHook(ModelHook):
    """Head block hook for MagCache - decides whether to skip computation."""

    _HOOK_NAME = "mag_cache_head"

    def __init__(
        self,
        state_manager: StateManager,
        config: MagCacheConfig,
        strategy: MagCacheStrategy | None = None,
        is_tail: bool = False,
    ):
        super().__init__()
        self.state_manager = state_manager
        self.config = config
        self._strategy = strategy
        self._metadata = None
        self._is_tail = is_tail

    def initialize_hook(self, module: torch.nn.Module) -> torch.nn.Module:
        unwrapped_module = unwrap_module(module)
        block_class = unwrapped_module.__class__

        try:
            self._metadata = TransformerBlockRegistry.get(block_class)
        except ValueError:
            if self._strategy is not None:
                metadata = self._strategy.register_block_metadata(block_class)
                if metadata is not None:
                    TransformerBlockRegistry.register(model_class=block_class, metadata=metadata)
                else:
                    TransformerBlockRegistry.register(
                        model_class=block_class,
                        metadata=TransformerBlockMetadata(
                            return_hidden_states_index=1,
                            return_encoder_hidden_states_index=0,
                        ),
                    )
            else:
                TransformerBlockRegistry.register(
                    model_class=block_class,
                    metadata=TransformerBlockMetadata(
                        return_hidden_states_index=1,
                        return_encoder_hidden_states_index=0,
                    ),
                )
            self._metadata = TransformerBlockRegistry.get(block_class)

        return module

    @torch.compiler.disable
    def new_forward(self, module: torch.nn.Module, *args, **kwargs):
        if self.state_manager._current_context is None:
            self.state_manager.set_context("magcache")

        if hasattr(self._metadata, "hidden_states_argument_name"):
            arg_name = self._metadata.hidden_states_argument_name
        else:
            arg_name = "hidden_states"
        hidden_states = self._metadata._get_parameter_from_args_kwargs(arg_name, args, kwargs)

        state: MagCacheState = self.state_manager.get_state()
        state.head_block_input = hidden_states

        if state._is_first_step:
            state.accumulated_ratio = 1.0
            state.accumulated_err = 0.0
            state.accumulated_steps = 0
            state._is_first_step = False

        should_compute = True

        if self.config.mag_calibrate:
            should_compute = True
        else:
            current_step = state.step_index
            if current_step >= len(self.config.mag_ratios):
                current_scale = 1.0
            else:
                current_scale = self.config.mag_ratios[current_step]

            retention_step = int(self.config.retention_ratio * self.config.num_inference_steps + 0.5)

            if current_step >= retention_step:
                state.accumulated_ratio *= current_scale
                state.accumulated_steps += 1
                state.accumulated_err += abs(1.0 - state.accumulated_ratio)

                if (
                    state.previous_residual is not None
                    and state.accumulated_err <= self.config.threshold
                    and state.accumulated_steps <= self.config.max_skip_steps
                ):
                    should_compute = False
                else:
                    state.accumulated_ratio = 1.0
                    state.accumulated_steps = 0
                    state.accumulated_err = 0.0

        state.should_compute = should_compute

        if not should_compute:
            res = state.previous_residual

            if isinstance(res, tuple):
                res = tuple(r.to(hidden_states.device) for r in res)

                if self._strategy is not None:
                    original_encoder_hidden_states = self._metadata._get_parameter_from_args_kwargs(
                        "encoder_hidden_states", args, kwargs
                    )
                    if original_encoder_hidden_states.device != res[1].device:
                        original_encoder_hidden_states = original_encoder_hidden_states.to(res[1].device)
                    output, enc_output = self._strategy.apply_residual_tuple(
                        hidden_states, original_encoder_hidden_states, res
                    )
                    ret_list = [None] * 2
                    ret_list[self._metadata.return_hidden_states_index] = output
                    ret_list[self._metadata.return_encoder_hidden_states_index] = enc_output
                    return self.log_cache_hit(state, output, ret_list)
                else:
                    raise RuntimeError(
                        f"MagCache residual is tuple but no strategy available for {self._metadata.transformer_type}. "
                        f"Please register a MagCacheStrategy for this model."
                    )
            elif res.device != hidden_states.device:
                res = res.to(hidden_states.device)

            if self._strategy is not None:
                output = self._strategy.apply_residual(hidden_states, res)
            elif res.shape == hidden_states.shape:
                output = hidden_states + res
            elif (
                hidden_states.ndim == 3
                and res.ndim == 3
                and hidden_states.shape[0] == res.shape[0]
                and hidden_states.shape[2] == res.shape[2]
            ):
                diff = hidden_states.shape[1] - res.shape[1]
                if diff > 0:
                    output = hidden_states.clone()
                    output[:, diff:, :] = output[:, diff:, :] + res
                else:
                    output = hidden_states + res
            else:
                output = hidden_states + res

            if self._metadata.return_encoder_hidden_states_index is not None:
                original_encoder_hidden_states = self._metadata._get_parameter_from_args_kwargs(
                    "encoder_hidden_states", args, kwargs
                )
                max_idx = max(
                    self._metadata.return_hidden_states_index, self._metadata.return_encoder_hidden_states_index
                )
                ret_list = [None] * (max_idx + 1)
                ret_list[self._metadata.return_hidden_states_index] = output
                ret_list[self._metadata.return_encoder_hidden_states_index] = original_encoder_hidden_states
                return self.log_cache_hit(state, output, ret_list)
            else:
                return self.log_cache_hit(state, output, None)
        else:
            output = self.fn_ref.original_forward(*args, **kwargs)
            result = self.log_cache_miss(state, output)

        if self._is_tail:
            if isinstance(output, tuple):
                out_hidden = output[self._metadata.return_hidden_states_index]
            else:
                out_hidden = output

            in_hidden = state.head_block_input

            if in_hidden is not None:
                if self._strategy is not None:
                    residual = self._strategy.compute_residual(out_hidden, in_hidden)
                elif out_hidden.shape == in_hidden.shape:
                    residual = out_hidden - in_hidden
                else:
                    residual = out_hidden

                if self.config.mag_calibrate:
                    self._perform_calibration_head(state, residual)

                state.previous_residual = residual
                self._advance_step_head(state)

        return result

    def _perform_calibration_head(
        self,
        state: MagCacheState,
        current_residual: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        if self._strategy is not None:
            ratio, std, cos_dis = self._strategy.compute_calibration_metrics(current_residual, state.previous_residual)
        else:
            if state.previous_residual is None:
                ratio, std, cos_dis = 1.0, 0.0, 0.0
            else:
                ratio, std, cos_dis = 1.0, 0.0, 0.0

        state.calibration_ratios.append(ratio)
        state.norm_ratios.append(round(ratio, 5))
        state.norm_stds.append(round(std, 5))
        state.cos_dises.append(round(cos_dis, 5))

    def _advance_step_head(self, state: MagCacheState) -> None:
        state.step_index += 1
        if state.step_index >= self.config.num_inference_steps:
            if self.config.mag_calibrate:
                logger.info("MagCache calibration complete.")
                logger.info(f"norm_ratios: {state.norm_ratios}")
                logger.info(f"norm_stds: {state.norm_stds}")
                logger.info(f"cos_dises: {state.cos_dises}")
                logger.info("Copy these values to DiffusionCacheConfig(mag_ratios=...) for production use")

            state.step_index = 0
            state.accumulated_ratio = 1.0
            state.accumulated_steps = 0
            state.accumulated_err = 0.0
            state.previous_residual = None
            state.calibration_ratios = []
            state.norm_ratios = []
            state.norm_stds = []
            state.cos_dises = []
            state._is_first_step = True

    def log_cache_hit(self, state: MagCacheState, output, ret):
        step = state.step_index
        if state.previous_residual is not None:
            if isinstance(state.previous_residual, tuple):
                residual_shape = tuple(r.shape for r in state.previous_residual)
            else:
                residual_shape = state.previous_residual.shape
        else:
            residual_shape = "None"
        logger.debug(
            f"[MagCache][HEAD] STEP={step}: CACHE_HIT (err={state.accumulated_err:.6f}, "
            f"steps_skipped={state.accumulated_steps}, residual_shape={residual_shape}"
        )
        return ret if ret is not None else output

    def log_cache_miss(self, state: MagCacheState, output):
        step = state.step_index
        residual_norm = 0.0
        if state.previous_residual is not None:
            if isinstance(state.previous_residual, tuple):
                residual_norm = sum(float(torch.norm(r).item()) for r in state.previous_residual)
            else:
                residual_norm = float(torch.norm(state.previous_residual).item())
        logger.debug(
            f"[MagCache][HEAD] STEP={step}: CACHE_MISS "
            f"(err={state.accumulated_err:.6f}, acc_ratio={state.accumulated_ratio:.6f}, "
            f"residual_norm={residual_norm:.6f}, threshold={self.config.threshold}, "
            f"max_skip={self.config.max_skip_steps})"
        )
        return output

    def reset_state(self, module: torch.nn.Module) -> torch.nn.Module:
        self.state_manager.reset()
        return module


class MagCacheBlockHook(ModelHook):
    """Block hook for MagCache - computes residuals at tail block."""

    _HOOK_NAME = "mag_cache_block"

    def __init__(
        self,
        state_manager: StateManager,
        is_tail: bool = False,
        config: MagCacheConfig | None = None,
        strategy: MagCacheStrategy | None = None,
    ):
        super().__init__()
        self.state_manager = state_manager
        self.is_tail = is_tail
        self.config = config
        self._strategy = strategy
        self._metadata = None

    def initialize_hook(self, module: torch.nn.Module) -> torch.nn.Module:
        unwrapped_module = unwrap_module(module)
        block_class = unwrapped_module.__class__

        try:
            self._metadata = TransformerBlockRegistry.get(block_class)
        except ValueError:
            if self._strategy is not None:
                metadata = self._strategy.register_block_metadata(block_class)
                if metadata is not None:
                    TransformerBlockRegistry.register(model_class=block_class, metadata=metadata)
                else:
                    TransformerBlockRegistry.register(
                        model_class=block_class,
                        metadata=TransformerBlockMetadata(
                            return_hidden_states_index=1,
                            return_encoder_hidden_states_index=0,
                        ),
                    )
            else:
                TransformerBlockRegistry.register(
                    model_class=block_class,
                    metadata=TransformerBlockMetadata(
                        return_hidden_states_index=1,
                        return_encoder_hidden_states_index=0,
                    ),
                )
            self._metadata = TransformerBlockRegistry.get(block_class)

        return module

    def reset_state(self, module: torch.nn.Module) -> torch.nn.Module:
        self.state_manager.reset()
        return module

    @torch.compiler.disable
    def new_forward(self, module: torch.nn.Module, *args, **kwargs):
        if self.state_manager._current_context is None:
            self.state_manager.set_context("magcache")
        state: MagCacheState = self.state_manager.get_state()

        if not state.should_compute:
            res = state.previous_residual
            if res is None:
                res = torch.zeros_like(args[0])

            if hasattr(self._metadata, "hidden_states_argument_name"):
                arg_name = self._metadata.hidden_states_argument_name
            else:
                arg_name = "hidden_states"
            hidden_states = self._metadata._get_parameter_from_args_kwargs(arg_name, args, kwargs)

            if self.is_tail:
                self.advance_step(state)

            if self._metadata.return_encoder_hidden_states_index is not None:
                encoder_hidden_states = self._metadata._get_parameter_from_args_kwargs(
                    "encoder_hidden_states", args, kwargs
                )

                if self._strategy is not None:
                    out_hidden, enc_out = self._strategy.apply_residual_tuple(hidden_states, encoder_hidden_states, res)
                else:
                    out_hidden = hidden_states + res
                    enc_out = encoder_hidden_states

                max_idx = max(
                    self._metadata.return_hidden_states_index, self._metadata.return_encoder_hidden_states_index
                )
                ret_list = [None] * (max_idx + 1)
                ret_list[self._metadata.return_hidden_states_index] = out_hidden
                ret_list[self._metadata.return_encoder_hidden_states_index] = enc_out
                return tuple(ret_list)

            if self._strategy is not None:
                output = self._strategy.apply_residual(hidden_states, res)
            else:
                output = hidden_states + res

            return output

        output = self.fn_ref.original_forward(*args, **kwargs)

        if self.is_tail:
            if isinstance(output, tuple):
                out_hidden = output[self._metadata.return_hidden_states_index]
            else:
                out_hidden = output

            in_hidden = state.head_block_input

            if in_hidden is None:
                return output

            if self._strategy is not None:
                residual = self._strategy.compute_residual(output, in_hidden)
            elif out_hidden.shape == in_hidden.shape:
                residual = out_hidden - in_hidden
            elif out_hidden.ndim == 3 and in_hidden.ndim == 3 and out_hidden.shape[2] == in_hidden.shape[2]:
                residual = out_hidden - in_hidden
            else:
                residual = out_hidden

            if self.config.mag_calibrate:
                self.perform_calibration(state, residual)

            state.previous_residual = residual
            self.advance_step(state)

            self.log_residual_computed(state, residual)

        return output

    def log_residual_computed(
        self,
        state: MagCacheState,
        residual: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        step = state.step_index
        if residual is None:
            residual_norm = 0.0
            residual_shape = "None"
        elif isinstance(residual, tuple):
            residual_norm = sum(float(torch.norm(r).item()) for r in residual)
            residual_shape = tuple(r.shape for r in residual)
        else:
            residual_norm = float(torch.norm(residual).item())
            residual_shape = residual.shape
        logger.debug(
            f"[MagCache][TAIL] STEP={step}: RESIDUAL_COMPUTED (norm={residual_norm:.6f}, shape={residual_shape})"
        )

    def perform_calibration(
        self,
        state: MagCacheState,
        current_residual: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        if self._strategy is not None:
            ratio, std, cos_dis = self._strategy.compute_calibration_metrics(current_residual, state.previous_residual)
        else:

            def _get_norm(residual: torch.Tensor | tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
                if isinstance(residual, tuple):
                    return sum(torch.linalg.norm(r.float(), dim=-1) for r in residual)
                return torch.linalg.norm(residual.float(), dim=-1)

            if state.previous_residual is None:
                ratio, std, cos_dis = 1.0, 0.0, 0.0
            else:
                curr_norm = _get_norm(current_residual)
                prev_norm = _get_norm(state.previous_residual)
                ratio = (curr_norm / (prev_norm + 1e-8)).mean().item()
                std = (curr_norm / (prev_norm + 1e-8)).std().item()
                cos_dis = (
                    (
                        1
                        - F.cosine_similarity(
                            current_residual.flatten(0, -2) if current_residual.ndim > 2 else current_residual,
                            state.previous_residual.flatten(0, -2)
                            if state.previous_residual.ndim > 2
                            else state.previous_residual,
                            dim=-1,
                            eps=1e-8,
                        )
                    )
                    .mean()
                    .item()
                )

        state.calibration_ratios.append(ratio)
        state.norm_ratios.append(round(ratio, 5))
        state.norm_stds.append(round(std, 5))
        state.cos_dises.append(round(cos_dis, 5))

    def advance_step(self, state: MagCacheState) -> None:
        state.step_index += 1
        if state.step_index >= self.config.num_inference_steps:
            if self.config.mag_calibrate:
                logger.info("MagCache calibration complete.")
                logger.info(f"norm_ratios: {state.norm_ratios}")
                logger.info(f"norm_stds: {state.norm_stds}")
                logger.info(f"cos_dises: {state.cos_dises}")
                logger.info("Copy these values to DiffusionCacheConfig(mag_ratios=...) for production use")

            state.step_index = 0
            state.accumulated_ratio = 1.0
            state.accumulated_steps = 0
            state.accumulated_err = 0.0
            state.previous_residual = None
            state.calibration_ratios = []
            state.norm_ratios = []
            state.norm_stds = []
            state.cos_dises = []
            state._is_first_step = True


def apply_mag_cache_hook(
    module: torch.nn.Module,
    config: MagCacheConfig,
    strategy: MagCacheStrategy | None = None,
) -> None:
    """Apply MagCache optimization to a transformer module.

    Args:
        module: Transformer model to optimize (e.g., FluxTransformer2DModel)
        config: MagCacheConfig specifying caching parameters
        strategy: Optional strategy to use. If None, will be looked up from registry.
    """
    HookRegistry.check_if_exists_or_initialize(module)

    transformer_type = config.transformer_type
    if strategy is None:
        strategy = MagCacheStrategyRegistry.get_if_exists(transformer_type)

    if strategy is None:
        logger.warning(
            f"MagCache: No strategy found for '{transformer_type}'. "
            f"Using default behavior. Available strategies: {list(MagCacheStrategyRegistry._registry.keys())}"
        )
    else:
        strategy_name = type(strategy).__name__
        logger.info(f"MagCache: Applying {strategy_name} for '{transformer_type}'")
        if hasattr(type(strategy), "register_blocks"):
            type(strategy).register_blocks()

    state_manager = StateManager(MagCacheState, (), {})
    remaining_blocks = []

    for name, submodule in module.named_children():
        if not isinstance(submodule, torch.nn.ModuleList):
            continue
        for index, block in enumerate(submodule):
            remaining_blocks.append((f"{name}.{index}", block))

    if not remaining_blocks:
        logger.warning("MagCache: No transformer blocks found to apply hooks.")
        return

    if len(remaining_blocks) == 1:
        name, block = remaining_blocks[0]
        logger.info(f"MagCache: Applying Head+Tail Hook to single block '{name}'")
        _apply_mag_cache_head_hook(block, state_manager, config, strategy, is_tail=True)
        return

    head_block_name, head_block = remaining_blocks.pop(0)
    tail_block_name, tail_block = remaining_blocks.pop(-1)

    logger.info(f"MagCache: Applying Head Hook to {head_block_name}")
    _apply_mag_cache_head_hook(head_block, state_manager, config, strategy)

    for name, block in remaining_blocks:
        _apply_mag_cache_block_hook(block, state_manager, config, strategy=strategy)

    logger.info(f"MagCache: Applying Tail Hook to {tail_block_name}")
    _apply_mag_cache_block_hook(tail_block, state_manager, config, is_tail=True, strategy=strategy)


def _apply_mag_cache_head_hook(
    block: torch.nn.Module,
    state_manager: StateManager,
    config: MagCacheConfig,
    strategy: MagCacheStrategy | None = None,
    is_tail: bool = False,
) -> None:
    registry = HookRegistry.check_if_exists_or_initialize(block)

    if registry.get_hook(_MAG_CACHE_LEADER_BLOCK_HOOK) is not None:
        registry.remove_hook(_MAG_CACHE_LEADER_BLOCK_HOOK)

    hook = MagCacheHeadHook(state_manager, config, strategy, is_tail=is_tail)
    registry.register_hook(_MAG_CACHE_LEADER_BLOCK_HOOK, hook)


def _apply_mag_cache_block_hook(
    block: torch.nn.Module,
    state_manager: StateManager,
    config: MagCacheConfig,
    is_tail: bool = False,
    strategy: MagCacheStrategy | None = None,
) -> None:
    registry = HookRegistry.check_if_exists_or_initialize(block)

    if registry.get_hook(_MAG_CACHE_BLOCK_HOOK) is not None:
        registry.remove_hook(_MAG_CACHE_BLOCK_HOOK)

    hook = MagCacheBlockHook(state_manager, is_tail, config, strategy)
    registry.register_hook(_MAG_CACHE_BLOCK_HOOK, hook)
