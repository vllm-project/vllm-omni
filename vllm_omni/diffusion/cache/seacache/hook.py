# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.cache.seacache.config import SeaCacheConfig
from vllm_omni.diffusion.cache.seacache.sea_filter import (
    apply_sea_filter,
    extrapolate_residual,
    indicator_distance,
)
from vllm_omni.diffusion.cache.seacache.state import SeaCacheState
from vllm_omni.diffusion.cache.teacache.extractors import CacheContext, get_extractor
from vllm_omni.diffusion.hooks import HookRegistry, ModelHook, StateManager

logger = init_logger(__name__)


def _is_parameter_sharded(module: torch.nn.Module) -> bool:
    """Detect parameter-sharding runtimes whose collectives cannot be skipped."""
    for submodule in module.modules():
        module_type = type(submodule)
        if callable(getattr(submodule, "_get_fsdp_state", None)):
            return True
        if module_type.__name__ == "FullyShardedDataParallel" and module_type.__module__.startswith(
            "torch.distributed.fsdp"
        ):
            return True
        for parameter in submodule.parameters(recurse=False):
            parameter_type = type(parameter)
            if (
                parameter_type.__name__ == "FlatParameter"
                and parameter_type.__module__.startswith("torch.distributed.fsdp")
            ) or (
                parameter_type.__name__ == "DTensor"
                and parameter_type.__module__.startswith("torch.distributed.tensor")
            ):
                return True
    return False


class SeaCacheRootHook(ModelHook):
    """Drive SeaCache gating and transformer forward control."""

    _HOOK_NAME = "sea_cache"

    def __init__(
        self,
        config: SeaCacheConfig,
        *,
        current_step_callback: Callable[[], int | torch.Tensor | None] | None = None,
        current_sigma_callback: Callable[[], float | torch.Tensor | None] | None = None,
        num_inference_steps_callback: Callable[[], int | torch.Tensor | None] | None = None,
        extractor_fn: Callable[..., CacheContext] | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.current_step_callback = current_step_callback
        self.current_sigma_callback = current_sigma_callback
        self.num_inference_steps_callback = num_inference_steps_callback
        self.state_manager = StateManager(SeaCacheState)
        self._warned_messages: set[str] = set()
        self.full_count = 0
        self.skip_count = 0
        self.extractor_fn = extractor_fn
        self._parameter_sharded = False
        self._collective_skip_groups: list[torch.distributed.ProcessGroup] = []

    def initialize_hook(self, module: torch.nn.Module) -> torch.nn.Module:
        if self.extractor_fn is None:
            self.extractor_fn = get_extractor(module.__class__.__name__)
        self._parameter_sharded = _is_parameter_sharded(module)
        seen_groups: set[int] = set()
        for block in getattr(module, "gen_layers", ()):
            registry = getattr(block, "_hook_registry", None)
            dlo_hook = registry.get_hook("distributed_layerwise_offload") if registry is not None else None
            group = getattr(dlo_hook, "dp_group", None)
            if group is not None and int(getattr(dlo_hook, "dp_size", 1)) > 1 and id(group) not in seen_groups:
                seen_groups.add(id(group))
                self._collective_skip_groups.append(group)
        return module

    def _warn_once(self, message: str) -> None:
        if message not in self._warned_messages:
            logger.warning(message)
            self._warned_messages.add(message)

    @contextmanager
    def cache_context(self, name: str) -> Iterator[None]:
        previous_context = self.state_manager._context
        self.state_manager.set_context(name)
        try:
            yield
        finally:
            self.state_manager.set_context(previous_context)

    def _build_indicator(
        self,
        vision_items: list[torch.Tensor] | None,
        sigma: float,
    ) -> list[torch.Tensor] | None:
        if not vision_items:
            return None
        hidden_states = vision_items[-1]
        if not isinstance(hidden_states, torch.Tensor) or hidden_states.ndim != 5:
            return None

        # Controls precede the denoised target in packed vision-token order,
        # and each vision item is filtered independently.
        if any(
            not isinstance(item, torch.Tensor)
            or item.ndim != 5
            or item.shape[0] != hidden_states.shape[0]
            or item.shape[1:] != hidden_states.shape[1:]
            for item in vision_items
        ):
            return None

        indicator = []
        for batch_index in range(hidden_states.shape[0]):
            for latent in vision_items:
                thwc = latent[batch_index].movedim(0, -1)
                indicator.append(
                    apply_sea_filter(
                        thwc,
                        sigma=sigma,
                        power_exp=self.config.power_exp,
                    ).detach()
                )
        return indicator or None

    def _resolve_gate(
        self,
        state: SeaCacheState,
        indicator: list[torch.Tensor] | None,
        step: int,
        num_inference_steps: int,
    ) -> bool:
        if state.last_step is not None and step != state.last_step + 1:
            state.reset()
        state.last_step = step
        max_consecutive = bool(
            self.config.max_consecutive_cached and state.consecutive_cached >= self.config.max_consecutive_cached
        )
        forced_compute = (
            step < 1
            or step >= num_inference_steps - 1
            or max_consecutive
            or not state.history
            or indicator is None
            or state.previous_indicator is None
        )
        if forced_compute:
            state.accumulated_distance = 0.0
            state.previous_indicator = None if indicator is None else [value.detach() for value in indicator]
            return True

        assert indicator is not None
        assert state.previous_indicator is not None
        distance = indicator_distance(indicator, state.previous_indicator)
        state.previous_indicator = [value.detach() for value in indicator]
        if not math.isfinite(distance):
            state.accumulated_distance = 0.0
            self._warn_once("SeaCache indicator history changed shape, device, or dtype; running full.")
            return True

        state.accumulated_distance += distance
        if state.accumulated_distance < self.config.threshold:
            return False
        state.accumulated_distance = 0.0
        return True

    def _synchronize_compute(self, compute: bool, device: torch.device) -> bool:
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return True if self._parameter_sharded else compute
        decision = torch.tensor(int(compute), dtype=torch.int32, device=device)
        if self._parameter_sharded:
            from vllm_omni.diffusion.distributed.parallel_state import get_world_group

            world_group = get_world_group()
            if world_group.world_size > 1:
                torch.distributed.all_reduce(
                    decision,
                    op=torch.distributed.ReduceOp.MAX,
                    group=world_group.device_group,
                )
            return bool(decision.item())

        for group in self._collective_skip_groups:
            torch.distributed.all_reduce(
                decision,
                op=torch.distributed.ReduceOp.MAX,
                group=group,
            )
        from vllm_omni.diffusion.distributed.parallel_state import (
            get_sp_group,
            get_ulysses_parallel_world_size,
        )

        if get_ulysses_parallel_world_size() > 1:
            torch.distributed.all_reduce(
                decision,
                op=torch.distributed.ReduceOp.MAX,
                group=get_sp_group().ulysses_group,
            )
        return bool(decision.item())

    @torch.compiler.disable
    def new_forward(
        self,
        module: torch.nn.Module,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if self.extractor_fn is None:
            raise RuntimeError("SeaCache extractor was not initialized")
        ctx = self.extractor_fn(module, *args, **kwargs)

        if torch.is_grad_enabled():
            self._warn_once("SeaCache is inference-only; autograd-enabled calls run in full.")
            return self._run_uncached(ctx)
        if self.state_manager._current_context is None:
            self._warn_once("SeaCache requires an explicit cache context; running full.")
            return self._run_uncached(ctx)
        callbacks = (
            self.current_step_callback,
            self.current_sigma_callback,
            self.num_inference_steps_callback,
        )
        if any(callback is None for callback in callbacks):
            self._warn_once("SeaCache requires scheduler step, sigma, and step-count callbacks; running full.")
            return self._run_uncached(ctx)
        assert self.current_step_callback is not None
        assert self.current_sigma_callback is not None
        assert self.num_inference_steps_callback is not None

        try:
            extra_states = ctx.extra_states or {}
            vision_items = extra_states.get("sea_cache_latents")
            if not isinstance(vision_items, list):
                raise ValueError("extractor did not provide SeaCache vision inputs")
            noisy_frame_mask = extra_states.get("sea_cache_noisy_frame_mask")
            if isinstance(noisy_frame_mask, torch.Tensor) and not bool(torch.any(noisy_frame_mask != 0).item()):
                self._warn_once("SeaCache requires noisy vision; conditioning-only calls run in full.")
                return self._run_uncached(ctx)

            step = self.current_step_callback()
            sigma = self.current_sigma_callback()
            num_inference_steps = self.num_inference_steps_callback()
            if isinstance(step, torch.Tensor):
                step = step.item()
            if isinstance(sigma, torch.Tensor):
                sigma = sigma.item()
            if isinstance(num_inference_steps, torch.Tensor):
                num_inference_steps = num_inference_steps.item()
            if step is None or sigma is None or num_inference_steps is None:
                raise ValueError("scheduler metadata is unavailable")
            step = int(step)
            sigma = float(sigma)
            num_inference_steps = int(num_inference_steps)
            if (
                step < 0
                or num_inference_steps <= 0
                or step >= num_inference_steps
                or not math.isfinite(sigma)
                or not 0.0 <= sigma <= 1.0
            ):
                raise ValueError("expected a valid step index and exact sigma in [0, 1]")
        except (IndexError, TypeError, ValueError, RuntimeError) as error:
            self._warn_once(f"SeaCache metadata is invalid; running full: {error}")
            return self._run_uncached(ctx)

        state: SeaCacheState = self.state_manager.get_state()
        try:
            indicator = self._build_indicator(vision_items, sigma)
        except (TypeError, ValueError, RuntimeError) as error:
            self._warn_once(f"SeaCache could not construct its vision indicator; running full: {error}")
            indicator = None

        local_compute = self._resolve_gate(state, indicator, step, num_inference_steps)
        should_compute = self._synchronize_compute(local_compute, ctx.hidden_states.device)
        if should_compute and not local_compute:
            state.accumulated_distance = 0.0

        if should_compute:
            self.full_count += 1
            output = self._run_full_stack(ctx)
            result = ctx.postprocess(output)
            self._record_execution(state, step, ctx.hidden_states, output)
            return result

        residual = extrapolate_residual(
            state.history,
            step,
            self.config.residual_order,
        )
        if residual.device != ctx.hidden_states.device:
            residual = residual.to(ctx.hidden_states.device)
        state.consecutive_cached += 1
        self.skip_count += 1

        can_reuse = (
            residual.shape == ctx.hidden_states.shape
            and residual.device == ctx.hidden_states.device
            and residual.dtype == ctx.hidden_states.dtype
        )
        if can_reuse:
            return ctx.postprocess(ctx.hidden_states + residual)

        output = self._run_full_stack(ctx)
        result = ctx.postprocess(output)
        self._record_execution(state, step, ctx.hidden_states, output)
        return result

    @staticmethod
    def _run_full_stack(ctx: CacheContext) -> torch.Tensor:
        outputs = ctx.run_transformer_blocks()
        if not outputs:
            raise RuntimeError("Cache extractor returned no transformer outputs")
        return outputs[0]

    @staticmethod
    def _run_uncached(ctx: CacheContext) -> Any:
        return ctx.postprocess(SeaCacheRootHook._run_full_stack(ctx))

    def _record_execution(
        self,
        state: SeaCacheState,
        step: int,
        execution_input: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        if (
            output.shape == execution_input.shape
            and output.device == execution_input.device
            and output.dtype == execution_input.dtype
        ):
            state.history.append((step, (output - execution_input).detach().clone()))
            state.history = state.history[-(self.config.residual_order + 1) :]
            state.consecutive_cached = 0
            return

        state.history.clear()
        state.accumulated_distance = 0.0
        self._warn_once("SeaCache execution boundary returned an incompatible tensor; clearing cache history.")

    def reset_state(self, module: torch.nn.Module) -> torch.nn.Module:
        self.state_manager.reset()
        self.full_count = 0
        self.skip_count = 0
        return module

    def refresh(self, module: torch.nn.Module) -> None:
        self.reset_state(module)


def apply_sea_cache_hook(
    module: torch.nn.Module,
    config: SeaCacheConfig,
    *,
    current_step_callback: Callable[[], int | torch.Tensor | None] | None = None,
    current_sigma_callback: Callable[[], float | torch.Tensor | None] | None = None,
    num_inference_steps_callback: Callable[[], int | torch.Tensor | None] | None = None,
    extractor_fn: Callable[..., CacheContext] | None = None,
) -> SeaCacheRootHook:
    registry = HookRegistry.get_or_create(module)
    hook = SeaCacheRootHook(
        config,
        current_step_callback=current_step_callback,
        current_sigma_callback=current_sigma_callback,
        num_inference_steps_callback=num_inference_steps_callback,
        extractor_fn=extractor_fn,
    )
    registry.register_hook(SeaCacheRootHook._HOOK_NAME, hook)
    return hook
