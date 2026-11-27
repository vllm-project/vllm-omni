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
        self._active_branches: tuple[str, ...] = ()
        self._last_evaluation_step: int | None = None

    def initialize_hook(self, module: torch.nn.Module) -> torch.nn.Module:
        if self.extractor_fn is None:
            self.extractor_fn = get_extractor(type(module))
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
        except BaseException:
            # A failed branch must not leave partially advanced trajectory state.
            self.state_manager.reset()
            self._active_branches = ()
            self._last_evaluation_step = None
            raise
        finally:
            self.state_manager.set_context(previous_context)

    def _step_metadata(self) -> tuple[int, float, int]:
        callbacks = (self.current_step_callback, self.current_sigma_callback, self.num_inference_steps_callback)
        if any(callback is None for callback in callbacks):
            raise ValueError("scheduler callbacks are unavailable")
        values = [callback() for callback in callbacks if callback is not None]
        values = [value.item() if isinstance(value, torch.Tensor) else value for value in values]
        step_value, sigma_value, num_steps_value = values
        if step_value is None or sigma_value is None or num_steps_value is None:
            raise ValueError("scheduler metadata is unavailable")
        step, sigma, num_steps = int(step_value), float(sigma_value), int(num_steps_value)
        if step < 0 or num_steps <= 0 or step >= num_steps or not math.isfinite(sigma) or not 0 <= sigma <= 1:
            raise ValueError("expected a valid step index and exact sigma in [0, 1]")
        return step, sigma, num_steps

    def begin_step(self, branches: tuple[str, ...]) -> None:
        """Register one velocity evaluation without precomputing branch decisions.

        All CFG ranks register the global branch tuple, including idle ranks.
        Changes in ownership/active guidance reset every local history together.
        Repeated evaluations at one solver index also restart extrapolation.
        Each forward computes a target-only indicator and retains its own residual.
        """
        if not branches or any(not name for name in branches) or len(set(branches)) != len(branches):
            raise ValueError("SeaCache requires unique, nonempty branch names")
        step, _, _ = self._step_metadata()
        if branches != self._active_branches or self._last_evaluation_step != step - 1:
            self.state_manager.reset()
        self._active_branches = branches
        self._last_evaluation_step = step

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

        # The extractor supplies only denoised targets, not fully conditioned
        # control hints. Keep partially conditioned I2V/V2V targets intact.
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
        return bool(self._synchronize_decision(int(compute), device))

    def _synchronize_decision(self, decision_value: int, device: torch.device) -> int:
        """MAX of skip=0, compute=1, bypass=2 across transformer collective peers."""
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return max(1, decision_value) if self._parameter_sharded else decision_value
        decision = torch.tensor(decision_value, dtype=torch.int32, device=device)
        if self._parameter_sharded:
            from vllm_omni.diffusion.distributed.parallel_state import (
                get_fs_group,
                get_sequence_parallel_world_size,
                get_sp_group,
            )

            fs_group = get_fs_group()
            if fs_group.world_size > 1:
                torch.distributed.all_reduce(
                    decision,
                    op=torch.distributed.ReduceOp.MAX,
                    group=fs_group.device_group,
                )
            if get_sequence_parallel_world_size() > 1:
                torch.distributed.all_reduce(
                    decision,
                    op=torch.distributed.ReduceOp.MAX,
                    group=get_sp_group().device_group,
                )
            return int(decision.item())

        for group in self._collective_skip_groups:
            torch.distributed.all_reduce(
                decision,
                op=torch.distributed.ReduceOp.MAX,
                group=group,
            )
        from vllm_omni.diffusion.distributed.parallel_state import (
            get_sequence_parallel_world_size,
            get_sp_group,
        )

        if get_sequence_parallel_world_size() > 1:
            torch.distributed.all_reduce(
                decision,
                op=torch.distributed.ReduceOp.MAX,
                group=get_sp_group().device_group,
            )
        return int(decision.item())

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

        state: SeaCacheState | None = None
        indicator = None
        eligible = True
        step = num_inference_steps = 0
        try:
            if torch.is_grad_enabled():
                raise ValueError("autograd-enabled call")
            if self.state_manager._current_context is None:
                raise ValueError("missing explicit cache context")
            extra_states = ctx.extra_states or {}
            vision_items = extra_states.get("sea_cache_latents")
            if not isinstance(vision_items, list):
                raise ValueError("extractor did not provide SeaCache vision inputs")
            noisy_frame_mask = extra_states.get("sea_cache_noisy_frame_mask")
            if isinstance(noisy_frame_mask, torch.Tensor) and not bool(torch.any(noisy_frame_mask != 0).item()):
                raise ValueError("conditioning-only input")
            step, sigma, num_inference_steps = self._step_metadata()
            state = self.state_manager.get_state()
            assert state is not None
            # Validate before collective agreement, never after a shared skip.
            if state.history and any(
                residual.shape != ctx.hidden_states.shape
                or residual.device != ctx.hidden_states.device
                or residual.dtype != ctx.hidden_states.dtype
                for _, residual in state.history
            ):
                state.reset()
            indicator = self._build_indicator(vision_items, sigma)
            if indicator is None:
                raise ValueError("invalid noisy-target indicator")
        except (IndexError, TypeError, ValueError, RuntimeError) as error:
            self._warn_once(f"SeaCache input is ineligible; running full: {error}")
            eligible = False

        local_decision = 2
        if eligible:
            assert state is not None
            local_decision = int(self._resolve_gate(state, indicator, step, num_inference_steps))
        decision = self._synchronize_decision(local_decision, ctx.hidden_states.device)
        if decision == 2:
            self.state_manager.reset()
            return self._run_uncached(ctx)
        assert state is not None
        should_compute = bool(decision)

        if should_compute:
            state.accumulated_distance = 0.0
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
        state.consecutive_cached += 1
        self.skip_count += 1

        return ctx.postprocess(ctx.hidden_states + residual)

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
        self._active_branches = ()
        self._last_evaluation_step = None
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
