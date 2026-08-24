# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Reference-hint cache backend (RFC #4710, P1).

The backend handles the complete acceleration lifecycle: request reset,
denoising-step/CFG-branch bookkeeping, retained hint history, and the selected
reuse strategy.  Models expose only the acceleration-neutral
``ModelRegion.REFERENCE_HINTS`` execution seam and contain no cache state or
policy.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar, cast

import torch
import torch.nn as nn
from vllm.logger import init_logger

from vllm_omni.diffusion.cache.base import CacheBackend
from vllm_omni.diffusion.cache.ref_hint_cache.state import (
    SUPPORTED_REF_HINT_STRATEGIES,
    RefHintCacheState,
)
from vllm_omni.diffusion.data import DiffusionCacheConfig
from vllm_omni.diffusion.forward_context import (
    get_forward_context,
    is_forward_context_available,
)
from vllm_omni.diffusion.model_region import (
    ModelRegion,
    ModelRegionHandler,
)

logger = init_logger(__name__)

T = TypeVar("T")
HintValue = list[torch.Tensor]

_TRANSFORMER_ATTRS = ("transformer", "transformer_2")
_FORECAST_GAIN = 0.5
_FORECAST_ALPHA_MAX = 1.5
_FORECAST_CORRECTION_MAX = 0.25


class RefHintCacheBackend(CacheBackend):
    """Framework-side reference-hint reuse and forecasting.

    ``ref_hint_strategy="reuse"`` returns the latest fresh hints on skipped
    steps.  ``"forecast50"`` retains two fresh observations and applies a
    damped first-order prediction with nominal gain 0.5 and a 0.25
    trust-region cap.  Both strategies remain approximate and require
    ``ref_hint_acknowledge_lossy=True`` when
    ``ref_hint_refresh_interval >= 2``.
    """

    def __init__(self, config: DiffusionCacheConfig):
        super().__init__(config)
        self._states: dict[int, RefHintCacheState[HintValue]] = {}

    def _get_transformers(self, pipeline: object) -> list[nn.Module]:
        """Return every present transformer for multi-expert/expert-only pipelines."""
        transformers: list[nn.Module] = []
        for attr in _TRANSFORMER_ATTRS:
            transformer = cast(nn.Module | None, getattr(pipeline, attr, None))
            if transformer is not None:
                transformers.append(transformer)
        if not transformers:
            raise ValueError("ref_hint cache backend requires pipeline.transformer or pipeline.transformer_2")
        return transformers

    def _strategy(self) -> str:
        strategy = str(getattr(self.config, "ref_hint_strategy", "forecast50"))
        if strategy not in SUPPORTED_REF_HINT_STRATEGIES:
            supported = ", ".join(sorted(SUPPORTED_REF_HINT_STRATEGIES))
            raise ValueError(f"Unsupported ref_hint_strategy={strategy!r}; expected one of: {supported}")
        return strategy

    def _refresh_interval(self) -> int:
        return max(1, int(getattr(self.config, "ref_hint_refresh_interval", 2)))

    def _new_state(self) -> RefHintCacheState[HintValue]:
        return RefHintCacheState(
            refresh_interval=self._refresh_interval(),
            strategy=self._strategy(),
        )

    def _state_for(self, owner: nn.Module) -> RefHintCacheState[HintValue]:
        owner_id = id(owner)
        state = self._states.get(owner_id)
        if state is None:
            state = self._new_state()
            self._states[owner_id] = state
        return state

    def _check_lossy_ack(self) -> None:
        refresh_interval = self._refresh_interval()
        if refresh_interval >= 2 and not getattr(self.config, "ref_hint_acknowledge_lossy", False):
            raise ValueError(
                "The 'ref_hint' cache is approximate: reusing or forecasting "
                f"reference hints (strategy={self._strategy()!r}, "
                f"ref_hint_refresh_interval={refresh_interval}) can change output quality. "
                "Set DiffusionCacheConfig.ref_hint_acknowledge_lossy=True to opt in, "
                "or use ref_hint_refresh_interval=1 for recompute-every-step."
            )

    @staticmethod
    def _as_hints(value: T) -> HintValue:
        if not isinstance(value, list) or not all(torch.is_tensor(item) for item in value):
            raise TypeError(f"ModelRegion.REFERENCE_HINTS must return list[torch.Tensor], got {type(value).__name__}")
        return cast(HintValue, value)

    @staticmethod
    def _forecast(
        history: tuple[tuple[int, HintValue], ...],
        step: int,
    ) -> HintValue:
        if len(history) != 2:
            raise RuntimeError("forecast50 requires exactly two retained fresh hint observations")
        (previous_step, previous), (current_step, current) = history
        if len(previous) != len(current):
            raise RuntimeError("reference-hint history changed shape between refreshes")
        step_distance = max(current_step - previous_step, 1)
        alpha = min(max((step - current_step) / step_distance, 0.0), _FORECAST_ALPHA_MAX)
        # The nominal gain is 0.5, but the first skipped step can otherwise
        # extrapolate by half of the entire calibration delta.  A small trust
        # region prevents that one-step overshoot while retaining forecasting.
        correction = min(_FORECAST_GAIN * alpha, _FORECAST_CORRECTION_MAX)
        return [
            current_hint + (current_hint - previous_hint) * correction
            for previous_hint, current_hint in zip(previous, current)
        ]

    def enable(self, pipeline: object) -> None:
        self._check_lossy_ack()
        transformers = self._get_transformers(pipeline)
        for transformer in transformers:
            if getattr(transformer, "vace_blocks", None) is None:
                raise ValueError(
                    f"{transformer.__class__.__name__} does not expose a reference-hint "
                    "model region. The 'ref_hint' backend currently supports "
                    "reference-conditioned Wan-VACE transformers."
                )

        self._states = {id(transformer): self._new_state() for transformer in transformers}
        self.enabled = True
        logger.info(
            "Reference-hint cache enabled: strategy=%s refresh_interval=%d owners=%d",
            self._strategy(),
            self._refresh_interval(),
            len(transformers),
        )

    def refresh(self, pipeline: object, num_inference_steps: int, verbose: bool = True) -> None:
        for transformer in self._get_transformers(pipeline):
            self._state_for(transformer).reset()
        if verbose:
            logger.debug(
                "Reference-hint cache reset for new %d-step generation",
                num_inference_steps,
            )

    def finish_request(self, pipeline: object) -> None:
        """Report cache activity, then release retained request tensors."""
        states = [self._state_for(transformer) for transformer in self._get_transformers(pipeline)]
        hits = sum(state.hits for state in states)
        misses = sum(state.misses for state in states)
        refreshes = sum(state.refreshes for state in states)
        logger.info(
            "Reference-hint cache request summary: strategy=%s "
            "refresh_interval=%d hits=%d misses=%d refreshes=%d owners=%d",
            self._strategy(),
            self._refresh_interval(),
            hits,
            misses,
            refreshes,
            len(states),
        )
        for state in states:
            state.reset()

    def get_model_region_handler(self) -> ModelRegionHandler:
        """Install this backend only in the active request's ForwardContext."""
        return self

    def execute(
        self,
        region: ModelRegion,
        owner: nn.Module,
        compute: Callable[[], T],
    ) -> T:
        """Handle a reference-hint region; pass unrelated regions through."""
        if not self.enabled or region is not ModelRegion.REFERENCE_HINTS:
            return compute()
        if self._refresh_interval() == 1:
            # K=1 never reuses hints. Avoid retaining full hint tensors for a
            # mode that is intentionally equivalent to direct computation.
            return compute()

        step = get_forward_context().denoise_step_idx if is_forward_context_available() else None
        state = self._state_for(owner)
        branch, should_refresh = state.begin_call(step)
        if should_refresh:
            value = compute()
            state.store(branch, step, self._as_hints(value))
            return value

        assert branch is not None and step is not None
        history = state.history(branch)
        strategy = self._strategy()
        if strategy == "reuse":
            return cast(T, history[-1][1])
        return cast(T, self._forecast(history, step))
