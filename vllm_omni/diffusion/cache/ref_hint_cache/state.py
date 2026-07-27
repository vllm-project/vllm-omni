# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Request-local scheduling and history for reference-hint acceleration.

The state is independent of any model definition.  A framework-side model
region handler calls :meth:`begin_call` exactly once for each invocation of a
reference-hint region.  Values are isolated by both model owner and CFG branch;
the backend owns one state object per model owner.

``branch`` is the call index within a denoising step.  Current Wan-VACE
scheduling invokes each transformer once for batched/no CFG or in a stable
positive/negative order for sequential CFG.  An unknown step is always a safe
direct-compute path and is never cached.
"""

from __future__ import annotations

from typing import Generic, TypeVar

ValueT = TypeVar("ValueT")

_REUSE = "reuse"
_FORECAST50 = "forecast50"
SUPPORTED_REF_HINT_STRATEGIES = frozenset({_REUSE, _FORECAST50})


class RefHintCacheState(Generic[ValueT]):
    """Schedule refreshes and retain strategy-specific history per CFG branch.

    ``reuse`` keeps one fresh value; ``forecast50`` keeps the two observations
    required by its first-order predictor.
    """

    def __init__(self, refresh_interval: int = 2, strategy: str = _REUSE):
        if strategy not in SUPPORTED_REF_HINT_STRATEGIES:
            supported = ", ".join(sorted(SUPPORTED_REF_HINT_STRATEGIES))
            raise ValueError(f"Unsupported ref_hint strategy {strategy!r}; expected one of: {supported}")
        self.refresh_interval = max(1, int(refresh_interval))
        self.strategy = strategy
        self._history: dict[int, list[tuple[int, ValueT]]] = {}
        self._last_step: int | None = None
        self._call_idx: int = 0
        self.hits: int = 0
        self.misses: int = 0

    def reset(self) -> None:
        """Clear retained values and counters for a new generation."""
        self._history.clear()
        self._last_step = None
        self._call_idx = 0
        self.hits = 0
        self.misses = 0

    def begin_call(self, step: int | None) -> tuple[int | None, bool]:
        """Return ``(branch, should_refresh)`` for this region invocation."""
        if step is None:
            return None, True
        if step != self._last_step:
            self._last_step = step
            self._call_idx = 0
        else:
            self._call_idx += 1
        branch = self._call_idx
        history = self._history.get(branch, [])

        if self.refresh_interval == 1 or not history:
            return branch, True
        if self.strategy == _FORECAST50 and len(history) < 2:
            # Two genuine observations calibrate the first-order predictor.
            return branch, True

        last_fresh_step = history[-1][0]
        should_refresh = step - last_fresh_step >= self.refresh_interval
        return branch, should_refresh

    def history(self, branch: int) -> tuple[tuple[int, ValueT], ...]:
        """Return retained fresh values for a reuse call and count one hit."""
        self.hits += 1
        return tuple(self._history[branch])

    def store(self, branch: int | None, step: int | None, value: ValueT) -> None:
        """Retain a fresh value; unknown branch/step calls are deliberate no-ops."""
        if branch is None or step is None:
            return
        self.misses += 1
        history = self._history.setdefault(branch, [])
        history.append((step, value))
        retained_values = 2 if self.strategy == _FORECAST50 else 1
        del history[:-retained_values]
