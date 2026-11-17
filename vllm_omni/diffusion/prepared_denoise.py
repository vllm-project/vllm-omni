# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Request-owned prepared inputs for repeated denoiser calls."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Generic, TypeVar, cast

T = TypeVar("T")


@dataclass
class PreparedDenoiseInputs(Generic[T]):
    """Publish a complete model-specific result for one request or branch.

    An explicit invalidation advances the generation. Failed preparation never
    publishes a partial result, and callers cannot bind an invalid generation.
    """

    generation: int = 0
    _published_generation: int | None = None
    _value: T | None = None

    def prepare(self, build: Callable[[], T]) -> T:
        if self._published_generation == self.generation:
            return self.bind()
        value = build()
        self._value = value
        self._published_generation = self.generation
        return value

    def bind(self) -> T:
        if self._published_generation != self.generation:
            raise RuntimeError("Prepared denoise inputs are unavailable or stale")
        return cast(T, self._value)

    def invalidate(self) -> None:
        self.generation += 1
        self._value = None
        self._published_generation = None

    def release(self) -> None:
        self.invalidate()


@contextmanager
def prepared_denoise_scope() -> Iterator[PreparedDenoiseInputs[object]]:
    """Own prepared inputs across a request-mode denoise loop.

    Enter after encoding and layout finalization. The same store used by step
    execution is released on success, cancellation, or any exception.
    """
    prepared = PreparedDenoiseInputs[object]()
    try:
        yield prepared
    finally:
        prepared.release()
