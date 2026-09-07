# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Declarative request-scoped Cache-DiT lifecycle."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vllm_omni.diffusion.cache.cachedit.backend import CacheDiTBackend
from vllm_omni.diffusion.data import DiffusionCacheConfig


@dataclass(frozen=True)
class CacheDiTRequestSpec:
    """Desired Cache-DiT state for one request.

    ``installation_key`` identifies configurations that require reinstalling
    hooks when switching between them. Step-count changes only require a
    context refresh.
    """

    installation_key: str
    cache_config: DiffusionCacheConfig
    num_inference_steps: int


class RequestScopedCacheDiTRuntime:
    """Reconcile declarative request state with installed Cache-DiT hooks."""

    def __init__(self, pipeline: Any) -> None:
        self._pipeline = pipeline
        self._backend: CacheDiTBackend | None = None
        self._installation_key: str | None = None

    @property
    def is_enabled(self) -> bool:
        return self._backend is not None and self._backend.is_enabled()

    def adopt(self, backend: CacheDiTBackend, *, installation_key: str) -> None:
        """Assume ownership of an already-enabled startup backend."""

        if self._backend is not None:
            if self._backend is backend and self._installation_key == installation_key:
                return
            raise RuntimeError(
                f"Cache-DiT is already installed with key {self._installation_key!r}; cannot adopt {installation_key!r}"
            )
        if not backend.is_enabled():
            raise ValueError("Cannot adopt a Cache-DiT backend before it is enabled")

        self._backend = backend
        self._installation_key = installation_key

    def prepare(self, spec: CacheDiTRequestSpec | None) -> None:
        """Apply the minimal transition needed before the request denoise.

        ``None`` disables installed hooks. An unchanged installation key only
        refreshes request context; a changed key disables the previous hooks
        before installing the requested profile.
        """

        desired_key = None if spec is None else spec.installation_key
        if self._installation_key != desired_key:
            self.disable()

        if spec is None:
            return

        if self._backend is None:
            backend = CacheDiTBackend(spec.cache_config)
            backend.enable(self._pipeline)
            self.adopt(backend, installation_key=spec.installation_key)

        assert self._backend is not None
        self._backend.refresh(
            self._pipeline,
            num_inference_steps=spec.num_inference_steps,
        )

    def disable(self) -> None:
        try:
            if self._backend is not None:
                self._backend.disable(self._pipeline)
        finally:
            self._backend = None
            self._installation_key = None


__all__ = [
    "CacheDiTRequestSpec",
    "RequestScopedCacheDiTRuntime",
]
