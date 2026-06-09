"""Lightweight NVTX helpers for Moshi request profiling."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import wraps
from typing import Any, ParamSpec, Protocol, TypeVar


class _NVTXBackend(Protocol):
    def range_push(self, message: str) -> object: ...

    def range_pop(self) -> object: ...


def _resolve_nvtx_backend() -> _NVTXBackend | None:
    try:
        import torch
    except Exception:
        return None

    nvtx = getattr(getattr(torch, "cuda", None), "nvtx", None)
    if nvtx is None:
        return None
    if not hasattr(nvtx, "range_push") or not hasattr(nvtx, "range_pop"):
        return None
    return nvtx


_NVTX_BACKEND = _resolve_nvtx_backend()

P = ParamSpec("P")
R = TypeVar("R")
_LabelSpec = str | Callable[..., str]


@contextmanager
def nvtx_range(message: str) -> Iterator[None]:
    """Push an NVTX range when available, otherwise behave as a no-op."""
    pushed = nvtx_range_push(message)
    try:
        yield
    finally:
        if pushed:
            nvtx_range_pop()


def nvtx_range_push(message: str) -> bool:
    """Push an NVTX range and return whether it was pushed."""
    backend = _NVTX_BACKEND
    if backend is None:
        return False
    try:
        backend.range_push(message)
        return True
    except Exception:
        return False


def nvtx_range_pop() -> None:
    """Pop an NVTX range when available."""
    backend = _NVTX_BACKEND
    if backend is None:
        return
    try:
        backend.range_pop()
    except Exception:
        pass


def _resolve_label(label: _LabelSpec, *args: Any, **kwargs: Any) -> str:
    if callable(label):
        return label(*args, **kwargs)
    return label


def nvtx_annotate(label: _LabelSpec) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorate sync or async functions with an NVTX range."""

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        if inspect.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> Any:
                with nvtx_range(_resolve_label(label, *args, **kwargs)):
                    return await func(*args, **kwargs)

            return async_wrapper

        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> Any:
            with nvtx_range(_resolve_label(label, *args, **kwargs)):
                return func(*args, **kwargs)

        return wrapper

    return decorator
