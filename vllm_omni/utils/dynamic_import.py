from __future__ import annotations

from collections.abc import Callable

from vllm.utils.import_utils import resolve_obj_by_qualname


def load_callable(path: str, *, context: str) -> Callable:
    """Load a callable from a dotted path string.

    This is the single approved entry point for user-configured dotted paths
    (stage hooks, connector builders, diffusion registry functions, etc.).

    Args:
        path: Fully qualified dotted path, e.g. "my.module.my_func".
        context: Human-readable description of what is being loaded,
                 used in error messages.

    Returns:
        The resolved callable.
    """
    if not path or "." not in path:
        raise ValueError(f"[{context}] Invalid dotted path {path!r}: expected 'package.module.callable_name'")
    try:
        obj = resolve_obj_by_qualname(path)
    except (ImportError, AttributeError, ValueError) as e:
        raise ValueError(f"[{context}] Failed to load {path!r}: {e}") from e
    if not callable(obj):
        raise ValueError(f"[{context}] {path!r} resolved to {type(obj).__name__}, expected a callable")
    return obj
