"""
Scheduling components for vLLM-Omni.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_LAZY_EXPORTS = {
    "OmniARAsyncScheduler": (".omni_ar_scheduler", "OmniARAsyncScheduler"),
    "OmniARScheduler": (".omni_ar_scheduler", "OmniARScheduler"),
    "OmniGenerationScheduler": (".omni_generation_scheduler", "OmniGenerationScheduler"),
    "OmniNewRequestData": (".output", "OmniNewRequestData"),
}


def __getattr__(name: str) -> Any:
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attribute_name = target
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value


__all__ = [
    "OmniARAsyncScheduler",
    "OmniARScheduler",
    "OmniGenerationScheduler",
    "OmniNewRequestData",
]
