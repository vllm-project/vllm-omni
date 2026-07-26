# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compatibility helpers for vLLM-Ascend configuration layout changes."""

from __future__ import annotations

from typing import Any


def profiling_chunk_enabled(ascend_config: Any) -> bool:
    """Return whether profiling chunks are enabled across old/new layouts."""
    scheduler_config = getattr(ascend_config, "scheduler_config", None)
    profiling_config = getattr(scheduler_config, "profiling_chunk_config", None)
    if profiling_config is None:
        profiling_config = getattr(ascend_config, "profiling_chunk_config", None)
    return bool(getattr(profiling_config, "enabled", False))


def async_exponential_enabled(ascend_config: Any) -> bool:
    """Preserve the removed legacy sampler flag when an old release has it."""
    return bool(getattr(ascend_config, "enable_async_exponential", False))
