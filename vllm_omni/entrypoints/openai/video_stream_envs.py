# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Environment variables for the streaming video OpenAI entrypoint."""

import logging
import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    VLLM_VIDEO_AUDIO_DELTA_MODE: Literal["fast", "slow"] = "fast"
    VLLM_VIDEO_ASYNC_CHUNK: Literal["on", "off"] = "on"

logger = logging.getLogger(__name__)
_warned_invalid_envs: set[tuple[str, str]] = set()
_VIDEO_AUDIO_DELTA_MODE = "VLLM_VIDEO_AUDIO_DELTA_MODE"
_VIDEO_ASYNC_CHUNK = "VLLM_VIDEO_ASYNC_CHUNK"


def _choice_env(
    name: str,
    default: str,
    allowed: tuple[str, ...],
) -> str:
    value = os.getenv(name, default).strip().lower()
    if value in allowed:
        return value
    warning_key = (name, value)
    if warning_key not in _warned_invalid_envs:
        logger.warning("%s=%s not recognized; falling back to %r", name, value, default)
        _warned_invalid_envs.add(warning_key)
    return default


environment_variables: dict[str, Callable[[], str]] = {
    _VIDEO_AUDIO_DELTA_MODE: lambda: _choice_env(
        _VIDEO_AUDIO_DELTA_MODE,
        "fast",
        ("fast", "slow"),
    ),
    _VIDEO_ASYNC_CHUNK: lambda: _choice_env(
        _VIDEO_ASYNC_CHUNK,
        "on",
        ("on", "off"),
    ),
}


def __getattr__(name: str) -> str:
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return list(environment_variables.keys())


def is_set(name: str) -> bool:
    if name in environment_variables:
        return name in os.environ
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
