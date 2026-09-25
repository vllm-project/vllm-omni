# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""API-process speech cache budgets, independent of stage engine caches."""

from dataclasses import dataclass, fields


@dataclass(frozen=True)
class SpeechCacheConfig:
    """LRU limits, not preallocated memory. Zero disables the respective cache."""

    resolve_max_bytes: int = 4 * 1024**3
    resolve_max_entries: int = 2048
    speaker_max_bytes: int = 512 * 1024**2

    def __post_init__(self) -> None:
        for item in fields(self):
            value = getattr(self, item.name)
            if type(value) is not int or value < 0:
                raise ValueError(f"speech_cache.{item.name} must be a non-negative integer, got {value!r}")
