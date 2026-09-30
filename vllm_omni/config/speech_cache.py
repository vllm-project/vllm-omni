# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""API-process speech cache budgets, independent of stage engine caches."""

from pydantic import ConfigDict, Field
from vllm.config.utils import config


@config(config=ConfigDict(strict=True, extra="forbid", frozen=True))
class SpeechCacheConfig:
    """LRU limits, not preallocated memory. Zero disables the respective cache."""

    resolve_max_bytes: int = Field(default=4 * 1024**3, ge=0)
    resolve_max_entries: int = Field(default=2048, ge=0)
    speaker_max_bytes: int = Field(default=512 * 1024**2, ge=0)
