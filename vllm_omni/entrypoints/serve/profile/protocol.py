# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Request models for Omni profile control routes."""

from typing import Any

from pydantic import BaseModel, Field


class ProfileRequest(BaseModel):
    """Request model for profiling endpoints.

    ``stages`` is Omni-specific. The other fields match upstream
    ``StartProfileRequest`` and are forwarded to ``start_profile``.
    """

    stages: list[int] | None = Field(
        default=None,
        description="List of stage IDs to profile. If None, profiles all stages.",
    )
    profile_prefix: str | None = None
    delay_iterations: int | None = None
    max_iterations: int | None = None
    profiler_kwargs: dict[str, Any] | None = None
