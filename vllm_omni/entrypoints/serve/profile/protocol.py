# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Request models for Omni profile control routes."""

from pydantic import BaseModel, Field


class ProfileRequest(BaseModel):
    """Request model for profiling endpoints."""

    stages: list[int] | None = Field(
        default=None,
        description="List of stage IDs to profile. If None, profiles all stages.",
    )
