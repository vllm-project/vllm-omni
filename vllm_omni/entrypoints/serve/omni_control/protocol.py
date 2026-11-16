# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Request models for Omni server control routes."""

from pydantic import BaseModel, Field


class OmniSleepRequest(BaseModel):
    stage_ids: list[int] = Field(..., min_length=1)
    level: int = Field(default=2, ge=0)


class OmniWakeupRequest(BaseModel):
    stage_ids: list[int] = Field(..., min_length=1)
