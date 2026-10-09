# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class RunRequest(BaseModel):
    """Run one stage: stage 0 takes a prompt dict, later stages the stage_input returned by the previous call."""

    stage_id: int = 0
    stage_input: dict[str, Any] | str
    sampling_params: list[dict[str, Any]] | None = None


class RunResponse(BaseModel):
    """The next call's stage_id and stage_input, or the final stage's output."""

    stage_id: int | None = None
    stage_input: str | None = None
    output: str | None = None
