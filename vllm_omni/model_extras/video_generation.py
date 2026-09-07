# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class VideoGenerationDefaults:
    """Model-owned defaults consumed by shared video examples."""

    width: int
    height: int
    num_frames: int
    num_inference_steps: int
    fps: float
    output: str | None = None
    guidance_scale: float | None = None
    flow_shift: float | None = None
    dimension_multiple: int = 16
    default_negative_prompt: str | None = ""
    duration_seconds: float | None = None

    @property
    def max_area(self) -> int:
        return self.width * self.height

    def cli_defaults(self) -> dict[str, Any]:
        defaults: dict[str, Any] = {
            "height": self.height,
            "width": self.width,
            "num_frames": self.num_frames,
            "num_inference_steps": self.num_inference_steps,
            "fps": self.fps,
        }
        if self.output is not None:
            defaults["output"] = self.output
        if self.guidance_scale is not None:
            defaults["guidance_scale"] = self.guidance_scale
        if self.flow_shift is not None:
            defaults["flow_shift"] = self.flow_shift
        return defaults
