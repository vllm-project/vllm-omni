# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Lightweight Praxis session settings shared by the frontend and native worker."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

DEFAULT_MAX_NUM_STEPS = 16


def decode_prompt(value: object) -> str:
    """Bound task text to 64 KiB before UTF-8 decoding or tokenization."""
    if not isinstance(value, (str, bytes)):
        raise ValueError("task/prompt must be a string or bytes")
    if len(value) > 65536 or (isinstance(value, str) and len(value.encode("utf-8")) > 65536):
        raise ValueError("task/prompt exceeds 64 KiB")
    return value.decode("utf-8", "replace") if isinstance(value, bytes) else value


@dataclass(frozen=True)
class SamplerSettings:
    """Shared sampler defaults; model_config may override these fields."""

    num_steps: int = 2
    t_shift: float = 12.5
    cfg_scale: float = 1.0
    cfg_video_scale: float = 4.0

    def __post_init__(self) -> None:
        if type(self.num_steps) is not int or self.num_steps < 1:
            raise ValueError("num_steps must be a positive integer")
        for name in ("cfg_scale", "cfg_video_scale", "t_shift"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid {name}: {value!r}")
        if self.t_shift == 0:
            raise ValueError("t_shift must be a finite float > 0")

    @property
    def guided(self) -> bool:
        return self.cfg_scale != 1.0 or self.cfg_video_scale != 1.0


@dataclass(frozen=True)
class PraxisSessionOptions(SamplerSettings):
    max_context_chunks: int = 1
    exec_horizon: int | None = 24
    control_mode: str = "droid_impedance"
    context_mode: str = "observed"

    def __post_init__(self) -> None:
        super().__post_init__()
        for name, high in (("max_context_chunks", 3), ("exec_horizon", 24)):
            value = getattr(self, name)
            if name == "exec_horizon" and value is None:
                continue
            low = 0 if name == "max_context_chunks" else 1
            if type(value) is not int or value < low or (high is not None and value > high):
                raise ValueError(f"Invalid {name}: {value!r}")
        if self.control_mode not in ("none", "droid_impedance"):
            raise ValueError("Unsupported control_mode")
        if self.context_mode not in ("observed", "stateless"):
            raise ValueError("Only observed and stateless context are supported")

    @property
    def stateless(self) -> bool:
        return (
            self.context_mode == "stateless"
            or self.max_context_chunks == 0
            or (self.exec_horizon is not None and self.exec_horizon < 24)
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
