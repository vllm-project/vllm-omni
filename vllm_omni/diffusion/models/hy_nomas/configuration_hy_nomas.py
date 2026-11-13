# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from typing import Any

from .history_sampling import ARCHIVED_HISTORY_MODES

MODEL_ID = "tencent/Hy-Nomas-Diffusers"
REQUIRED_FILES = (
    "modular_model_index.json",
    "transformer/config.json",
    "decoder/config.json",
    "scheduler/scheduler_config.json",
)


@dataclass(frozen=True)
class HyNomasARConfig:
    height: int = 704
    width: int = 1280
    chunk_latent_frames: int = 16
    max_latent_frames: int = 120
    max_segments: int = 16
    session_capacity: int = 1
    history_mode: str = "full"
    history_latent_frames: int = 72
    history_temporal_gamma: float = 2.0
    max_total_latent_frames: int = 4096
    max_total_segments: int = 1024
    history_archive_max_bytes: int = 0
    history_archive_max_frames: int = 4096

    @property
    def max_chunk_latent_frames(self) -> int:
        # Reference chunk-AR merges tails shorter than half a nominal chunk.
        return self.chunk_latent_frames + max(0, self.chunk_latent_frames // 2 - 1)

    @classmethod
    def from_args(cls, args: Mapping[str, Any]) -> HyNomasARConfig:
        values = {name: args.get(f"ar_{name}", info.default) for name, info in cls.__dataclass_fields__.items()}
        mode = values.pop("history_mode")
        gamma = values.pop("history_temporal_gamma")
        if isinstance(gamma, bool) or not isinstance(gamma, Real) or not math.isfinite(gamma) or gamma <= 0:
            raise ValueError("Hy-Nomas ar_history_temporal_gamma must be finite and positive.")
        archive_bytes = values.pop("history_archive_max_bytes")
        if type(archive_bytes) is not int or archive_bytes < 0:
            raise ValueError("Hy-Nomas ar_history_archive_max_bytes must be a nonnegative integer.")
        if archive_bytes and mode == "full":
            raise ValueError("Hy-Nomas CPU history archival requires a bounded history mode.")
        if mode not in ("full", "recent", *ARCHIVED_HISTORY_MODES):
            raise ValueError(
                "Hy-Nomas ar_history_mode must be full, recent, uniform, recent_mix_uniform or temporal_context."
            )
        if mode in ARCHIVED_HISTORY_MODES and not archive_bytes:
            raise ValueError("Hy-Nomas archived history sampling requires a positive archive byte quota.")
        for name, value in values.items():
            if type(value) is not int or value <= 0:
                raise ValueError(f"Hy-Nomas ar_{name} must be a positive integer.")
        if values["height"] % 32 or values["width"] % 32:
            raise ValueError("Hy-Nomas ar_height and ar_width must be multiples of 32.")
        if values["chunk_latent_frames"] > values["max_latent_frames"]:
            raise ValueError("Hy-Nomas ar_chunk_latent_frames must not exceed ar_max_latent_frames.")
        if mode == "recent" and values["chunk_latent_frames"] > values["history_latent_frames"]:
            raise ValueError("Hy-Nomas nominal chunk size must not exceed its recent history budget.")
        return cls(
            history_mode=mode, history_temporal_gamma=float(gamma), history_archive_max_bytes=archive_bytes, **values
        )
