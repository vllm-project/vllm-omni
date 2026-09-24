# SPDX-License-Identifier: Apache-2.0
"""Explicit inference settings, separate from exported artifact metadata."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any

from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.config import (
    Cosmos3NanoSimBimanualManifest,
    deploy_option,
)


@dataclass(frozen=True)
class Cosmos3NanoSimBimanualInferenceConfig:
    frame_sigma_schedules: tuple[tuple[float, ...], ...]
    window_frames: int
    sink_frames: int
    history_mode: str

    @classmethod
    def from_od_config(
        cls, config: Any, manifest: Cosmos3NanoSimBimanualManifest
    ) -> Cosmos3NanoSimBimanualInferenceConfig:
        overrides = deploy_option(config, "inference_overrides", {})
        if not isinstance(overrides, dict):
            raise ValueError("inference_overrides must be a mapping")
        unknown = set(overrides) - {
            "frame_sigma_schedules",
            "history_mode",
            "max_num_frames",
            "kv_cache_inference_size",
            "attention_sink_size",
        }
        if unknown:
            raise ValueError(f"Unknown Bimanual inference_overrides: {sorted(unknown)}")
        schedules = overrides.get("frame_sigma_schedules", [manifest.t_list])
        if not isinstance(schedules, (list, tuple)) or not schedules:
            raise ValueError("frame_sigma_schedules must contain at least one sigma sequence")
        for schedule in schedules:
            if not isinstance(schedule, (list, tuple)) or not schedule:
                raise ValueError("Each frame sigma schedule must be a nonempty sequence")
            if any(
                isinstance(sigma, bool)
                or not isinstance(sigma, (int, float))
                or not math.isfinite(sigma)
                or not 0 < sigma <= 1
                for sigma in schedule
            ):
                raise ValueError("Frame sigmas must be finite numbers in (0, 1]; omit terminal zero")
            if schedule[0] != 1 or any(a <= b for a, b in zip(schedule, schedule[1:])):
                raise ValueError("Frame sigmas must start at 1 and strictly decrease")
        mode = overrides.get("history_mode", "sliding")
        if mode not in ("sliding", "full"):
            raise ValueError("history_mode must be 'sliding' or 'full'")
        sink = overrides.get("attention_sink_size", 0 if mode == "full" else manifest.sink_frames)
        if isinstance(sink, bool) or not isinstance(sink, int) or sink < 0:
            raise ValueError("attention_sink_size must be a non-negative integer")
        if mode == "full":
            if "kv_cache_inference_size" in overrides or sink:
                raise ValueError("Full history requires zero sinks and max_num_frames instead of a sliding window")
            frames = overrides.get("max_num_frames")
            factor = manifest.temporal_compression_factor
            if isinstance(frames, bool) or not isinstance(frames, int) or frames <= 0 or (frames - 1) % factor:
                raise ValueError(f"Full history requires max_num_frames = 1 + {factor} * N")
            window = (frames - 1) // factor + 1
        else:
            if "max_num_frames" in overrides:
                raise ValueError("max_num_frames requires history_mode='full'")
            window = overrides.get("kv_cache_inference_size", manifest.window_frames)
            if isinstance(window, bool) or not isinstance(window, int) or window <= 0:
                raise ValueError("kv_cache_inference_size must be a positive integer")
        return cls(tuple(tuple(float(sigma) for sigma in row) for row in schedules), window, sink, mode)

    def sigmas_for_frame(self, frame_idx: int) -> tuple[float, ...]:
        if frame_idx < 0:
            raise ValueError("frame_idx must be non-negative")
        return self.frame_sigma_schedules[min(frame_idx, len(self.frame_sigma_schedules) - 1)]

    @property
    def num_steps(self) -> int:
        return max(map(len, self.frame_sigma_schedules))

    def validate_target(self, target_frame: int) -> None:
        if self.history_mode == "full" and target_frame > self.window_frames:
            raise ValueError(
                f"Full-history rollout needs {target_frame} latent frames; configured capacity is {self.window_frames}"
            )

    @property
    def digest(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()
