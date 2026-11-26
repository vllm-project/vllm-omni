# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Portions derived from https://github.com/step-out/Duplex-MPE
# (Copyright (c) 2026 MPeval authors, MIT License; see benchmarks/duplex_mpe/LICENSE).
"""Causal Silero voice activity detection with Duplex-MPE hysteresis and hangover."""

from __future__ import annotations

import importlib
import importlib.metadata
import math
import threading
from collections.abc import Callable
from dataclasses import dataclass
from math import gcd
from typing import Protocol

import numpy as np
import torch
from numpy.typing import NDArray
from pydantic import JsonValue
from scipy.signal import resample_poly

ENDPOINT_EPSILON_SECONDS = 1e-9
SILERO_MODEL_LOAD_LOCK = threading.Lock()
SILERO_UNINSTALLED_SENTINEL = "uninstalled"


class SileroVadModel(Protocol):
    def reset_states(self) -> None: ...

    def __call__(self, audio_tensor: torch.Tensor, sample_rate_hz: int) -> torch.Tensor: ...


SileroVadModelFactory = Callable[[], SileroVadModel]


def silero_package_version() -> str:
    try:
        return importlib.metadata.version("silero-vad")
    except importlib.metadata.PackageNotFoundError:
        return SILERO_UNINSTALLED_SENTINEL


def ensure_default_silero_vad_importable() -> object:
    try:
        silero_module = importlib.import_module("silero_vad")
    except ModuleNotFoundError as exc:
        if exc.name == "silero_vad":
            raise ModuleNotFoundError(
                "Duplex-MPE speech activity detection requires `silero-vad`. "
                'Install it with `pip install "silero-vad==6.2.1"` (or `pip install -e ".[server-vad]"`).'
            ) from exc
        raise
    if not callable(getattr(silero_module, "load_silero_vad", None)):
        raise RuntimeError("Installed `silero_vad` package does not expose callable `load_silero_vad`")
    return silero_module


def load_default_silero_vad_model() -> SileroVadModel:
    silero_module = ensure_default_silero_vad_importable()
    loader = getattr(silero_module, "load_silero_vad", None)
    if not callable(loader):
        raise RuntimeError("Installed `silero_vad` package does not expose callable `load_silero_vad`")
    loaded_model: SileroVadModel = loader()
    return loaded_model


@dataclass(frozen=True, kw_only=True)
class SileroActivityConfig:
    onset_probability: float = 0.5
    offset_probability: float = 0.35
    hangover_milliseconds: float = 800.0
    target_endpoint_hangover_milliseconds: float = 2000.0
    minimum_speech_milliseconds: float = 120.0
    silero_sample_rate_hz: int = 16000
    silero_window_samples: int = 512

    def __post_init__(self) -> None:
        if not (
            math.isfinite(self.onset_probability)
            and math.isfinite(self.offset_probability)
            and 0.0 < self.offset_probability <= self.onset_probability <= 1.0
        ):
            raise ValueError("Invalid VAD onset/offset probability thresholds")
        if (
            not math.isfinite(self.hangover_milliseconds)
            or self.hangover_milliseconds <= 0.0
            or not math.isfinite(self.target_endpoint_hangover_milliseconds)
            or self.target_endpoint_hangover_milliseconds <= 0.0
            or not math.isfinite(self.minimum_speech_milliseconds)
            or self.minimum_speech_milliseconds <= 0.0
        ):
            raise ValueError("VAD duration thresholds must be finite and positive")
        if self.silero_sample_rate_hz <= 0 or self.silero_window_samples <= 0:
            raise ValueError("silero_sample_rate_hz and silero_window_samples must be positive")

    @property
    def frame_milliseconds(self) -> float:
        return 1000.0 * self.silero_window_samples / self.silero_sample_rate_hz

    @property
    def hangover_frames(self) -> int:
        return max(1, int(round(self.hangover_milliseconds / self.frame_milliseconds)))

    @property
    def minimum_speech_frames(self) -> int:
        return max(1, int(round(self.minimum_speech_milliseconds / self.frame_milliseconds)))

    def to_report_dict(self, *, custom_vad_injected: bool = False) -> dict[str, JsonValue]:
        payload: dict[str, JsonValue] = {
            "detector": "silero_streaming_hysteresis_hangover",
            "silero_vad_version": silero_package_version(),
            "silero_sr": self.silero_sample_rate_hz,
            "silero_window": self.silero_window_samples,
            "frame_ms": self.frame_milliseconds,
            "onset_prob": self.onset_probability,
            "offset_prob": self.offset_probability,
            "hangover_ms": self.hangover_milliseconds,
            "hangover_ms_role": "episode merge distance (M3 occupancy)",
            "t_endpoint_hangover_ms": self.target_endpoint_hangover_milliseconds,
            "t_endpoint_hangover_role": (
                "silence required to declare the T response finished; "
                "applied to raw voicing runs, NOT to episode merging"
            ),
            "min_speech_ms": self.minimum_speech_milliseconds,
            "frames_hangover": self.hangover_frames,
            "frames_min_speech": self.minimum_speech_frames,
        }
        if custom_vad_injected:
            payload["custom_vad_injected"] = True
        return payload


class SileroActivityDetector:
    """Streaming causal speech segmenter tracking merged episodes and raw voicing runs."""

    def __init__(
        self,
        *,
        vad_config: SileroActivityConfig | None = None,
        sample_rate_hz: int = 24000,
        vad_model: SileroVadModel | None = None,
        resample_context_blocks: int = 3,
    ) -> None:
        if sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive")
        if resample_context_blocks < 0:
            raise ValueError("resample_context_blocks must be >= 0")
        self.vad_config: SileroActivityConfig = vad_config if vad_config is not None else SileroActivityConfig()
        self.sample_rate_hz: int = int(sample_rate_hz)
        # Each detector owns an isolated Silero RNN state so concurrent sessions never cross-contaminate.
        self.uses_custom_vad_model: bool = vad_model is not None
        if vad_model is not None:
            self.vad_model: SileroVadModel = vad_model
        else:
            with SILERO_MODEL_LOAD_LOCK:
                self.vad_model = load_default_silero_vad_model()
        self.vad_model.reset_states()

        rate_gcd = gcd(self.vad_config.silero_sample_rate_hz, self.sample_rate_hz)
        self.resample_up: int = self.vad_config.silero_sample_rate_hz // rate_gcd
        self.resample_down: int = self.sample_rate_hz // rate_gcd
        self.resample_context_blocks: int = int(resample_context_blocks)

        frame_numerator = self.sample_rate_hz * self.vad_config.silero_window_samples
        if frame_numerator % self.vad_config.silero_sample_rate_hz == 0:
            self.is_integer_frame_ratio: bool = True
            self.native_block_samples: int = frame_numerator // self.vad_config.silero_sample_rate_hz
            self.resampled_block_samples: int = self.vad_config.silero_window_samples
        else:
            # 22.05 kHz endpoints resample exact 441->320 GCD blocks before slicing 512-sample Silero windows.
            self.is_integer_frame_ratio = False
            self.native_block_samples = self.resample_down
            self.resampled_block_samples = self.resample_up

        self.is_in_speech: bool = False
        self.speech_run_frames: int = 0
        self.silence_run_frames: int = 0
        self.frames_seen: int = 0
        self.episodes: list[tuple[float, float]] = []
        self.open_episode_start_frame: int | None = None
        self.pending_episode_end_frame: int | None = None
        self.native_carry: NDArray[np.float32] = np.zeros(0, dtype=np.float32)
        self.resample_history: NDArray[np.float32] = np.zeros(0, dtype=np.float32)
        self.silero_frame_carry: NDArray[np.float32] = np.zeros(0, dtype=np.float32)

        self.voiced_runs: list[tuple[float, float]] = []
        self.is_voicing_active: bool = False
        self.voicing_run_frames: int = 0
        self.voicing_start_frame: int | None = None
        self.frame_probabilities: list[float] = []

    def frame_index_to_seconds(self, frame_index: int) -> float:
        return frame_index * self.vad_config.frame_milliseconds / 1000.0

    @property
    def now_seconds(self) -> float:
        return self.frame_index_to_seconds(self.frames_seen)

    @property
    def is_speaking(self) -> bool:
        return self.is_in_speech

    @property
    def is_voicing_now(self) -> bool:
        return self.is_voicing_active

    def last_voiced_end_seconds(self) -> float | None:
        if self.is_voicing_active:
            return self.frame_index_to_seconds(self.frames_seen)
        if self.voiced_runs:
            return self.voiced_runs[-1][1]
        return None

    def is_endpoint_confirmed(self, hangover_milliseconds: float) -> bool:
        if self.is_voicing_active:
            return False
        last_end_seconds = self.last_voiced_end_seconds()
        if last_end_seconds is None:
            return False
        return self.now_seconds - last_end_seconds >= hangover_milliseconds / 1000.0 - ENDPOINT_EPSILON_SECONDS

    def voicing_onset_at_or_after(self, start_threshold_seconds: float, *, since_run_index: int = 0) -> float | None:
        for run_start_seconds, _ in self.voiced_runs[since_run_index:]:
            if run_start_seconds >= start_threshold_seconds:
                return run_start_seconds
        if self.is_voicing_active and self.voicing_start_frame is not None:
            active_start_seconds = self.frame_index_to_seconds(self.voicing_start_frame)
            if active_start_seconds >= start_threshold_seconds:
                return active_start_seconds
        return None

    def resample_block_to_16khz(self, native_block: NDArray[np.float32]) -> NDArray[np.float32]:
        if self.resample_up == 1 and self.resample_down == 1:
            return native_block
        combined_history = np.concatenate([self.resample_history, native_block])
        resampled = resample_poly(
            combined_history.astype(np.float64),
            self.resample_up,
            self.resample_down,
        )
        history_limit = self.resample_context_blocks * self.native_block_samples
        if history_limit == 0:
            self.resample_history = np.zeros(0, dtype=np.float32)
        elif len(combined_history) > history_limit:
            self.resample_history = combined_history[-history_limit:]
        else:
            self.resample_history = combined_history
        target_length = self.resampled_block_samples
        if len(resampled) >= target_length:
            return resampled[-target_length:].astype(np.float32)
        return np.pad(resampled, (target_length - len(resampled), 0)).astype(np.float32)

    def step_voicing_state(
        self,
        *,
        is_above_onset: bool,
        is_below_offset: bool,
        minimum_speech_frames: int,
    ) -> None:
        if not self.is_voicing_active:
            if is_above_onset:
                self.voicing_run_frames += 1
                if self.voicing_run_frames >= minimum_speech_frames:
                    self.is_voicing_active = True
                    self.voicing_start_frame = self.frames_seen - self.voicing_run_frames
            else:
                self.voicing_run_frames = 0
        elif is_below_offset:
            if self.voicing_start_frame is not None:
                start_seconds = self.frame_index_to_seconds(self.voicing_start_frame)
                end_seconds = self.frame_index_to_seconds(self.frames_seen - 1)
                if end_seconds > start_seconds:
                    self.voiced_runs.append((round(start_seconds, 4), round(end_seconds, 4)))
            self.is_voicing_active = False
            self.voicing_run_frames = 0
            self.voicing_start_frame = None

    def step_silero_frame(
        self,
        frame_16khz: NDArray[np.float32],
        closed_episodes: list[tuple[float, float]],
    ) -> None:
        with torch.no_grad():
            probability = float(
                self.vad_model(
                    torch.from_numpy(np.ascontiguousarray(frame_16khz))[None, :],
                    self.vad_config.silero_sample_rate_hz,
                )
            )
        self.frame_probabilities.append(probability)
        self.frames_seen += 1
        is_above_onset = probability >= self.vad_config.onset_probability
        is_below_offset = probability < self.vad_config.offset_probability
        self.step_voicing_state(
            is_above_onset=is_above_onset,
            is_below_offset=is_below_offset,
            minimum_speech_frames=self.vad_config.minimum_speech_frames,
        )
        if not self.is_in_speech:
            if is_above_onset:
                self.speech_run_frames += 1
                if self.speech_run_frames >= self.vad_config.minimum_speech_frames:
                    self.is_in_speech = True
                    self.silence_run_frames = 0
                    self.open_episode_start_frame = self.frames_seen - self.speech_run_frames
            else:
                self.speech_run_frames = 0
        elif is_below_offset:
            self.silence_run_frames += 1
            if self.pending_episode_end_frame is None:
                self.pending_episode_end_frame = self.frames_seen - 1
            if self.silence_run_frames >= self.vad_config.hangover_frames:
                start_frame = self.open_episode_start_frame if self.open_episode_start_frame is not None else 0
                end_frame = (
                    self.pending_episode_end_frame if self.pending_episode_end_frame is not None else self.frames_seen
                )
                start_seconds = self.frame_index_to_seconds(start_frame)
                end_seconds = self.frame_index_to_seconds(end_frame)
                if end_seconds > start_seconds:
                    segment = (round(start_seconds, 4), round(end_seconds, 4))
                    self.episodes.append(segment)
                    closed_episodes.append(segment)
                self.is_in_speech = False
                self.speech_run_frames = 0
                self.silence_run_frames = 0
                self.open_episode_start_frame = None
                self.pending_episode_end_frame = None
        else:
            self.silence_run_frames = 0
            self.pending_episode_end_frame = None

    def push(self, audio_samples: NDArray[np.float32]) -> list[tuple[float, float]]:
        flat_samples: NDArray[np.float32] = np.asarray(audio_samples, dtype=np.float32).reshape(-1)
        if len(self.native_carry) > 0:
            flat_samples = np.concatenate([self.native_carry, flat_samples])
            self.native_carry = np.zeros(0, dtype=np.float32)
        block_length = self.native_block_samples
        full_blocks = len(flat_samples) // block_length
        if full_blocks == 0:
            self.native_carry = flat_samples
            return []
        remainder = flat_samples[full_blocks * block_length :]
        if len(remainder) > 0:
            self.native_carry = remainder

        closed_episodes: list[tuple[float, float]] = []
        if self.is_integer_frame_ratio:
            for block_index in range(full_blocks):
                native_block = flat_samples[block_index * block_length : (block_index + 1) * block_length]
                frame_16khz = self.resample_block_to_16khz(native_block)
                self.step_silero_frame(frame_16khz, closed_episodes)
        else:
            resampled_chunks: list[NDArray[np.float32]] = []
            if len(self.silero_frame_carry) > 0:
                resampled_chunks.append(self.silero_frame_carry)
                self.silero_frame_carry = np.zeros(0, dtype=np.float32)
            for block_index in range(full_blocks):
                native_block = flat_samples[block_index * block_length : (block_index + 1) * block_length]
                resampled_chunks.append(self.resample_block_to_16khz(native_block))
            merged_16khz = np.concatenate(resampled_chunks)
            window_samples = self.vad_config.silero_window_samples
            window_count = len(merged_16khz) // window_samples
            for window_index in range(window_count):
                frame_16khz = merged_16khz[window_index * window_samples : (window_index + 1) * window_samples]
                self.step_silero_frame(frame_16khz, closed_episodes)
            leftover_16khz = merged_16khz[window_count * window_samples :]
            if len(leftover_16khz) > 0:
                self.silero_frame_carry = leftover_16khz
        return closed_episodes

    def finalize(self) -> list[tuple[float, float]]:
        if self.is_voicing_active and self.voicing_start_frame is not None:
            voicing_start = self.frame_index_to_seconds(self.voicing_start_frame)
            voicing_end = self.frame_index_to_seconds(self.frames_seen)
            if voicing_end > voicing_start:
                self.voiced_runs.append((round(voicing_start, 4), round(voicing_end, 4)))
        self.is_voicing_active = False
        self.voicing_start_frame = None

        if self.is_in_speech and self.open_episode_start_frame is not None:
            episode_start = self.frame_index_to_seconds(self.open_episode_start_frame)
            episode_end = self.frame_index_to_seconds(self.frames_seen)
            self.is_in_speech = False
            self.open_episode_start_frame = None
            if episode_end > episode_start:
                segment = (round(episode_start, 4), round(episode_end, 4))
                self.episodes.append(segment)
                return [segment]
        return []

    def summary_statistics(self) -> dict[str, JsonValue]:
        if self.frame_probabilities:
            probabilities = np.asarray(self.frame_probabilities, dtype=np.float64)
            mean_probability: float | None = round(float(probabilities.mean()), 4)
            fraction_over_onset: float | None = round(
                float((probabilities >= self.vad_config.onset_probability).mean()), 4
            )
        else:
            mean_probability = None
            fraction_over_onset = None
        return {
            "detector": "silero_streaming",
            "frames": self.frames_seen,
            "episodes": len(self.episodes),
            "mean_prob": mean_probability,
            "frac_frames_over_onset": fraction_over_onset,
        }
