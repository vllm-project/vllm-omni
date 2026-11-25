# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Portions derived from https://github.com/step-out/Duplex-MPE
# (Copyright (c) 2026 MPeval authors, MIT License; see benchmarks/duplex_mpe/LICENSE).
"""Model-independent Duplex-MPE reactive playback protocol and timing windows."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Generator
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Literal, Protocol

import numpy as np
import soundfile
from numpy.typing import NDArray
from pydantic import JsonValue

from benchmarks.duplex_mpe.dataset import TurnEventType
from benchmarks.duplex_mpe.vad import (
    SileroActivityConfig,
    SileroActivityDetector,
    SileroVadModel,
)

PROTOCOL_ID = "mpeval.unified.v3"
TARGET_ONSET_DEADLINE_SECONDS = 5.0
TARGET_MAX_RESPONSE_SECONDS = 60.0
N4_PAUSE_SECONDS = 3.0
ORDINARY_GAP_SECONDS = 0.5
CLOCK_DRIFT_TOLERANCE_SAMPLES = 1.01
RESPONSE_CAP_EPSILON_SECONDS = 1e-6

TargetOutcome = Literal[
    "T_HIT_5S",
    "T_MISS_5S",
    "T_OVERLONG_60S",
    "T_HIT_CONTINUATION",
]

TARGET_HIT_OUTCOME: TargetOutcome = "T_HIT_5S"
TARGET_MISS_OUTCOME: TargetOutcome = "T_MISS_5S"
TARGET_OVERLONG_OUTCOME: TargetOutcome = "T_OVERLONG_60S"
TARGET_CONTINUATION_OUTCOME: TargetOutcome = "T_HIT_CONTINUATION"


class AudioFeedAdapter(Protocol):
    @property
    def input_sample_rate_hz(self) -> int: ...

    @property
    def output_sample_rate_hz(self) -> int: ...

    def feed(self, input_samples: NDArray[np.float32]) -> NDArray[np.float32]: ...


class StopCondition(Protocol):
    def __call__(self) -> bool: ...


class ScenarioProgressCallback(Protocol):
    def __call__(
        self,
        turn_index: int,
        event_type: TurnEventType,
        target_outcome: TargetOutcome | None,
    ) -> None: ...


@dataclass(frozen=True, kw_only=True)
class ScenarioTurnInput:
    turn_index: int
    event_type: TurnEventType
    speaker: str
    audio_path: Path | None = None
    waveform_samples: NDArray[np.float32] | None = None


@dataclass(frozen=True, kw_only=True)
class TurnTimingWindow:
    turn_index: int
    event_type: TurnEventType
    speaker: str
    turn_start_seconds: float
    turn_end_seconds: float
    window_start_seconds: float
    window_end_seconds: float


@dataclass(kw_only=True)
class ScenarioScheduleResult:
    protocol_id: str
    windows: list[TurnTimingWindow] = field(default_factory=list)
    assistant_episodes: list[tuple[float, float]] = field(default_factory=list)
    target_outcome: TargetOutcome | None = None
    target_end_seconds: float | None = None
    target_onset_seconds: float | None = None
    target_onset_latency_seconds: float | None = None
    target_endpoint_seconds: float | None = None
    target_response_duration_seconds: float | None = None
    target_forced_floor_return: bool = False
    is_speaking_at_target_end: bool = False
    total_duration_seconds: float = 0.0
    aed_config: dict[str, JsonValue] = field(default_factory=dict)
    assistant_waveform: NDArray[np.float32] | None = None

    def to_report_dict(self) -> dict[str, JsonValue]:
        return {
            "protocol_id": self.protocol_id,
            "aed_config": self.aed_config,
            "total_duration_s": round(self.total_duration_seconds, 4),
            "assistant_speech_segments": [
                [segment_start, segment_end] for segment_start, segment_end in self.assistant_episodes
            ],
            "T": {
                "outcome": self.target_outcome,
                "t_end_s": self.target_end_seconds,
                "onset_s": self.target_onset_seconds,
                "onset_latency_s": self.target_onset_latency_seconds,
                "endpoint_s": self.target_endpoint_seconds,
                "response_duration_s": self.target_response_duration_seconds,
                "forced_floor_return": self.target_forced_floor_return,
                "speaking_at_t_end": self.is_speaking_at_target_end,
            },
            "rows": [
                {
                    "turn_index": window.turn_index,
                    "event_type": window.event_type,
                    "speaker": window.speaker,
                    "turn_start": round(window.turn_start_seconds, 4),
                    "turn_end": round(window.turn_end_seconds, 4),
                    "win_start": round(window.window_start_seconds, 4),
                    "win_end": round(window.window_end_seconds, 4),
                }
                for window in self.windows
            ],
        }


class ClockedAudioAdapter:
    """Enforce continuous sample-preserving assistant output on the input audio clock."""

    def __init__(self, inner_adapter: AudioFeedAdapter) -> None:
        self.inner_adapter: AudioFeedAdapter = inner_adapter
        self.input_sample_rate_hz: int = int(inner_adapter.input_sample_rate_hz)
        self.output_sample_rate_hz: int = int(inner_adapter.output_sample_rate_hz)
        if self.input_sample_rate_hz <= 0 or self.output_sample_rate_hz <= 0:
            raise ValueError("Adapter sample rates must be positive")
        self.total_input_samples: int = 0
        self.total_output_samples: int = 0

    def feed(self, input_samples: NDArray[np.float32]) -> NDArray[np.float32]:
        raw_output = self.inner_adapter.feed(input_samples)
        output_samples = np.array(raw_output, dtype=np.float32, copy=True)
        if output_samples.ndim != 1 or not np.isfinite(output_samples).all():
            raise ValueError("Adapter output must be finite mono float32 audio")
        self.total_input_samples += len(input_samples)
        self.total_output_samples += len(output_samples)
        expected_output_samples = self.total_input_samples * self.output_sample_rate_hz / self.input_sample_rate_hz
        if abs(self.total_output_samples - expected_output_samples) > CLOCK_DRIFT_TOLERANCE_SAMPLES:
            raise ValueError(
                "Model adapter lost audio clock alignment; return silence when idle and buffer excess samples"
            )
        return output_samples


@lru_cache(maxsize=1)
def compute_protocol_fingerprint() -> str:
    digest = hashlib.sha256()
    module_directory = Path(__file__).resolve().parent
    for python_file in sorted(module_directory.glob("*.py")):
        digest.update(python_file.name.encode("utf-8"))
        digest.update(python_file.read_bytes())
    digest.update((module_directory / "seeds.json").read_bytes())
    return digest.hexdigest()[:12]


@lru_cache(maxsize=1)
def load_reference_seeds() -> dict[str, int]:
    seeds_path = Path(__file__).with_name("seeds.json")
    raw_seeds = json.loads(seeds_path.read_text(encoding="utf-8"))
    if not isinstance(raw_seeds, dict):
        raise ValueError("seeds.json must be a JSON object")
    validated_seeds: dict[str, int] = {}
    for scenario_id, seed_value in raw_seeds.items():
        if not isinstance(scenario_id, str) or type(seed_value) is not int or not 0 <= seed_value < 2**32:
            raise ValueError(f"Invalid seed entry for {scenario_id!r}")
        validated_seeds[scenario_id] = seed_value
    return validated_seeds


def resolve_scenario_seed(scenario_id: str, seed_mapping: dict[str, int] | None = None) -> int:
    active_mapping = load_reference_seeds() if seed_mapping is None else seed_mapping
    if scenario_id not in active_mapping:
        raise ValueError(f"No reference seed for {scenario_id}; provide a custom seed mapping")
    seed_value = active_mapping[scenario_id]
    if type(seed_value) is not int or not 0 <= seed_value < 2**32:
        raise ValueError("Scenario seed must be an integer in [0, 2**32)")
    return seed_value


def resample_linear_mono(
    waveform_samples: NDArray[np.float32],
    *,
    source_sample_rate_hz: int,
    target_sample_rate_hz: int,
) -> NDArray[np.float32]:
    if source_sample_rate_hz == target_sample_rate_hz or len(waveform_samples) == 0:
        return waveform_samples.astype(np.float32, copy=False)
    target_length = int(round(len(waveform_samples) * target_sample_rate_hz / source_sample_rate_hz))
    if target_length <= 0:
        return np.zeros(0, dtype=np.float32)
    return np.interp(
        np.linspace(0, len(waveform_samples) - 1, target_length),
        np.arange(len(waveform_samples)),
        waveform_samples,
    ).astype(np.float32)


def load_mono_waveform_at_rate(audio_path: Path, *, target_sample_rate_hz: int) -> NDArray[np.float32]:
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio clip missing: {audio_path}")
    waveform, file_sample_rate_hz = soundfile.read(str(audio_path), dtype="float32")
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    return resample_linear_mono(
        np.asarray(waveform, dtype=np.float32).reshape(-1),
        source_sample_rate_hz=int(file_sample_rate_hz),
        target_sample_rate_hz=target_sample_rate_hz,
    )


def iter_scenario_schedule(
    turns: list[ScenarioTurnInput],
    *,
    input_sample_rate_hz: int,
    output_sample_rate_hz: int,
    vad_config: SileroActivityConfig | None = None,
    vad_model: SileroVadModel | None = None,
    chunk_milliseconds: float = 80.0,
    ordinary_gap_seconds: float = ORDINARY_GAP_SECONDS,
    n4_pause_seconds: float = N4_PAUSE_SECONDS,
    onset_deadline_seconds: float = TARGET_ONSET_DEADLINE_SECONDS,
    maximum_response_seconds: float = TARGET_MAX_RESPONSE_SECONDS,
    target_endpoint_hangover_milliseconds: float | None = None,
    preamble_samples: NDArray[np.float32] | None = None,
    progress_callback: ScenarioProgressCallback | None = None,
    collect_audio: bool = False,
) -> Generator[NDArray[np.float32], NDArray[np.float32], ScenarioScheduleResult]:
    """Yield input audio chunks and consume aligned output chunks under mpeval.unified.v3."""
    if input_sample_rate_hz <= 0 or output_sample_rate_hz <= 0:
        raise ValueError("input_sample_rate_hz and output_sample_rate_hz must be positive")
    if not math.isfinite(chunk_milliseconds) or chunk_milliseconds <= 0.0:
        raise ValueError("chunk_milliseconds must be finite and positive")
    for param_name, param_val in (
        ("ordinary_gap_seconds", ordinary_gap_seconds),
        ("n4_pause_seconds", n4_pause_seconds),
        ("onset_deadline_seconds", onset_deadline_seconds),
        ("maximum_response_seconds", maximum_response_seconds),
    ):
        if not math.isfinite(param_val) or param_val <= 0.0:
            raise ValueError(f"{param_name} must be finite and positive")
    resolved_vad_config = vad_config if vad_config is not None else SileroActivityConfig()
    detector = SileroActivityDetector(
        vad_config=resolved_vad_config,
        sample_rate_hz=output_sample_rate_hz,
        vad_model=vad_model,
    )
    endpoint_hangover_milliseconds = (
        float(target_endpoint_hangover_milliseconds)
        if target_endpoint_hangover_milliseconds is not None
        else float(resolved_vad_config.target_endpoint_hangover_milliseconds)
    )

    schedule_result = ScenarioScheduleResult(
        protocol_id=PROTOCOL_ID,
        aed_config=resolved_vad_config.to_report_dict(),
    )
    schedule_result.aed_config["code_fingerprint"] = compute_protocol_fingerprint()
    schedule_result.aed_config["t_endpoint_hangover_ms"] = endpoint_hangover_milliseconds
    schedule_result.aed_config["input_sr"] = input_sample_rate_hz
    schedule_result.aed_config["assistant_sr"] = output_sample_rate_hz
    if vad_model is not None:
        schedule_result.aed_config["custom_vad_injected"] = True

    chunk_samples = max(1, int(round(input_sample_rate_hz * chunk_milliseconds / 1000.0)))
    captured_chunks: list[NDArray[np.float32]] = []

    def record_output(output_samples: NDArray[np.float32]) -> None:
        if len(output_samples) > 0:
            detector.push(output_samples)
            if collect_audio:
                captured_chunks.append(np.asarray(output_samples, dtype=np.float32).reshape(-1))

    def stream_waveform(
        waveform_samples: NDArray[np.float32],
    ) -> Generator[NDArray[np.float32], NDArray[np.float32], None]:
        for sample_offset in range(0, len(waveform_samples), chunk_samples):
            output_chunk = yield waveform_samples[sample_offset : sample_offset + chunk_samples]
            record_output(output_chunk)

    def stream_silence_until(
        stop_condition: StopCondition,
        limit_seconds: float,
    ) -> Generator[NDArray[np.float32], NDArray[np.float32], float]:
        total_silence_samples = int(round(limit_seconds * input_sample_rate_hz))
        fed_samples = 0
        while fed_samples < total_silence_samples:
            step_samples = min(chunk_samples, total_silence_samples - fed_samples)
            output_chunk = yield np.zeros(max(0, step_samples), dtype=np.float32)
            record_output(output_chunk)
            fed_samples += step_samples
            if stop_condition():
                break
        return fed_samples / input_sample_rate_hz

    def never_stop() -> bool:
        return False

    if preamble_samples is not None and len(preamble_samples) > 0:
        yield from stream_waveform(np.asarray(preamble_samples, dtype=np.float32).reshape(-1))
        yield from stream_silence_until(never_stop, ordinary_gap_seconds)

    for turn in turns:
        turn_waveform: NDArray[np.float32]
        if turn.waveform_samples is not None:
            turn_waveform = np.asarray(turn.waveform_samples, dtype=np.float32).reshape(-1)
        elif turn.audio_path is not None:
            turn_waveform = load_mono_waveform_at_rate(
                turn.audio_path,
                target_sample_rate_hz=input_sample_rate_hz,
            )
        else:
            raise ValueError(f"Turn {turn.turn_index} provides neither waveform_samples nor audio_path")

        turn_start_seconds = detector.now_seconds
        yield from stream_waveform(turn_waveform)
        turn_end_seconds = detector.now_seconds

        if turn.event_type != "T":
            gap_seconds = n4_pause_seconds if turn.event_type == "N4_Q" else ordinary_gap_seconds
            yield from stream_silence_until(never_stop, gap_seconds)
            schedule_result.windows.append(
                TurnTimingWindow(
                    turn_index=turn.turn_index,
                    event_type=turn.event_type,
                    speaker=turn.speaker,
                    turn_start_seconds=turn_start_seconds,
                    turn_end_seconds=turn_end_seconds,
                    window_start_seconds=turn_start_seconds,
                    window_end_seconds=detector.now_seconds,
                )
            )
            if progress_callback is not None:
                progress_callback(turn.turn_index, turn.event_type, None)
        else:
            schedule_result.target_end_seconds = round(turn_end_seconds, 4)
            was_speaking_at_target_end = detector.is_voicing_now
            voiced_runs_at_target_end = len(detector.voiced_runs)
            schedule_result.is_speaking_at_target_end = bool(was_speaking_at_target_end)
            schedule_result.aed_config["t_end_state"] = {
                "voicing_now": bool(was_speaking_at_target_end),
                "episode_open": bool(detector.is_speaking),
                "disagree_due_to_hangover": bool(detector.is_speaking and (not was_speaking_at_target_end)),
            }

            def is_endpoint_confirmed() -> bool:
                return detector.is_endpoint_confirmed(endpoint_hangover_milliseconds)

            def is_fresh_onset_seen() -> bool:
                return (
                    detector.voicing_onset_at_or_after(
                        turn_end_seconds,
                        since_run_index=voiced_runs_at_target_end,
                    )
                    is not None
                )

            if was_speaking_at_target_end:
                fed_seconds = yield from stream_silence_until(is_endpoint_confirmed, maximum_response_seconds)
                hit_response_cap = fed_seconds >= maximum_response_seconds - RESPONSE_CAP_EPSILON_SECONDS
                if hit_response_cap and (not is_endpoint_confirmed()):
                    schedule_result.target_outcome = TARGET_OVERLONG_OUTCOME
                    schedule_result.target_forced_floor_return = True
                    schedule_result.target_response_duration_seconds = round(maximum_response_seconds, 4)
                else:
                    schedule_result.target_outcome = TARGET_CONTINUATION_OUTCOME
                    voiced_end_seconds = detector.last_voiced_end_seconds()
                    if voiced_end_seconds is None or voiced_end_seconds < turn_end_seconds:
                        voiced_end_seconds = detector.episodes[-1][1] if detector.episodes else detector.now_seconds
                    schedule_result.target_endpoint_seconds = round(float(voiced_end_seconds), 4)
                    schedule_result.target_response_duration_seconds = round(
                        float(voiced_end_seconds) - turn_end_seconds,
                        4,
                    )
            else:
                yield from stream_silence_until(is_fresh_onset_seen, onset_deadline_seconds)
                if not is_fresh_onset_seen():
                    schedule_result.target_outcome = TARGET_MISS_OUTCOME
                else:
                    detected_onset = detector.voicing_onset_at_or_after(
                        turn_end_seconds,
                        since_run_index=voiced_runs_at_target_end,
                    )
                    onset_seconds = float(detected_onset) if detected_onset is not None else turn_end_seconds
                    schedule_result.target_onset_seconds = round(onset_seconds, 4)
                    schedule_result.target_onset_latency_seconds = round(onset_seconds - turn_end_seconds, 4)
                    remaining_budget_seconds = max(
                        0.0,
                        onset_seconds + maximum_response_seconds - detector.now_seconds,
                    )
                    yield from stream_silence_until(is_endpoint_confirmed, remaining_budget_seconds)
                    if is_endpoint_confirmed():
                        endpoint_seconds = detector.last_voiced_end_seconds()
                        if endpoint_seconds is None or endpoint_seconds < onset_seconds:
                            endpoint_seconds = detector.episodes[-1][1] if detector.episodes else detector.now_seconds
                        schedule_result.target_outcome = TARGET_HIT_OUTCOME
                        schedule_result.target_endpoint_seconds = round(float(endpoint_seconds), 4)
                        schedule_result.target_response_duration_seconds = round(
                            float(endpoint_seconds) - onset_seconds,
                            4,
                        )
                    else:
                        schedule_result.target_outcome = TARGET_OVERLONG_OUTCOME
                        schedule_result.target_forced_floor_return = True
                        schedule_result.target_response_duration_seconds = round(maximum_response_seconds, 4)

            schedule_result.windows.append(
                TurnTimingWindow(
                    turn_index=turn.turn_index,
                    event_type=turn.event_type,
                    speaker=turn.speaker,
                    turn_start_seconds=turn_start_seconds,
                    turn_end_seconds=turn_end_seconds,
                    window_start_seconds=turn_start_seconds,
                    window_end_seconds=detector.now_seconds,
                )
            )
            if progress_callback is not None:
                progress_callback(turn.turn_index, turn.event_type, schedule_result.target_outcome)

    detector.finalize()
    if collect_audio:
        schedule_result.assistant_waveform = (
            np.concatenate(captured_chunks) if captured_chunks else np.zeros(0, dtype=np.float32)
        )
    schedule_result.assistant_episodes = list(detector.episodes)
    schedule_result.total_duration_seconds = detector.now_seconds
    return schedule_result


def run_scenario_schedule(
    adapter: AudioFeedAdapter,
    turns: list[ScenarioTurnInput],
    *,
    vad_config: SileroActivityConfig | None = None,
    vad_model: SileroVadModel | None = None,
    chunk_milliseconds: float = 80.0,
    ordinary_gap_seconds: float = ORDINARY_GAP_SECONDS,
    n4_pause_seconds: float = N4_PAUSE_SECONDS,
    onset_deadline_seconds: float = TARGET_ONSET_DEADLINE_SECONDS,
    maximum_response_seconds: float = TARGET_MAX_RESPONSE_SECONDS,
    target_endpoint_hangover_milliseconds: float | None = None,
    preamble_samples: NDArray[np.float32] | None = None,
    progress_callback: ScenarioProgressCallback | None = None,
    collect_audio: bool = False,
) -> ScenarioScheduleResult:
    """Execute the mpeval.unified.v3 schedule synchronously against an AudioFeedAdapter."""
    schedule = iter_scenario_schedule(
        turns,
        input_sample_rate_hz=int(adapter.input_sample_rate_hz),
        output_sample_rate_hz=int(adapter.output_sample_rate_hz),
        vad_config=vad_config,
        vad_model=vad_model,
        chunk_milliseconds=chunk_milliseconds,
        ordinary_gap_seconds=ordinary_gap_seconds,
        n4_pause_seconds=n4_pause_seconds,
        onset_deadline_seconds=onset_deadline_seconds,
        maximum_response_seconds=maximum_response_seconds,
        target_endpoint_hangover_milliseconds=target_endpoint_hangover_milliseconds,
        preamble_samples=preamble_samples,
        progress_callback=progress_callback,
        collect_audio=collect_audio,
    )
    try:
        input_chunk = next(schedule)
        while True:
            output_chunk = adapter.feed(input_chunk)
            input_chunk = schedule.send(output_chunk)
    except StopIteration as completed:
        return completed.value
