# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Audio preparation and output processing for OmniVoice."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torchaudio

_END_PUNCTUATION = {
    ";",
    ":",
    ",",
    ".",
    "!",
    "?",
    "…",
    ")",
    "]",
    "}",
    '"',
    "'",
    "“",
    "”",
    "‘",
    "’",
    "；",
    "：",
    "，",
    "。",
    "！",
    "？",
    "、",
    "……",
    "）",
    "】",
}


@dataclass(frozen=True)
class PreparedReferenceAudio:
    """Reference waveform shared by ASR and the audio tokenizer."""

    waveform: np.ndarray
    sample_rate: int
    original_rms: float


def _segment_length_ms(num_samples: int, sample_rate: int) -> int:
    """Return the rounded millisecond length used by the reference algorithm."""
    return round(1000 * num_samples / sample_rate)


def _frame_index(milliseconds: int, sample_rate: int, num_samples: int) -> int:
    """Map a millisecond position to a bounded sample index."""
    return min(num_samples, max(0, int(milliseconds * sample_rate / 1000.0)))


def _slice_milliseconds(audio: np.ndarray, start_ms: int, end_ms: int | None, sample_rate: int) -> np.ndarray:
    num_samples = audio.shape[-1]
    start = _frame_index(start_ms, sample_rate, num_samples)
    end = num_samples if end_ms is None else _frame_index(end_ms, sample_rate, num_samples)
    return audio[:, start:end]


def _rms(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    squared = np.square(np.asarray(audio, dtype=np.float32), dtype=np.float32)
    return float(np.sqrt(np.mean(squared, dtype=np.float32)))


def _dbfs(audio: np.ndarray) -> float:
    rms = _rms(audio)
    if rms == 0.0:
        return -float("inf")
    return 20.0 * float(np.log10(rms))


def _window_rms(audio: np.ndarray, starts_ms: list[int], window_ms: int, sample_rate: int) -> np.ndarray:
    starts = np.fromiter(
        (_frame_index(start, sample_rate, audio.shape[-1]) for start in starts_ms),
        dtype=np.int64,
    )
    ends = np.fromiter(
        (_frame_index(start + window_ms, sample_rate, audio.shape[-1]) for start in starts_ms),
        dtype=np.int64,
    )
    squared = np.square(audio, dtype=np.float32)
    cumulative = np.concatenate(
        [np.zeros((audio.shape[0], 1), dtype=np.float64), np.cumsum(squared, axis=-1, dtype=np.float64)],
        axis=-1,
    )
    sums = cumulative[:, ends] - cumulative[:, starts]
    counts = ends - starts
    rms = np.zeros(len(starts_ms), dtype=np.float32)
    valid = counts > 0
    if np.any(valid):
        means = sums[:, valid].sum(axis=0) / (counts[valid] * audio.shape[0])
        rms[valid] = np.sqrt(means).astype(np.float32)
    return rms


def _detect_silence_ranges(
    audio: np.ndarray,
    sample_rate: int,
    *,
    min_silence_ms: int,
    silence_threshold_db: float,
    seek_step_ms: int,
) -> list[list[int]]:
    """Find silent ranges with the same window and range rules as pydub."""
    segment_length_ms = _segment_length_ms(audio.shape[-1], sample_rate)
    if segment_length_ms < min_silence_ms:
        return []

    threshold = 10.0 ** (silence_threshold_db / 20.0)
    last_start = segment_length_ms - min_silence_ms
    silence_starts = list(range(0, last_start + 1, seek_step_ms))
    if last_start % seek_step_ms:
        silence_starts.append(last_start)

    window_rms = _window_rms(audio, silence_starts, min_silence_ms, sample_rate)
    silent_starts = [start for start, rms in zip(silence_starts, window_rms) if rms <= threshold]
    if not silent_starts:
        return []

    silent_ranges: list[list[int]] = []
    previous = silent_starts[0]
    current_start = previous
    for start in silent_starts[1:]:
        continuous = start == previous + seek_step_ms
        has_gap = start > previous + min_silence_ms
        if not continuous and has_gap:
            silent_ranges.append([current_start, previous + min_silence_ms])
            current_start = start
        previous = start
    silent_ranges.append([current_start, previous + min_silence_ms])
    return silent_ranges


def _detect_nonsilent_ranges(
    audio: np.ndarray,
    sample_rate: int,
    *,
    min_silence_ms: int,
    silence_threshold_db: float,
    seek_step_ms: int,
) -> list[list[int]]:
    segment_length_ms = _segment_length_ms(audio.shape[-1], sample_rate)
    silent_ranges = _detect_silence_ranges(
        audio,
        sample_rate,
        min_silence_ms=min_silence_ms,
        silence_threshold_db=silence_threshold_db,
        seek_step_ms=seek_step_ms,
    )
    if not silent_ranges:
        return [[0, segment_length_ms]]
    if silent_ranges[0] == [0, segment_length_ms]:
        return []

    nonsilent_ranges: list[list[int]] = []
    previous_end = 0
    for start, end in silent_ranges:
        nonsilent_ranges.append([previous_end, start])
        previous_end = end
    if previous_end != segment_length_ms:
        nonsilent_ranges.append([previous_end, segment_length_ms])
    if nonsilent_ranges and nonsilent_ranges[0] == [0, 0]:
        nonsilent_ranges.pop(0)
    return nonsilent_ranges


def _split_on_silence(
    audio: np.ndarray,
    sample_rate: int,
    *,
    min_silence_ms: int,
    silence_threshold_db: float,
    keep_silence_ms: int,
    seek_step_ms: int,
) -> np.ndarray:
    ranges = _detect_nonsilent_ranges(
        audio,
        sample_rate,
        min_silence_ms=min_silence_ms,
        silence_threshold_db=silence_threshold_db,
        seek_step_ms=seek_step_ms,
    )
    output_ranges = [[start - keep_silence_ms, end + keep_silence_ms] for start, end in ranges]
    for previous, current in zip(output_ranges, output_ranges[1:]):
        if current[0] < previous[1]:
            split = (previous[1] + current[0]) // 2
            previous[1] = split
            current[0] = split

    segment_length_ms = _segment_length_ms(audio.shape[-1], sample_rate)
    segments = [
        _slice_milliseconds(audio, max(start, 0), min(end, segment_length_ms), sample_rate)
        for start, end in output_ranges
    ]
    if not segments:
        return np.empty((audio.shape[0], 0), dtype=np.float32)
    return np.concatenate(segments, axis=-1)


def _detect_leading_silence_ms(audio: np.ndarray, sample_rate: int, *, threshold_db: float, chunk_ms: int = 10) -> int:
    segment_length_ms = _segment_length_ms(audio.shape[-1], sample_rate)
    trim_ms = 0
    while trim_ms < segment_length_ms:
        chunk = _slice_milliseconds(audio, trim_ms, trim_ms + chunk_ms, sample_rate)
        if _dbfs(chunk) >= threshold_db:
            break
        trim_ms += chunk_ms
    return min(trim_ms, segment_length_ms)


def remove_silence(
    audio: np.ndarray,
    sample_rate: int,
    *,
    middle_silence_ms: int,
    leading_silence_ms: int,
    trailing_silence_ms: int,
) -> np.ndarray:
    """Remove long gaps and trim edge silence using official thresholds."""
    wave = np.asarray(audio, dtype=np.float32)
    if wave.ndim == 1:
        wave = wave[np.newaxis, :]
    if wave.ndim != 2:
        raise ValueError(f"OmniVoice audio must be 1D or 2D, got {wave.ndim} dimensions.")
    if middle_silence_ms > 0:
        wave = _split_on_silence(
            wave,
            sample_rate,
            min_silence_ms=middle_silence_ms,
            silence_threshold_db=-50,
            keep_silence_ms=middle_silence_ms,
            seek_step_ms=10,
        )

    start_idx = _detect_leading_silence_ms(wave, sample_rate, threshold_db=-50)
    start_idx = max(0, start_idx - leading_silence_ms)
    wave = _slice_milliseconds(wave, start_idx, None, sample_rate)

    wave = wave[:, ::-1]
    start_idx = _detect_leading_silence_ms(wave, sample_rate, threshold_db=-50)
    start_idx = max(0, start_idx - trailing_silence_ms)
    wave = _slice_milliseconds(wave, start_idx, None, sample_rate)
    wave = wave[:, ::-1]

    return np.ascontiguousarray(wave, dtype=np.float32)


def trim_long_audio(
    audio: np.ndarray,
    sample_rate: int,
    *,
    max_duration_s: float = 15.0,
    min_duration_s: float = 3.0,
    trim_threshold_s: float = 20.0,
) -> np.ndarray:
    """Trim long reference audio at a suitable silence boundary."""
    if audio.shape[-1] / sample_rate <= trim_threshold_s:
        return audio

    non_silent_ranges = _detect_nonsilent_ranges(
        np.asarray(audio, dtype=np.float32),
        sample_rate=sample_rate,
        min_silence_ms=100,
        silence_threshold_db=-40,
        seek_step_ms=10,
    )
    if not non_silent_ranges:
        return audio

    max_ms = int(max_duration_s * 1000)
    min_ms = int(min_duration_s * 1000)
    best_split = 0
    for start, end in non_silent_ranges:
        if start > best_split and start <= max_ms:
            best_split = start
        if end > max_ms:
            break

    if best_split < min_ms:
        best_split = min(max_ms, _segment_length_ms(audio.shape[-1], sample_rate))
    return _slice_milliseconds(np.asarray(audio, dtype=np.float32), 0, best_split, sample_rate)


def prepare_reference_audio(
    waveform: np.ndarray | torch.Tensor,
    sample_rate: int,
    *,
    target_sample_rate: int,
    hop_length: int,
    trim_long: bool,
) -> PreparedReferenceAudio:
    """Prepare reference audio before ASR and audio-tokenizer encoding."""
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.detach().cpu().numpy()
    waveform = np.array(waveform, dtype=np.float32, copy=True)
    if waveform.ndim == 1:
        waveform = waveform[np.newaxis, :]
    elif waveform.ndim != 2:
        raise ValueError(f"OmniVoice reference audio must be 1D or 2D, got {waveform.ndim} dimensions.")
    if not waveform.size or not np.any(waveform):
        raise ValueError("Reference audio is empty after silence removal.")
    if waveform.shape[0] > 1:
        waveform = waveform.mean(axis=0, keepdims=True)

    if sample_rate != target_sample_rate:
        waveform = torchaudio.functional.resample(
            torch.from_numpy(waveform),
            orig_freq=sample_rate,
            new_freq=target_sample_rate,
        ).numpy()

    original_rms = float(np.sqrt(np.mean(waveform**2))) if waveform.size else 0.0
    if 0 < original_rms < 0.1:
        waveform = waveform * (0.1 / original_rms)
        waveform = np.clip(waveform, -1.0, 1.0)

    if trim_long:
        waveform = trim_long_audio(waveform, target_sample_rate)
    waveform = remove_silence(
        waveform,
        target_sample_rate,
        middle_silence_ms=200,
        leading_silence_ms=100,
        trailing_silence_ms=200,
    )
    if waveform.shape[-1] == 0:
        raise ValueError("Reference audio is empty after silence removal.")

    remainder = waveform.shape[-1] % hop_length
    if remainder:
        waveform = waveform[:, :-remainder]
    if waveform.shape[-1] == 0:
        raise ValueError("Reference audio is shorter than one audio-tokenizer hop after preprocessing.")

    return PreparedReferenceAudio(
        waveform=np.ascontiguousarray(waveform, dtype=np.float32),
        sample_rate=target_sample_rate,
        original_rms=original_rms,
    )


def add_reference_punctuation(text: str) -> str:
    """Add an English or Chinese sentence terminator when one is missing."""
    text = text.strip()
    if not text:
        return text
    if text[-1] in _END_PUNCTUATION:
        return text
    if any("\u4e00" <= char <= "\u9fff" for char in text):
        return f"{text}。"
    return f"{text}."


def postprocess_generated_audio(
    audio: np.ndarray,
    *,
    sample_rate: int,
    reference_rms: float,
) -> np.ndarray:
    """Apply official OmniVoice output silence, volume, fade, and padding rules."""
    audio = np.array(audio, dtype=np.float32, copy=True)
    if audio.ndim == 1:
        audio = audio[np.newaxis, :]
    if audio.ndim != 2:
        raise ValueError(f"OmniVoice generated audio must be 1D or 2D, got {audio.ndim} dimensions.")

    audio = remove_silence(
        audio,
        sample_rate,
        middle_silence_ms=500,
        leading_silence_ms=100,
        trailing_silence_ms=100,
    )

    if audio.shape[-1] == 0:
        return np.ascontiguousarray(audio, dtype=np.float32)

    if reference_rms < 0.1:
        audio = audio * (reference_rms / 0.1)

    fade_samples = int(0.1 * sample_rate)
    if fade_samples > 0:
        fade_length = min(fade_samples, audio.shape[-1] // 2)
        if fade_length > 0:
            fade_in = np.linspace(0, 1, fade_length, dtype=np.float32)[np.newaxis, :]
            fade_out = np.linspace(1, 0, fade_length, dtype=np.float32)[np.newaxis, :]
            audio[:, :fade_length] *= fade_in
            audio[:, -fade_length:] *= fade_out

    pad_samples = int(0.1 * sample_rate)
    if pad_samples > 0:
        padding = np.zeros((audio.shape[0], pad_samples), dtype=np.float32)
        audio = np.concatenate([padding, audio, padding], axis=-1)
    return np.ascontiguousarray(audio, dtype=np.float32)
