# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Level-2 validation for ACE-Step generated audio.

Checks whether a WAV file produced by ``text_to_music.py`` is *plausibly
music-shaped*: right duration, right sample rate, stereo, non-silent, has
frequency content, has dynamic range, has stereo correlation but not pure
mono dupe.

Does NOT check whether the audio matches the prompt (that requires listening
or perceptual metrics out of scope for a smoke test).

Usage:
    python validate_audio.py ace_step_output.wav --expected-duration 30.0

Exit status: 0 if all checks pass, 1 if any fail.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str


def _check(name: str, passed: bool, detail: str) -> CheckResult:
    return CheckResult(name=name, passed=passed, detail=detail)


def load_audio(path: str) -> tuple[np.ndarray, int]:
    """Load a WAV file as float64 in [-1, 1]. Returns (audio[T, C], sample_rate)."""
    try:
        import soundfile as sf

        audio, sample_rate = sf.read(path, always_2d=True)
        return np.asarray(audio, dtype=np.float64), int(sample_rate)
    except ImportError:
        import scipy.io.wavfile as wav

        sample_rate, audio = wav.read(path)
        if audio.ndim == 1:
            audio = audio[:, None]
        if np.issubdtype(audio.dtype, np.integer):
            max_val = float(np.iinfo(audio.dtype).max)
            audio = audio.astype(np.float64) / max_val
        else:
            audio = audio.astype(np.float64)
        return audio, int(sample_rate)


def check_duration(audio: np.ndarray, sr: int, expected_s: float, tol_s: float) -> CheckResult:
    actual_s = audio.shape[0] / sr
    passed = abs(actual_s - expected_s) <= tol_s
    return _check(
        "duration_matches_request",
        passed,
        f"actual={actual_s:.2f}s, expected={expected_s:.2f}s, tol=±{tol_s}s",
    )


def check_sample_rate(sr: int, expected: int) -> CheckResult:
    return _check(
        "sample_rate_is_48000",
        sr == expected,
        f"actual={sr} Hz, expected={expected} Hz",
    )


def check_stereo(audio: np.ndarray) -> CheckResult:
    return _check(
        "is_stereo",
        audio.shape[1] == 2,
        f"channels={audio.shape[1]}",
    )


def check_not_silent(audio: np.ndarray, threshold: float = 0.01) -> CheckResult:
    peak = float(np.max(np.abs(audio)))
    return _check(
        "not_silent",
        peak >= threshold,
        f"peak={peak:.4f} (need >= {threshold})",
    )


def check_has_frequency_content(audio: np.ndarray, sr: int) -> CheckResult:
    """The FFT should have energy above 100 Hz (not just DC offset)."""
    mono = audio.mean(axis=1)
    spectrum = np.abs(np.fft.rfft(mono))
    freqs = np.fft.rfftfreq(len(mono), d=1.0 / sr)

    above_100hz = spectrum[freqs >= 100.0]
    below_100hz = spectrum[freqs < 100.0]
    if len(above_100hz) == 0 or below_100hz.sum() == 0:
        return _check("has_frequency_content_above_100hz", False, "no FFT bins above 100 Hz")
    ratio = float(above_100hz.sum() / (below_100hz.sum() + 1e-12))
    return _check(
        "has_frequency_content_above_100hz",
        ratio >= 0.1,
        f"ratio(above100Hz/below100Hz)={ratio:.3f} (need >= 0.1)",
    )


def check_has_dynamic_range(audio: np.ndarray) -> CheckResult:
    """Audio should not be near-DC and not 100% clipped to one value."""
    std = float(np.std(audio))
    return _check(
        "has_dynamic_range",
        std >= 0.01,
        f"std={std:.4f} (need >= 0.01)",
    )


def check_not_clipped(audio: np.ndarray, clip_fraction_threshold: float = 0.05) -> CheckResult:
    """No more than 5% of samples should be at ±1.0 (full-scale clipping)."""
    clipped = np.sum(np.abs(audio) >= 0.999) / audio.size
    return _check(
        "not_severely_clipped",
        clipped <= clip_fraction_threshold,
        f"clipped_fraction={clipped:.4f} (need <= {clip_fraction_threshold})",
    )


def check_stereo_correlation(audio: np.ndarray) -> CheckResult:
    """Two channels should be related (real stereo) but not identical (mono dupe).

    Acceptable range: correlation in [0.1, 0.999].
    Below 0.1: channels look uncorrelated (bug in stereo handling).
    At ~1.0: probably duplicated mono (real audio is never exactly identical L/R).
    """
    if audio.shape[1] < 2:
        return _check("stereo_correlation_sane", True, "(mono, skipped)")
    left = audio[:, 0]
    right = audio[:, 1]
    if np.std(left) < 1e-8 or np.std(right) < 1e-8:
        return _check("stereo_correlation_sane", False, "one channel is silent")
    corr = float(np.corrcoef(left, right)[0, 1])
    passed = 0.1 <= corr <= 0.999
    return _check(
        "stereo_correlation_sane",
        passed,
        f"corr(L, R)={corr:.4f} (need 0.1 <= corr <= 0.999)",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate ACE-Step generated audio.")
    parser.add_argument("wav_path", help="Path to the WAV file to validate.")
    parser.add_argument(
        "--expected-duration",
        type=float,
        default=30.0,
        help="Expected audio duration in seconds (default: 30.0).",
    )
    parser.add_argument(
        "--duration-tolerance",
        type=float,
        default=1.0,
        help="Allowed deviation from expected duration in seconds.",
    )
    parser.add_argument(
        "--expected-sample-rate",
        type=int,
        default=48000,
        help="Expected sample rate (default: 48000).",
    )
    args = parser.parse_args()

    print(f"Loading {args.wav_path}...")
    audio, sr = load_audio(args.wav_path)
    print(f"Shape: {audio.shape} (samples, channels), sample_rate={sr} Hz")
    print()

    checks: list[Callable[[], CheckResult]] = [
        lambda: check_duration(audio, sr, args.expected_duration, args.duration_tolerance),
        lambda: check_sample_rate(sr, args.expected_sample_rate),
        lambda: check_stereo(audio),
        lambda: check_not_silent(audio),
        lambda: check_has_frequency_content(audio, sr),
        lambda: check_has_dynamic_range(audio),
        lambda: check_not_clipped(audio),
        lambda: check_stereo_correlation(audio),
    ]

    results = [c() for c in checks]
    passed = sum(r.passed for r in results)
    failed = len(results) - passed

    print("Level-2 audio-shape checks:")
    for r in results:
        marker = "PASS" if r.passed else "FAIL"
        print(f"  [{marker}]  {r.name:<40}  {r.detail}")
    print()
    print(f"Summary: {passed}/{len(results)} checks passed.")

    if failed == 0:
        print("Verdict: Audio is plausibly music-shaped.")
        return 0
    else:
        print("Verdict: Audio failed structural checks. Listen to confirm and debug.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
