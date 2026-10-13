# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Hand-computed contracts and an exact rational oracle for continuity.

Run directly with Python and the existing pybase64 dependency, or collect with pytest in the
normal vLLM development environment. Direct execution loads the entire real
helper by path, avoiding the unrelated vllm_omni package bootstrap.
"""

from __future__ import annotations

import base64
import importlib.util
import json
import os
import random
import sys
import unittest
from contextlib import contextmanager
from fractions import Fraction
from pathlib import Path

if __name__ == "__main__":
    helper_path = Path(__file__).resolve().parents[2] / "vllm_omni/benchmarks/audio_continuity.py"
    spec = importlib.util.spec_from_file_location("_audio_continuity_contract", helper_path)
    assert spec is not None and spec.loader is not None
    helper = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = helper
    spec.loader.exec_module(helper)
else:
    import pytest

    from vllm_omni.benchmarks import audio_continuity as production_helper

    helper = production_helper

    pytestmark = [pytest.mark.core_model, pytest.mark.benchmark, pytest.mark.cpu]

CASES = json.loads(Path(__file__).with_name("audio_continuity_cases.json").read_text(encoding="utf-8"))
THRESHOLD_ENV = "VLLM_OMNI_BENCH_AUDIO_CONTINUITY_THRESHOLD_S"


def pcm_event(response_id: str, duration_s: float = 0.1, rate: int = 24000) -> dict:
    return {
        "type": "response.output_audio.delta",
        "response_id": response_id,
        "sample_rate_hz": rate,
        "delta": base64.b64encode(bytes(round(duration_s * rate) * 2)).decode(),
    }


def done_event(response_id: str, status: str = "completed") -> dict:
    return {"type": "response.done", "response": {"id": response_id, "status": status}}


@contextmanager
def temporary_env(values):
    previous = dict(os.environ)
    os.environ.update(values)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(previous)


class TestAudioContinuityCorrectness(unittest.TestCase):
    def setUp(self):
        self.env = temporary_env({THRESHOLD_ENV: "0.1"})
        self.env.__enter__()
        self.addCleanup(self.env.__exit__, None, None, None)

    def test_hand_computed_timelines_and_pcm_formats(self):
        # Each hand-computed timeline x 5 formats. Frame count is per channel; WAV headers
        # and base64 text bytes must never be included in these PCM lengths.
        formats = [(16000, 2, 1), (24000, 2, 1), (48000, 2, 1), (24000, 2, 2), (24000, 4, 1)]
        for case in CASES:
            for rate, width, channels in formats:
                with self.subTest(case=case["id"], rate=rate, width=width, channels=channels):
                    sizes = [round(d * rate) * width * channels for d in case["durations_s"]]
                    stats = helper.compute_continuity_stats(
                        case["arrivals_s"], sizes, rate, width, channels, case["threshold_s"]
                    )
                    sample = helper.continuity_sample(
                        case["arrivals_s"], case["durations_s"], threshold_s=case["threshold_s"]
                    )
                    self.assertIsNotNone(sample)
                    self.assertAlmostEqual(stats.max_single_underrun_s, case["underrun_s"], places=9)
                    self.assertEqual(stats.underrun_event_count, case["events"])
                    self.assertEqual(stats.is_continuous, case["continuous"])
                    self.assertAlmostEqual(sample["audio_max_single_underrun_s"], case["underrun_s"], places=9)
                    self.assertEqual(sample["audio_underrun_event_count"], case["events"])
                    self.assertEqual(sample["audio_continuity_ok"], case["continuous"])
                    self.assertEqual(sample["audio_chunk_count"], len(case["arrivals_s"]))
                    self.assertAlmostEqual(sample["total_underrun_s"], case["total_underrun_s"], places=9)
                    self.assertAlmostEqual(stats.total_underrun_s, case["total_underrun_s"], places=9)
                    duration = sum(case["durations_s"])
                    ratio = case["total_underrun_s"] / (duration + case["total_underrun_s"])
                    frequency = case["events"] * 60 / duration
                    self.assertAlmostEqual(sample["audio_duration_s"], duration)
                    self.assertAlmostEqual(stats.audio_duration_s, duration)
                    self.assertAlmostEqual(sample["audio_underrun_ratio"], ratio)
                    self.assertAlmostEqual(stats.underrun_ratio, ratio)
                    self.assertAlmostEqual(sample["audio_underrun_events_per_minute"], frequency)
                    self.assertAlmostEqual(stats.underrun_events_per_minute, frequency)

    def test_seeded_timelines_against_exact_rational_oracle(self):
        # Binary-exact timestamps isolate arithmetic from decimal boundary
        # rounding. The oracle uses integer ticks and Fraction, not floats.
        rng = random.Random(20261008)
        for trial in range(200):
            ticks = [rng.randrange(10000)]
            for _ in range(rng.randrange(1, 30)):
                ticks.append(ticks[-1] + rng.randrange(32))
            duration_ticks = [rng.randrange(1, 32) for _ in ticks]
            deficits = [
                max(Fraction(0), Fraction(t - ticks[0] - sum(duration_ticks[:i]), 64)) for i, t in enumerate(ticks)
            ]
            # Exact FIFO schedule, with playback stopped until the next arrival.
            end = Fraction(ticks[0], 64)
            gaps = []
            for t, d in zip(ticks, duration_ticks, strict=True):
                arrival = Fraction(t, 64)
                gaps.append(max(Fraction(0), arrival - end))
                end = max(end, arrival) + Fraction(d, 64)
            worst = max(gaps)
            total = sum(gaps)
            events = sum(g > 0 for g in gaps)
            duration = Fraction(sum(duration_ticks), 64)
            self.assertEqual(total, max(deficits))  # former metric is cumulative stall time
            threshold = Fraction(rng.randrange(16), 64)
            with self.subTest(trial=trial):
                sample = helper.continuity_sample(
                    [t / 64 for t in ticks], [d / 64 for d in duration_ticks], threshold_s=float(threshold)
                )
                stats = helper.compute_continuity_stats(
                    [t / 64 for t in ticks],
                    [d * 1024 for d in duration_ticks],
                    32768,
                    threshold_s=float(threshold),
                )
                self.assertEqual(sample["audio_max_single_underrun_s"], float(worst))
                self.assertEqual(stats.max_single_underrun_s, float(worst))
                self.assertEqual(sample["audio_underrun_event_count"], events)
                self.assertEqual(stats.underrun_event_count, events)
                self.assertEqual(sample["audio_underrun_s"], float(total))
                self.assertEqual(stats.max_underrun_s, float(total))
                self.assertEqual(sample["total_underrun_s"], float(total))
                self.assertEqual(stats.total_underrun_s, float(total))
                self.assertEqual(sample["audio_underrun_ratio"], float(total / (duration + total)))
                self.assertEqual(sample["audio_underrun_events_per_minute"], float(events * 60 / duration))
                self.assertEqual(sample["audio_continuity_ok"], total <= threshold)
                self.assertEqual(stats.is_continuous, total <= threshold)

    def test_time_origin_does_not_change_deficit(self):
        for offset in (0, 10, 10000):
            with self.subTest(offset=offset):
                sample = helper.continuity_sample([offset, offset + 0.5, offset + 0.75], [0.125] * 3, threshold_s=0.125)
                self.assertEqual(sample["audio_max_single_underrun_s"], 0.375)
                self.assertEqual(sample["audio_underrun_s"], 0.5)
                self.assertEqual(sample["total_underrun_s"], 0.5)
                self.assertFalse(sample["audio_continuity_ok"])

    def test_invalid_timelines_are_unmeasured(self):
        invalid = [
            ([], [], 0.1),
            ([0], [], 0.1),
            ([0], [0], 0.1),
            ([0], [-0.1], 0.1),
            ([1, 0], [0.1, 0.1], 0.1),
            ([float("nan")], [0.1], 0.1),
            ([float("inf")], [0.1], 0.1),
            ([0], [float("nan")], 0.1),
            ([0], [float("inf")], 0.1),
            ([0], [0.1], -0.1),
            ([0], [0.1], float("nan")),
            ([0], [0.1], float("inf")),
        ]
        for arrivals, durations, threshold in invalid:
            with self.subTest(arrivals=arrivals, durations=durations, threshold=threshold):
                self.assertIsNone(helper.continuity_sample(arrivals, durations, threshold_s=threshold))

    def test_threshold_environment_fallback_and_clamping(self):
        values = [("0.25", 0.25), ("0", 0.0), ("-1", 0.0), ("bad", 0.1), ("nan", 0.1), ("inf", 0.1), ("-inf", 0.1)]
        for value, expected in values:
            with self.subTest(value=value), temporary_env({THRESHOLD_ENV: value}):
                self.assertEqual(helper.audio_continuity_threshold_s(), expected)
        with temporary_env({}):
            os.environ.pop(THRESHOLD_ENV, None)
            self.assertEqual(helper.audio_continuity_threshold_s(), 0.1)

    def test_duplex_reply_isolation_and_end_wait_exclusion(self):
        events = [
            pcm_event("a"),
            pcm_event("a"),
            done_event("a"),
            pcm_event("b"),
            pcm_event("b"),
            done_event("b"),
            pcm_event("b"),
        ]
        rows = helper.collect_duplex_audio_continuity(events, [10, 10.5, 90, 100, 100.05, 1000, 2000])
        self.assertEqual(set(rows), {"a", "b"})
        self.assertAlmostEqual(rows["a"]["audio_underrun_s"], 0.4)
        self.assertFalse(rows["a"]["audio_continuity_ok"])
        self.assertEqual(rows["b"]["audio_underrun_s"], 0.0)
        self.assertTrue(rows["b"]["audio_continuity_ok"])
        self.assertEqual(rows["b"]["audio_chunk_count"], 2)

    def test_duplex_multiple_stalls_propagate_all_fields_per_response(self):
        events = [
            pcm_event("a", 0.1),
            pcm_event("a", 0.2),
            pcm_event("a", 0.2),
            done_event("a"),
            pcm_event("b", 1.0),
            done_event("b"),
        ]
        rows = helper.collect_duplex_audio_continuity(events, [0.3, 0.45, 0.85, 10, 100, 1000])
        self.assertAlmostEqual(rows["a"]["audio_max_single_underrun_s"], 0.2)
        self.assertAlmostEqual(rows["a"]["total_underrun_s"], 0.25)
        self.assertEqual(rows["a"]["audio_underrun_event_count"], 2)
        self.assertAlmostEqual(rows["a"]["audio_underrun_ratio"], 1 / 3)
        self.assertAlmostEqual(rows["a"]["audio_underrun_events_per_minute"], 240)
        for key in (
            "audio_underrun_s",
            "total_underrun_s",
            "audio_underrun_event_count",
            "audio_underrun_ratio",
            "audio_underrun_events_per_minute",
        ):
            self.assertEqual(rows["b"][key], 0)

    def test_duplex_failures_and_missing_audio_are_excluded(self):
        for status in ("cancelled", "failed", "incomplete"):
            with self.subTest(status=status):
                events = [pcm_event("a"), pcm_event("a"), done_event("a", status)]
                self.assertEqual(helper.collect_duplex_audio_continuity(events, [1, 2, 3]), {})
        self.assertEqual(helper.collect_duplex_audio_continuity([done_event("listen")], [10]), {})
        self.assertEqual(helper.collect_duplex_audio_continuity([pcm_event("unfinished")], [10]), {})

    def test_duplex_invalid_packet_rates_and_payloads(self):
        for value in (0, -1, True, 24000.0, "24000"):
            event = pcm_event("a")
            event["sample_rate_hz"] = value
            with self.subTest(rate=value):
                rows = helper.collect_duplex_audio_continuity([pcm_event("a"), event, done_event("a")], [1, 2, 3])
                self.assertEqual(rows, {})
        for value in ("!not-base64!", base64.b64encode(b"x").decode()):
            event = pcm_event("a")
            event["delta"] = value
            with self.subTest(payload=value):
                rows = helper.collect_duplex_audio_continuity([pcm_event("a"), event, done_event("a")], [1, 2, 3])
                self.assertEqual(rows, {})

    def test_server_and_benchmark_agree_at_shifted_threshold_boundary(self):
        # Both callers use the same floating-point tolerance and recurrence.
        sample = helper.continuity_sample([100, 100.2], [0.1, 0.1], threshold_s=0.1)
        stats = helper.compute_continuity_stats([100, 100.2], [4800, 4800], 24000, threshold_s=0.1)
        self.assertTrue(sample["audio_continuity_ok"])
        self.assertAlmostEqual(stats.max_underrun_s, 0.1)
        self.assertEqual(stats.is_continuous, sample["audio_continuity_ok"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
