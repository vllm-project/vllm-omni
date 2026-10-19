# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Regression tests for TTS benchmark comparison reports."""

import pytest

from benchmarks.tts.plot_results import print_comparison_table

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize(
    ("candidate", "latency_change", "throughput_change"),
    [
        (20, "-100.0%", "+100.0%"),
        (5, "+50.0%", "-50.0%"),
        (10, "+0.0%", "+0.0%"),
    ],
    ids=["increase", "decrease", "unchanged"],
)
def test_improvement_respects_metric_direction(
    capsys: pytest.CaptureFixture[str],
    candidate: int,
    latency_change: str,
    throughput_change: str,
) -> None:
    all_runs = [
        [
            {
                "_task": "default_voice",
                "_concurrency": 1,
                "mean_audio_ttfp_ms": value,
                "mean_e2el_ms": value,
                "mean_audio_rtf": value,
                "audio_throughput": value,
            }
        ]
        for value in (candidate, 10)
    ]

    print_comparison_table(all_runs, ["candidate", "baseline"])

    improvement = capsys.readouterr().out.split("### Improvement (candidate vs baseline)", 1)[1]
    assert f"| TTFP (ms) | 1 | {latency_change} |" in improvement
    assert f"| E2E (ms) | 1 | {latency_change} |" in improvement
    assert f"| RTF | 1 | {latency_change} |" in improvement
    assert f"| Throughput (a-s/s) | 1 | {throughput_change} |" in improvement
