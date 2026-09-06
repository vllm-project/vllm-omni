# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import numpy as np
import pytest

from examples.offline_inference.dreamzero import benchmark_utils


pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_summarize_benchmark_uses_generate_time_for_model_fps() -> None:
    request_timings = [
        benchmark_utils.RequestTiming(
            index=0,
            label="initial",
            latency_s=2.0,
            action_count=24,
            action_shape=(24, 8),
            video_latent_frames=5,
        ),
        benchmark_utils.RequestTiming(
            index=1,
            label="chunk_0",
            latency_s=1.0,
            action_count=24,
            action_shape=(24, 8),
            video_latent_frames=4,
        ),
        benchmark_utils.RequestTiming(
            index=2,
            label="chunk_1",
            latency_s=3.0,
            action_count=24,
            action_shape=(24, 8),
            video_latent_frames=4,
        ),
    ]

    summary = benchmark_utils.summarize_benchmark(
        request_timings,
        decoded_video_frames=13,
        decode_latency_s=0.5,
        write_latency_s=0.25,
        output_video_path="outputs/dreamzero/benchmark/prediction.mp4",
        actions_path="outputs/dreamzero/benchmark/actions.npz",
    )

    assert summary["requests"]["count"] == 3
    assert summary["latency_s"]["first_request"] == 2.0
    assert summary["latency_s"]["steady_state"]["count"] == 2
    assert summary["latency_s"]["steady_state"]["mean"] == 2.0
    assert summary["latency_s"]["steady_state"]["min"] == 1.0
    assert summary["latency_s"]["steady_state"]["max"] == 3.0
    assert summary["latency_s"]["model_generate_total"] == 6.0
    assert summary["latency_s"]["model_plus_decode_total"] == 6.5
    assert summary["throughput"]["model_request_hz"] == pytest.approx(3 / 6)
    assert summary["throughput"]["model_video_fps"] == pytest.approx(13 / 6)
    assert summary["throughput"]["model_action_hz"] == pytest.approx(72 / 6)
    assert summary["throughput"]["model_plus_decode_video_fps"] == pytest.approx(13 / 6.5)
    assert summary["artifacts"]["output_video"] == "outputs/dreamzero/benchmark/prediction.mp4"
    assert summary["artifacts"]["actions"] == "outputs/dreamzero/benchmark/actions.npz"


def test_validate_action_array_rejects_bad_shape_and_nonfinite_values() -> None:
    good = np.zeros((24, 8), dtype=np.float32)
    result = benchmark_utils.validate_action_array(good, request_index=0)
    assert result == {"shape": [24, 8], "count": 24}

    with pytest.raises(AssertionError, match="Action 1 shape mismatch"):
        benchmark_utils.validate_action_array(np.zeros((24, 7), dtype=np.float32), request_index=1)

    bad = good.copy()
    bad[0, 0] = np.nan
    with pytest.raises(AssertionError, match="Action 2 contains non-finite values"):
        benchmark_utils.validate_action_array(bad, request_index=2)


def test_compare_action_sequences_reports_tolerance_failures() -> None:
    actual = [
        np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        np.asarray([[5.0, 6.0]], dtype=np.float32),
    ]
    reference = [
        np.asarray([[1.0, 2.1], [2.9, 4.0]], dtype=np.float32),
        np.asarray([[5.5, 6.0]], dtype=np.float32),
    ]

    comparison = benchmark_utils.compare_action_sequences(
        actual,
        reference,
        atol=0.11,
    )

    assert comparison["passed"] is False
    assert comparison["atol"] == 0.11
    assert comparison["chunks"][0]["passed"] is True
    assert comparison["chunks"][0]["max_abs_error"] == pytest.approx(0.1)
    assert comparison["chunks"][1]["passed"] is False
    assert comparison["chunks"][1]["max_abs_error"] == pytest.approx(0.5)
