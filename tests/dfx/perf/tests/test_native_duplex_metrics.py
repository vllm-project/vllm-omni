# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import copy

import pytest

from tests.dfx.perf.native_metrics import native_measurement, native_performance_gate

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _result():
    observations = [
        {
            "request_id": "duplex-s.a.i.0.e.0.r.stage0",
            "replica_id": 0,
            "omni_append_receipt_count": n,
            "num_prompt_tokens": 81 + n * 13,
            "num_computed_tokens": 81,
            "omni_context_tokens": 81 + n * 13,
            "omni_context_limit": 40960,
        }
        for n in (0, 1)
    ]
    return {
        "ok": True,
        "session_count": 1,
        "identity_isolation_ok": True,
        "native_model_turn_end_ok": True,
        "model_turn_end_count": 1,
        "sessions": [
            {
                "error_count": 0,
                "done_count": 1,
                "audio_delta_count": 3,
                "native_append_observations": observations,
                "request_metrics": [{"ttft_ms": 10.0, "ttfp_ms": 30.0, "rtf": 0.1, "audio_duration_ms": 320}],
            }
        ],
    }


def test_accepts_v028_receipts_without_legacy_flags():
    result = _result()
    before = copy.deepcopy(result)
    metrics = native_measurement(result, sessions=1, turns=1, elapsed_s=2.0)
    assert metrics["responses_per_s"] == 0.5
    assert metrics["max_context_tokens"] == 94
    assert result == before
    assert metrics["audio_seconds_per_s"] == 0.16
    assert not metrics["stage0_token_metrics_available"]


def test_native_stage0_metrics_are_separate_from_audio_synchronous_text():
    result = _result()
    result["sessions"][0]["response_timings"] = {
        "response": {
            "stage0_tokens": {
                "source": "engine_stage_metrics",
                "output_token_count": 3,
                "ttft_ms": 2.0,
                "itls_ms": [1.0, 3.0],
            }
        }
    }
    metrics = native_measurement(result, sessions=1, turns=1, elapsed_s=2.0)
    assert metrics["stage0_mean_ttft_ms"] == 2.0
    assert metrics["mean_ttft_ms"] == 10.0
    assert metrics["stage0_mean_itl_ms"] == 2.0


def test_native_performance_gate_requires_same_hardware_and_excludes_instrumentation():
    aggregate = {"hardware": "H200", "median_responses_per_s": 1, "median_ttfp_ms": 100}
    baseline = {"H200": {"median_responses_per_s": 1, "median_ttfp_ms": 100, "regression_tolerance": 0.1}}
    assert native_performance_gate(aggregate, baseline) == "passed"
    assert native_performance_gate({**aggregate, "hardware": "H100"}, baseline) == "unbaselined_hardware"
    with pytest.raises(AssertionError):
        native_performance_gate({**aggregate, "median_responses_per_s": 0.5}, baseline)
    with pytest.raises(AssertionError):
        native_performance_gate({**aggregate, "median_ttfp_ms": 150}, baseline)
    assert native_performance_gate({**aggregate, "profiling_run_not_performance_baseline": True}, baseline) == (
        "not_checked_instrumented"
    )


@pytest.mark.parametrize("bad_case", ["fallback", "no_commit", "no_audio", "no_eos", "context_overflow", "no_replica"])
def test_rejects_non_native_or_incomplete_samples(bad_case):
    result = _result()
    session = result["sessions"][0]
    if bad_case == "fallback":
        session["native_append_observations"] = []
    elif bad_case == "no_commit":
        session["native_append_observations"][-1]["omni_append_receipt_count"] = 0
    elif bad_case == "no_audio":
        session["audio_delta_count"] = 0
    elif bad_case == "no_eos":
        result["native_model_turn_end_ok"] = False
    elif bad_case == "no_replica":
        for observation in session["native_append_observations"]:
            observation.pop("replica_id")
    else:
        session["native_append_observations"][-1]["omni_context_tokens"] = 40961
    with pytest.raises(AssertionError):
        native_measurement(result, sessions=1, turns=1, elapsed_s=2.0)
