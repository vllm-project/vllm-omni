# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""
Unit tests for metrics.py
"""

import math

import pytest
from vllm.benchmarks.serve import TaskType

from vllm_omni.benchmarks.metrics.metrics import calculate_metrics
from vllm_omni.benchmarks.patch.patch import MixRequestFuncOutput

pytestmark = [pytest.mark.core_model, pytest.mark.benchmark, pytest.mark.cpu]


def test_tpot_matches_mean_itl_per_request():
    """TPOT and pooled ITL should agree when output_len tracks ITL samples."""
    output = MixRequestFuncOutput()
    output.success = True
    output.prompt_len = 100
    # Simulate server reporting more tokens than SSE chunks (bundled tokens).
    # len(itl)=2 → 2 inter-chunk intervals, but server generated 10 tokens.
    output.output_tokens = 10
    output.generated_text = "hello world"
    output.ttft = 0.05
    output.text_latency = 0.25
    output.latency = 0.30
    output.itl = [0.10, 0.10]

    metrics, _ = calculate_metrics(
        input_requests=[],
        outputs=[output],
        dur_s=1.0,
        tokenizer=None,
        selected_percentiles=[50.0],
        goodput_config_dict={},
        task_type=TaskType.GENERATION,
        selected_percentile_metrics=["tpot", "itl"],
        max_concurrency=None,
        request_rate=float("inf"),
        benchmark_duration=1.0,
    )

    assert metrics.mean_tpot_ms == pytest.approx(metrics.mean_itl_ms, rel=1e-6, abs=1e-6)


def _make_output(prompt_len: int, output_tokens: int = 10) -> MixRequestFuncOutput:
    """Build a minimal successful MixRequestFuncOutput for metrics aggregation."""
    output = MixRequestFuncOutput()
    output.success = True
    output.prompt_len = prompt_len
    output.output_tokens = output_tokens
    output.generated_text = "x" * output_tokens
    output.ttft = 0.1
    output.text_latency = 1.0
    output.latency = 1.0
    output.start_time = 0.0
    output.itl = [0.1] * max(output_tokens - 1, 0)
    output.audio_ttfp = 0.0
    output.audio_rtf = 0.0
    output.audio_duration = 0.0
    output.audio_frames = 0
    output.input_audio_duration = 0.0
    output.error = ""
    return output


def _calculate_test_metrics(outputs, goodput=None):
    return calculate_metrics(
        input_requests=[],
        outputs=outputs,
        dur_s=1.0,
        tokenizer=None,
        selected_percentiles=[50.0],
        goodput_config_dict=goodput or {},
        task_type=TaskType.GENERATION,
        selected_percentile_metrics=[],
        max_concurrency=None,
        request_rate=float("inf"),
        benchmark_duration=1.0,
    )[0]


# ============================================================================
# total_input Tests
# ============================================================================


def test_total_input_aggregated_from_output_prompt_len():
    """Test that total_input sums outputs[i].prompt_len, not input_requests[i].prompt_len."""
    outputs = [_make_output(4992), _make_output(3000)]

    metrics, _ = calculate_metrics(
        input_requests=[],
        outputs=outputs,
        dur_s=10.0,
        tokenizer=None,
        selected_percentiles=[99.0],
        goodput_config_dict={},
        task_type=TaskType.GENERATION,
        selected_percentile_metrics=[],
        max_concurrency=None,
        request_rate=float("inf"),
        benchmark_duration=10.0,
    )

    assert metrics.total_input == 7992, (
        "total_input should aggregate from outputs[i].prompt_len to reflect the true multimodal input token count"
    )


def test_audio_continuity_aggregation():
    """Continuity rate and underrun percentile must aggregate from per-output fields."""
    bad = _make_output(100)
    bad.audio_underrun_s = 0.5
    bad.audio_continuity_ok = False
    good_a = _make_output(100)
    good_a.audio_underrun_s = 0.02
    good_a.audio_continuity_ok = True
    good_b = _make_output(100)
    good_b.audio_underrun_s = 0.0
    good_b.audio_continuity_ok = True

    metrics, _ = calculate_metrics(
        input_requests=[],
        outputs=[bad, good_a, good_b],
        dur_s=10.0,
        tokenizer=None,
        selected_percentiles=[50.0, 99.0],
        goodput_config_dict={},
        task_type=TaskType.GENERATION,
        selected_percentile_metrics=["audio_underrun"],
        max_concurrency=None,
        request_rate=float("inf"),
        benchmark_duration=10.0,
    )

    assert metrics.audio_continuity_ok_rate == pytest.approx(2 / 3, abs=1e-6)
    # p99 of [0.5, 0.02, 0.0] is dominated by the 0.5 outlier.
    p99 = dict(metrics.percentiles_audio_underrun_s).get(99.0)
    assert p99 is not None and p99 > 0.4


def test_unmeasured_duplex_latency_does_not_add_zero_samples():
    measured = _make_output(100, output_tokens=2)
    measured.ttft = 0.1
    measured.audio_ttfp = 0.2
    measured.audio_rtf = 0.5
    measured.duplex_session_metrics = {
        "mean_ttft_ms": 100.0,
        "mean_ttfp_ms": 200.0,
        "mean_rtf": 0.5,
    }
    listen_only = _make_output(100, output_tokens=0)
    listen_only.ttft = listen_only.audio_ttfp = listen_only.audio_rtf = 0.0
    listen_only.duplex_session_metrics = {
        "mean_ttft_ms": None,
        "mean_ttfp_ms": None,
        "mean_rtf": None,
    }

    metrics = _calculate_test_metrics([measured, listen_only], {"ttft": 150.0})

    assert metrics.mean_ttft_ms == 100.0
    assert metrics.mean_audio_ttfp_ms == 200.0
    assert metrics.mean_audio_rtf == 0.5
    assert (metrics.num_ttft_samples, metrics.num_audio_ttfp_samples, metrics.num_audio_rtf_samples) == (1, 1, 1)
    assert metrics.request_goodput == 1.0


def test_all_unmeasured_duplex_latency_is_not_reported_as_zero():
    listen_only = _make_output(100, output_tokens=0)
    listen_only.duplex_session_metrics = {
        "mean_ttft_ms": None,
        "mean_ttfp_ms": None,
        "mean_rtf": None,
    }

    metrics = _calculate_test_metrics(
        [listen_only],
        {"ttft": float("inf"), "audio_ttft": float("inf")},
    )

    assert (metrics.num_ttft_samples, metrics.num_audio_ttfp_samples, metrics.num_audio_rtf_samples) == (0, 0, 0)
    assert math.isnan(metrics.mean_ttft_ms)
    assert math.isnan(metrics.mean_audio_ttfp_ms)
    assert math.isnan(metrics.mean_audio_rtf)
    assert metrics.request_goodput == 0.0


def test_unmeasured_duplex_tpot_does_not_add_zero_or_misalign_goodput():
    missing_tpot = _make_output(100, output_tokens=5)
    missing_tpot.itl = []
    missing_tpot.text_latency = missing_tpot.ttft = 0.1
    missing_tpot.tpot_measured = False
    missing_tpot.duplex_session_metrics = {"mean_ttft_ms": 100.0}

    slow_ttft = _make_output(100, output_tokens=5)
    slow_ttft.ttft = 1.0
    slow_ttft.text_latency = 1.4
    slow_ttft.itl = [0.1] * 4
    slow_ttft.duplex_session_metrics = {"mean_ttft_ms": 1000.0}

    metrics = _calculate_test_metrics(
        [missing_tpot, slow_ttft],
        {"ttft": 500.0, "tpot": 200.0},
    )

    assert metrics.num_tpot_samples == 1
    assert metrics.mean_tpot_ms == pytest.approx(100.0)
    assert metrics.request_goodput == 0.0


def test_all_unmeasured_duplex_token_timing_is_not_reported_as_zero():
    output = _make_output(100, output_tokens=5)
    output.itl = []
    output.text_latency = output.ttft
    output.tpot_measured = False
    output.duplex_session_metrics = {"mean_ttft_ms": 100.0}

    metrics = _calculate_test_metrics([output], {"tpot": float("inf")})

    assert (metrics.num_tpot_samples, metrics.num_itl_samples) == (0, 0)
    assert math.isnan(metrics.mean_tpot_ms)
    assert math.isnan(metrics.mean_itl_ms)
    assert metrics.request_goodput == 0.0


def test_duplex_response_timings_do_not_build_a_session_token_timeline():
    output = _make_output(100, output_tokens=5)
    output.latency = 101.0
    output.duplex_request_metrics = [
        {"response_id": "r1", "stage0_tokens": {"itls_ms": [100.0, 100.0]}},
        {"response_id": "r2", "stage0_tokens": {"itls_ms": [100.0, 100.0]}},
    ]
    output.duplex_session_metrics = {"mean_ttft_ms": 100.0}

    metrics = _calculate_test_metrics([output])

    assert math.isnan(metrics.max_output_tokens_per_s)
    assert metrics.max_concurrent_requests == 1
    assert metrics.mean_tpot_ms == pytest.approx(100.0)
    assert metrics.mean_itl_ms == pytest.approx(100.0)


def test_unmeasured_tpot_stays_missing_after_tokenizer_fallback():
    output = _make_output(100, output_tokens=0)
    output.generated_text = "timing metadata missing"
    output.itl = []
    output.text_latency = output.ttft
    output.tpot_measured = False
    output.duplex_session_metrics = {"mean_ttft_ms": 100.0}

    def tokenizer(text, *, add_special_tokens):
        assert text == output.generated_text
        assert add_special_tokens is False
        return type("Tokenized", (), {"input_ids": [1, 2, 3]})()

    metrics, _ = calculate_metrics(
        input_requests=[],
        outputs=[output],
        dur_s=1.0,
        tokenizer=tokenizer,
        selected_percentiles=[50.0],
        goodput_config_dict={"tpot": float("inf")},
        task_type=TaskType.GENERATION,
        selected_percentile_metrics=[],
        max_concurrency=None,
        request_rate=float("inf"),
        benchmark_duration=1.0,
    )

    assert metrics.total_output == 3
    assert metrics.num_tpot_samples == 0
    assert math.isnan(metrics.mean_tpot_ms)
    assert metrics.request_goodput == 0.0


def test_measured_zero_itl_is_not_treated_as_missing():
    output = _make_output(100, output_tokens=3)
    output.itl = [0.0, 0.0]
    output.duplex_session_metrics = {"mean_ttft_ms": 100.0}

    metrics = _calculate_test_metrics([output], {"tpot": 1.0})

    assert (metrics.num_tpot_samples, metrics.num_itl_samples) == (1, 2)
    assert metrics.mean_tpot_ms == 0.0
    assert metrics.mean_itl_ms == 0.0
    assert metrics.request_goodput == 1.0


def test_duplex_goodput_does_not_pair_measurements_from_different_requests():
    text_only, audio_only = _make_output(100), _make_output(100)
    text_only.ttft, text_only.audio_ttfp = 0.1, 0.0
    text_only.duplex_session_metrics = {"mean_ttft_ms": 100.0, "mean_ttfp_ms": None, "mean_rtf": None}
    audio_only.ttft, audio_only.audio_ttfp = 0.0, 0.2
    audio_only.duplex_session_metrics = {"mean_ttft_ms": None, "mean_ttfp_ms": 200.0, "mean_rtf": None}

    metrics = _calculate_test_metrics([text_only, audio_only], {"ttft": 500.0, "audio_ttft": 500.0})

    assert metrics.request_goodput == 0.0


# ============================================================================
# TTFT suppression for pure-audio (TTS) benchmarks
# ============================================================================


class _EmptyAwareTokenizer:
    """Minimal tokenizer stub: token count == len(text), so '' -> 0 tokens.

    Mirrors production where a TTS speech endpoint returns empty generated_text,
    making total_output == 0 (the real CI path uses a real tokenizer, not None).
    """

    def __call__(self, text, add_special_tokens=False):
        class _R:
            pass

        r = _R()
        r.input_ids = [0] * len(text)
        return r


def _make_tts_output(prompt_len: int) -> MixRequestFuncOutput:
    """Pure-TTS output: no text tokens, only audio. ttft is left unset (0.0)."""
    output = MixRequestFuncOutput()
    output.success = True
    output.prompt_len = prompt_len
    output.output_tokens = 0
    output.generated_text = ""
    output.ttft = 0.0
    output.text_latency = 1.0
    output.latency = 1.0
    output.start_time = 0.0
    output.itl = []
    output.audio_ttfp = 0.05
    output.audio_rtf = 0.2
    output.audio_duration = 5.0
    output.audio_frames = 120000
    output.input_audio_duration = 0.0
    output.error = ""
    return output


_TTS_PERCENTILE_METRICS = ["ttft", "e2el", "audio_rtf", "audio_ttfp", "audio_duration"]


def test_tts_benchmark_omits_ttft(capsys):
    """Pure-TTS run (total_output == 0) must not print a Time to First Token section."""
    outputs = [_make_tts_output(100), _make_tts_output(120)]

    calculate_metrics(
        input_requests=[],
        outputs=outputs,
        dur_s=10.0,
        tokenizer=_EmptyAwareTokenizer(),
        selected_percentiles=[99.0],
        goodput_config_dict={},
        task_type=TaskType.GENERATION,
        selected_percentile_metrics=_TTS_PERCENTILE_METRICS,
        max_concurrency=None,
        request_rate=float("inf"),
        benchmark_duration=10.0,
    )

    out = capsys.readouterr().out
    assert "Time to First Token" not in out, "TTS bench must not surface a meaningless TTFT"
    assert "Time to First Packet" in out, "audio TTFP must still be reported"
    assert "End-to-end Latency" in out, "e2el must still be reported"


def test_text_benchmark_still_reports_ttft(capsys):
    """Regression guard: real text generation (total_output > 0) keeps TTFT."""
    outputs = [_make_output(100, output_tokens=10), _make_output(120, output_tokens=10)]

    calculate_metrics(
        input_requests=[],
        outputs=outputs,
        dur_s=10.0,
        tokenizer=_EmptyAwareTokenizer(),
        selected_percentiles=[99.0],
        goodput_config_dict={},
        task_type=TaskType.GENERATION,
        selected_percentile_metrics=_TTS_PERCENTILE_METRICS,
        max_concurrency=None,
        request_rate=float("inf"),
        benchmark_duration=10.0,
    )

    out = capsys.readouterr().out
    assert "Time to First Token" in out, "text benchmarks must keep TTFT"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
