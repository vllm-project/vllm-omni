# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""
Unit tests for metrics.py
"""

import math
from types import SimpleNamespace

import pytest
from vllm.benchmarks.serve import TaskType

from vllm_omni.benchmarks.metrics.metrics import (
    _is_audio_speech_without_text_output,
    _is_zero_text_result,
    aggregate_stage_durations,
    calculate_metrics,
    print_stage_durations_metrics,
)
from vllm_omni.benchmarks.patch.patch import MixRequestFuncOutput
from vllm_omni.metrics.definitions import stage_modality_flags

pytestmark = [pytest.mark.core_model, pytest.mark.benchmark, pytest.mark.cpu]


def test_stage_modality_flags_match_print_stage():
    text = stage_modality_flags("text", "")
    audio = stage_modality_flags("", "audio")
    stream = stage_modality_flags("latent", "stream")
    video_frames = stage_modality_flags("video", "image")

    assert text.is_text_stage and not text.is_audio_stage
    assert audio.is_audio_stage and not audio.is_internal_stream_stage
    assert stream.is_internal_stream_stage and not stream.is_text_stage
    assert video_frames.is_video_stage and not video_frames.is_image_stage
    assert not stage_modality_flags(None, None).is_text_stage


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
        "ttft_ms": 100.0,
        "ttfp_ms": 200.0,
        "rtf": 0.5,
    }
    listen_only = _make_output(100, output_tokens=0)
    listen_only.ttft = listen_only.audio_ttfp = listen_only.audio_rtf = 0.0
    listen_only.duplex_session_metrics = {
        "ttft_ms": None,
        "ttfp_ms": None,
        "rtf": None,
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
        "ttft_ms": None,
        "ttfp_ms": None,
        "rtf": None,
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
    missing_tpot.duplex_session_metrics = {"ttft_ms": 100.0}

    slow_ttft = _make_output(100, output_tokens=5)
    slow_ttft.ttft = 1.0
    slow_ttft.text_latency = 1.4
    slow_ttft.itl = [0.1] * 4
    slow_ttft.duplex_session_metrics = {"ttft_ms": 1000.0}

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
    output.duplex_session_metrics = {"ttft_ms": 100.0}

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
    output.duplex_session_metrics = {"ttft_ms": 100.0}

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
    output.duplex_session_metrics = {"ttft_ms": 100.0}

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


def test_zero_itl_does_not_create_zero_tpot():
    output = _make_output(100, output_tokens=3)
    output.itl = [0.0, 0.0]
    output.duplex_session_metrics = {"ttft_ms": 100.0}

    metrics = _calculate_test_metrics([output], {"tpot": 1.0})

    assert (metrics.num_tpot_samples, metrics.num_itl_samples) == (0, 2)
    assert math.isnan(metrics.mean_tpot_ms)
    assert metrics.mean_itl_ms == 0.0
    assert metrics.request_goodput == 0.0


def test_stage_output_tokens_without_client_timing_omit_tpot(capsys):
    output = _make_output(100, output_tokens=5)
    output.itl = []
    output.text_latency = 0.0
    output.ttft = 0.1

    metrics, _ = calculate_metrics(
        input_requests=[],
        outputs=[output],
        dur_s=1.0,
        tokenizer=None,
        selected_percentiles=[50.0, 99.0],
        goodput_config_dict={},
        task_type=TaskType.GENERATION,
        selected_percentile_metrics=["tpot"],
        max_concurrency=None,
        request_rate=float("inf"),
        benchmark_duration=1.0,
    )

    printed = capsys.readouterr().out
    assert metrics.num_tpot_samples == 0
    assert math.isnan(metrics.mean_tpot_ms)
    assert "Time per Output Token" not in printed
    assert "Mean TPOT" not in printed


def test_consistent_client_latency_can_supply_tpot_fallback():
    output = _make_output(100, output_tokens=10)
    output.itl = []
    output.ttft = 0.1
    output.text_latency = 1.0

    metrics = _calculate_test_metrics([output])

    assert metrics.num_tpot_samples == 1
    assert metrics.mean_tpot_ms == pytest.approx(100.0)


def test_single_token_responses_do_not_report_zero_tpot(capsys):
    output = _make_output(100, output_tokens=1)

    metrics, _ = calculate_metrics(
        input_requests=[],
        outputs=[output],
        dur_s=1.0,
        tokenizer=None,
        selected_percentiles=[50.0],
        goodput_config_dict={},
        task_type=TaskType.GENERATION,
        selected_percentile_metrics=["tpot"],
        max_concurrency=None,
        request_rate=float("inf"),
        benchmark_duration=1.0,
    )

    printed = capsys.readouterr().out
    assert metrics.num_tpot_samples == 0
    assert math.isnan(metrics.mean_tpot_ms)
    assert "Time per Output Token" not in printed


def test_duplex_goodput_does_not_pair_measurements_from_different_requests():
    text_only, audio_only = _make_output(100), _make_output(100)
    text_only.ttft, text_only.audio_ttfp = 0.1, 0.0
    text_only.duplex_session_metrics = {"ttft_ms": 100.0, "ttfp_ms": None, "rtf": None}
    audio_only.ttft, audio_only.audio_ttfp = 0.0, 0.2
    audio_only.duplex_session_metrics = {"ttft_ms": None, "ttfp_ms": 200.0, "rtf": None}

    metrics = _calculate_test_metrics([text_only, audio_only], {"ttft": 500.0, "audio_ttft": 500.0})

    assert metrics.request_goodput == 0.0


# ============================================================================
# openai-audio-speech text-output gating
# ============================================================================


class _MetricsWithTotalOutput:
    def __init__(self, total_output: int) -> None:
        self.total_output = total_output


@pytest.mark.parametrize(
    ("backend", "total_output", "expected"),
    [
        ("openai-audio-speech", 0, True),
        ("openai-audio-speech", 1, False),
        (None, 0, False),
        ("openai-chat", 0, False),
        ("openai-audio-speech", 10, False),
    ],
)
def test_is_audio_speech_without_text_output(backend, total_output, expected):
    assert _is_audio_speech_without_text_output(backend, _MetricsWithTotalOutput(total_output)) is expected


# ============================================================================
# TTFT suppression for pure-audio (TTS) benchmarks
# ============================================================================


class _EmptyAwareTokenizer:
    """Minimal tokenizer stub: token count == len(text), so '' -> 0 tokens.

    Mirrors production where a TTS speech endpoint returns empty generated_text,
    making total_output == 0 (the real CI path uses a real tokenizer, not None).
    """

    def __call__(self, text, add_special_tokens=False):
        return SimpleNamespace(input_ids=[0] * len(text))


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


def _run_tts_calculate_metrics(outputs, *, tokenizer, backend=None):
    return calculate_metrics(
        input_requests=[],
        outputs=outputs,
        dur_s=10.0,
        tokenizer=tokenizer,
        selected_percentiles=[99.0],
        goodput_config_dict={},
        task_type=TaskType.GENERATION,
        selected_percentile_metrics=_TTS_PERCENTILE_METRICS,
        max_concurrency=None,
        request_rate=float("inf"),
        benchmark_duration=10.0,
        backend=backend,
    )


def test_zero_text_result_is_decided_before_tokenizer_fallback():
    speech = _make_tts_output(100)
    assert _is_zero_text_result(speech, "openai-audio-speech") is True
    assert _is_zero_text_result(speech, None) is True

    text = _make_output(100, output_tokens=0)
    text.generated_text = "hello"
    text.audio_duration = 0.0
    text.audio_frames = 0
    assert _is_zero_text_result(text, None) is False


def test_tokenizer_none_speech_keeps_zero_text_metrics(capsys):
    """tokenizer=None must not invent output_len=1 for successful TTS."""
    outputs = [_make_tts_output(100), _make_tts_output(120)]

    metrics, actual_output_lens = _run_tts_calculate_metrics(
        outputs,
        tokenizer=None,
        backend="openai-audio-speech",
    )

    assert actual_output_lens == [0, 0]
    assert metrics.total_output == 0
    out = capsys.readouterr().out
    assert "Total generated tokens:" not in out
    assert "Output token throughput" not in out
    assert "Time to First Token" not in out
    assert "Time to First Packet" in out


def test_tokenizer_none_text_fallback_still_uses_placeholder():
    """skip-tokenizer text runs with generated_text still get output_len=1."""
    output = _make_output(100, output_tokens=0)
    output.generated_text = "hello"
    output.audio_duration = 0.0
    output.audio_frames = 0
    output.itl = []

    metrics, actual_output_lens = calculate_metrics(
        input_requests=[],
        outputs=[output],
        dur_s=1.0,
        tokenizer=None,
        selected_percentiles=[50.0],
        goodput_config_dict={},
        task_type=TaskType.GENERATION,
        selected_percentile_metrics=["ttft"],
        max_concurrency=None,
        request_rate=float("inf"),
        benchmark_duration=1.0,
    )

    assert actual_output_lens == [1]
    assert metrics.total_output == 1


def test_audio_speech_backend_omits_generated_token_lines(capsys):
    """Speech backend with zero text tokens must skip generated-token / TTFT lines."""
    calculate_metrics(
        input_requests=[],
        outputs=[_make_tts_output(100)],
        dur_s=10.0,
        tokenizer=_EmptyAwareTokenizer(),
        selected_percentiles=[99.0],
        goodput_config_dict={},
        task_type=TaskType.GENERATION,
        selected_percentile_metrics=_TTS_PERCENTILE_METRICS,
        max_concurrency=None,
        request_rate=float("inf"),
        benchmark_duration=10.0,
        backend="openai-audio-speech",
    )

    out = capsys.readouterr().out
    assert "Total input tokens:" in out
    assert "Total generated tokens:" not in out
    assert "Output token throughput" not in out
    assert "Time to First Token" not in out
    assert "Time to First Packet" in out


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


# ============================================================================
# Suppress text metrics for pure image / video endpoints
# ============================================================================


def _make_image_output(prompt_len: int) -> MixRequestFuncOutput:
    output = MixRequestFuncOutput()
    output.success = True
    output.prompt_len = prompt_len
    output.output_tokens = 0
    output.generated_text = ""
    output.ttft = 0.0
    output.text_latency = 0.0
    output.latency = 2.5
    output.start_time = 0.0
    output.itl = []
    output.image_count = 1
    output.image_generation_time_ms = 2000.0
    output.image_pixels = 1024 * 1024
    output.input_audio_duration = 0.0
    output.error = ""
    return output


def _make_video_output(prompt_len: int) -> MixRequestFuncOutput:
    output = MixRequestFuncOutput()
    output.success = True
    output.prompt_len = prompt_len
    output.output_tokens = 0
    output.generated_text = ""
    output.ttft = 0.0
    output.text_latency = 0.0
    output.latency = 5.0
    output.start_time = 0.0
    output.itl = []
    output.video_duration = 2.0
    output.video_frames = 48
    output.video_generation_time_ms = 4000.0
    output.video_rtf = 2.0
    output.input_audio_duration = 0.0
    output.error = ""
    return output


_MEDIA_PERCENTILE_METRICS = ["e2el", "ttft"]


def test_pure_image_benchmark_omits_text_result(capsys):
    """Pure image generation must not print Text Result or invent peak tok/s."""
    outputs = [_make_image_output(64), _make_image_output(64)]

    metrics, actual_output_lens = calculate_metrics(
        input_requests=[],
        outputs=outputs,
        dur_s=5.0,
        tokenizer=None,
        selected_percentiles=[99.0],
        goodput_config_dict={},
        task_type=TaskType.GENERATION,
        selected_percentile_metrics=_MEDIA_PERCENTILE_METRICS,
        max_concurrency=None,
        request_rate=float("inf"),
        benchmark_duration=5.0,
    )

    out = capsys.readouterr().out
    assert actual_output_lens == [0, 0]
    assert metrics.total_output == 0
    assert metrics.max_output_tokens_per_s == 0.0
    assert " Text Result " not in out
    assert "Peak output token throughput" not in out
    assert "Time to First Token" not in out
    assert " Image Result " in out


def test_pure_video_benchmark_omits_text_result(capsys):
    """Pure video generation must not print Text Result or invent peak tok/s."""
    outputs = [_make_video_output(63), _make_video_output(63)]

    metrics, actual_output_lens = calculate_metrics(
        input_requests=[],
        outputs=outputs,
        dur_s=10.0,
        tokenizer=None,
        selected_percentiles=[99.0],
        goodput_config_dict={},
        task_type=TaskType.GENERATION,
        selected_percentile_metrics=_MEDIA_PERCENTILE_METRICS,
        max_concurrency=None,
        request_rate=float("inf"),
        benchmark_duration=10.0,
    )

    out = capsys.readouterr().out
    assert actual_output_lens == [0, 0]
    assert metrics.total_output == 0
    assert metrics.max_output_tokens_per_s == 0.0
    assert " Text Result " not in out
    assert "Peak output token throughput" not in out
    assert "Time to First Token" not in out
    assert " Video Result " in out


def test_image_with_generated_text_still_reports_text_result(capsys):
    """Image edits / AR text alongside images must keep Text Result."""
    output = _make_image_output(64)
    output.output_tokens = 8
    output.generated_text = "revised caption"
    output.ttft = 0.05
    output.itl = [0.01] * 7

    calculate_metrics(
        input_requests=[],
        outputs=[output],
        dur_s=3.0,
        tokenizer=_EmptyAwareTokenizer(),
        selected_percentiles=[99.0],
        goodput_config_dict={},
        task_type=TaskType.GENERATION,
        selected_percentile_metrics=_MEDIA_PERCENTILE_METRICS,
        max_concurrency=None,
        request_rate=float("inf"),
        benchmark_duration=3.0,
    )

    out = capsys.readouterr().out
    assert " Text Result " in out
    assert "Time to First Token" in out
    assert " Image Result " in out


def test_aggregate_stage_durations_mean_p50_p99() -> None:
    ok_a = MixRequestFuncOutput()
    ok_a.success = True
    ok_a.stage_durations = {"diffuse": 1.0, "vae.decode": 0.2, "stage_0_gen_ms": 50.0}
    ok_b = MixRequestFuncOutput()
    ok_b.success = True
    ok_b.stage_durations = {"diffuse": 3.0, "vae.decode": 0.4, "stage_1_gen_ms": 80.0}
    failed = MixRequestFuncOutput()
    failed.success = False
    failed.stage_durations = {"diffuse": 99.0}

    summaries = aggregate_stage_durations([ok_a, ok_b, failed])
    assert summaries["stage_durations_mean"]["diffuse"] == pytest.approx(2.0)
    assert summaries["stage_durations_p50"]["diffuse"] == pytest.approx(2.0)
    assert summaries["stage_durations_mean"]["vae.decode"] == pytest.approx(0.3)
    assert "stage_durations_p99" in summaries
    assert "stage_0_gen_ms" not in summaries["stage_durations_mean"]
    assert "stage_1_gen_ms" not in summaries["stage_durations_mean"]


def test_print_stage_durations_metrics(capsys) -> None:
    output = MixRequestFuncOutput()
    output.success = True
    output.stage_durations = {
        "Wan22I2VPipeline.diffuse": 1.25,
        "Wan22I2VPipeline.text_encoder.forward": 0.4,
        "queue_wait_ms": 0.5,
        "stage_0_gen_ms": 1000.0,
    }
    summaries = aggregate_stage_durations([output])
    metrics = SimpleNamespace(**summaries)
    print_stage_durations_metrics([99.0], metrics)
    out = capsys.readouterr().out
    assert "Wan22I2VPipeline" not in out
    assert "Mean Diffuse (s):" in out
    assert "Median Diffuse (s):" in out
    assert "P99 Diffuse (s):" in out
    assert "Mean Text Encoder Forward (s):" in out
    assert "Mean Queue Wait (ms):" in out
    assert "Stage 0 Gen" not in out
    assert "stage_0_gen" not in out


def test_profiler_stage_durations_print_without_print_stage(capsys) -> None:
    output = MixRequestFuncOutput()
    output.success = True
    output.prompt_len = 8
    output.latency = 1.0
    output.stage_durations = {
        "Wan22I2VPipeline.diffuse": 1.25,
        "queue_wait_ms": 0.5,
        "stage_0_gen_ms": 1000.0,
    }
    common = dict(
        input_requests=[],
        outputs=[output],
        dur_s=1.0,
        tokenizer=None,
        selected_percentiles=[50.0, 99.0],
        goodput_config_dict={},
        task_type=TaskType.GENERATION,
        selected_percentile_metrics=["e2el"],
        max_concurrency=None,
        request_rate=float("inf"),
        benchmark_duration=1.0,
    )
    metrics, _ = calculate_metrics(**common, print_stage=False)
    shown = capsys.readouterr().out
    assert "Mean Diffuse (s):" in shown
    assert "Mean Queue Wait (ms):" in shown
    assert "Stage 0 Gen" not in shown
    assert metrics.stage_durations_mean["Wan22I2VPipeline.diffuse"] == pytest.approx(1.25)
    assert "stage_0_gen_ms" not in metrics.stage_durations_mean


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
