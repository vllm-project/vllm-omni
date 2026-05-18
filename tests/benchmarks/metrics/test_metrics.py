# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Unit tests for metrics.py
"""

import pytest
from vllm.benchmarks.serve import TaskType

from vllm_omni.benchmarks.metrics.metrics import calculate_metrics
from vllm_omni.benchmarks.patch.patch import MixRequestFuncOutput

pytestmark = [pytest.mark.core_model, pytest.mark.benchmark, pytest.mark.cpu]


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
