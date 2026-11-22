# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import pytest

from vllm_omni.benchmarks.duplex_session_metrics import (
    build_duplex_metrics_report,
    collect_duplex_session_metrics,
    duplex_response_latency_metrics,
    print_duplex_response_latency_metrics,
)
from vllm_omni.clients.duplex import EventCollector

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.benchmark]


def _server_timed(event: dict[str, object], *, ttft_ms: float, ttfp_ms: float) -> dict[str, object]:
    """Attach the server's request-start TTFT/TTFP, as the duplex runtime does."""
    event["metadata"] = {
        "vllm_omni": {
            "response_request_metrics": {
                "source": "server_monotonic_request_start",
                "measurement_origin": {"ttft": "accepted native-append start", "ttfp": "accepted native-append start"},
                "ttft_ms": ttft_ms,
                "ttfp_ms": ttfp_ms,
            }
        }
    }
    return event


def _turn(collector: EventCollector, response_id: str, created_at_s: float, *, text_s: float, audio_s: float) -> None:
    collector.add({"type": "response.created", "response": {"id": response_id}}, received_at_s=created_at_s)
    collector.add(
        _server_timed(
            {"type": "response.output_audio_transcript.delta", "response_id": response_id, "delta": "Hi"},
            ttft_ms=50.0,
            ttfp_ms=150.0,
        ),
        received_at_s=created_at_s + text_s,
    )
    collector.add(
        _server_timed(
            {"type": "response.output_audio.delta", "response_id": response_id, "delta": "AAAA"},
            ttft_ms=50.0,
            ttfp_ms=150.0,
        ),
        received_at_s=created_at_s + audio_s,
    )
    collector.add({"type": "response.done", "response": {"id": response_id}}, received_at_s=created_at_s + 1.0)


def test_client_latency_includes_what_the_server_ttft_leaves_out():
    # The server stamps TTFT after it built the prompt and submitted it, so its
    # 50 ms hides the 250 ms the client waited after response.created.
    collector = EventCollector()
    _turn(collector, "r1", 10.0, text_s=0.3, audio_s=0.4)

    bundle = collect_duplex_session_metrics(collector, stream_start=9.0, session_id="s")

    (metric,) = bundle.request_metrics
    assert (metric["ttft_ms"], metric["ttfp_ms"]) == (50.0, 150.0)
    assert (metric["client_ttft_ms"], metric["client_ttfp_ms"]) == (300.0, 400.0)
    assert "prompt preparation" in metric["measurement_origin"]["client_ttft"]
    assert metric["measurement_origin"]["ttft"] == "accepted native-append start"
    assert bundle.session_metrics["client_ttft_ms"]["mean"] == 300.0
    assert bundle.session_metrics["client_ttfp_ms"]["mean"] == 400.0


def test_client_latency_is_absent_for_a_response_without_output():
    collector = EventCollector()
    _turn(collector, "r1", 10.0, text_s=0.3, audio_s=0.4)
    collector.add({"type": "response.created", "response": {"id": "r2"}}, received_at_s=12.0)
    collector.add({"type": "response.done", "response": {"id": "r2"}}, received_at_s=12.1)

    bundle = collect_duplex_session_metrics(collector, stream_start=9.0, session_id="s")

    assert [metric["response_id"] for metric in bundle.request_metrics] == ["r1"]


def test_response_latency_counts_every_response_once_across_sessions():
    # One session has three responses, the other one. A mean of session means
    # would weigh the lone response like the other three together.
    rows: list[dict[str, object]] = [{"ttft_ms": 10.0, "client_ttft_ms": 100.0}] * 3
    rows.append({"ttft_ms": 50.0, "client_ttft_ms": 500.0})
    rows.append({"ttft_ms": None, "client_ttft_ms": float("nan"), "ttfp_ms": True})

    metrics = duplex_response_latency_metrics(rows)

    assert metrics == {
        "num_duplex_response_ttft_ms_samples": 4,
        "mean_duplex_response_ttft_ms": 20.0,
        "median_duplex_response_ttft_ms": 10.0,
        "p99_duplex_response_ttft_ms": 50.0,
        "num_duplex_client_ttft_ms_samples": 4,
        "mean_duplex_client_ttft_ms": 200.0,
        "median_duplex_client_ttft_ms": 100.0,
        "p99_duplex_client_ttft_ms": 500.0,
    }
    assert duplex_response_latency_metrics([]) == {}


def test_report_and_console_expose_the_flat_latency_keys(capsys):
    collector = EventCollector()
    _turn(collector, "r1", 10.0, text_s=0.3, audio_s=0.4)
    bundle = collect_duplex_session_metrics(collector, stream_start=9.0, session_id="s")

    report = build_duplex_metrics_report(
        request_metrics=bundle.request_metrics,
        session_metrics=[bundle.session_metrics],
    )
    assert report["mean_duplex_client_ttfp_ms"] == 400.0
    assert report["mean_duplex_response_ttfp_ms"] == 150.0

    print_duplex_response_latency_metrics(duplex_response_latency_metrics(bundle.request_metrics))
    printed = capsys.readouterr().out
    assert "Duplex Per-Response Latency" in printed
    assert "Mean client TTFP (ms):" in printed and "400.00" in printed
    print_duplex_response_latency_metrics({})
    assert capsys.readouterr().out == ""
