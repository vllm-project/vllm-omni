# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import pytest

from vllm_omni.benchmarks.duplex_session_metrics import (
    build_duplex_metrics_report,
    collect_duplex_session_metrics,
    duplex_response_latency_metrics,
    print_duplex_response_latency_metrics,
    session_stats_turn_metrics,
    union_ns,
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
    assert metric["client_ttft_ms"] == 300.0
    assert metric["response_created_to_first_audio_ms"] == 400.0
    assert "client_ttfp_ms" not in metric
    assert "prompt preparation" in metric["measurement_origin"]["client_ttft"]
    assert metric["measurement_origin"]["ttft"] == "accepted native-append start"
    assert bundle.session_metrics["client_ttft_ms"]["mean"] == 300.0
    assert "client_ttfp_ms" not in bundle.session_metrics


def test_client_append_ttfp_is_not_mixed_with_created_ttfp():
    collector = EventCollector()
    collector.note_client_append({"sent_at_s": 9.6, "event_id": "append-last"})
    _turn(collector, "r1", 10.0, text_s=0.3, audio_s=0.4)

    bundle = collect_duplex_session_metrics(collector, stream_start=9.0, session_id="s")

    (metric,) = bundle.request_metrics
    assert metric["client_ttfp_ms"] == 800.0
    assert metric["response_created_to_first_audio_ms"] == 400.0
    assert bundle.session_metrics["client_ttfp_ms"]["mean"] == 800.0


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
    assert "mean_duplex_client_ttfp_ms" not in report
    assert report["mean_duplex_response_ttfp_ms"] == 150.0
    assert bundle.request_metrics[0]["response_created_to_first_audio_ms"] == 400.0

    print_duplex_response_latency_metrics(duplex_response_latency_metrics(bundle.request_metrics))
    printed = capsys.readouterr().out
    assert "Duplex Per-Response Latency" in printed
    assert "Mean client TTFT (ms):" in printed and "300.00" in printed
    assert "Mean client append TTFP" not in printed
    print_duplex_response_latency_metrics({})
    assert capsys.readouterr().out == ""


# --------------------------------------------------------------------------- #
# Server-side session stats (RFC #8325)                                       #
# --------------------------------------------------------------------------- #

_MS = 1_000_000


def _audio(duration_ms):
    return {"response_id": "r", "emitted_ns": 0, "duration_ms": duration_ms}


def _stats_chunk(seq, submitted, done, spans_by_stage, *, complete=True, truncated=False):
    return {
        "seq": seq,
        "submitted_ns": submitted * _MS,
        "done_ns": None if done is None else done * _MS,
        "complete": complete,
        "truncated": truncated,
        "stage_work": [
            {"stage_id": stage_id, "spans": [[start * _MS, length * _MS] for start, length in spans]}
            for stage_id, spans in spans_by_stage.items()
        ],
    }


def test_union_ns_counts_overlap_once() -> None:
    assert union_ns([]) == 0
    assert union_ns([[0, 10], [5, 10], [30, 5]]) == 20
    assert union_ns([[30, 5], [0, 10], [10, 5]]) == 20


def test_turn_metrics_union_pipelined_chunks_and_skip_input_waits() -> None:
    # Chunk 1: [0, 100] in engine; stage spans 10-50 and 40-90. Chunk 2 arrives at
    # 1000 (input wait not counted) and pipelines with nothing: 1010-1060.
    turn = {
        "epoch": 0,
        "turn_id": 3,
        "status": "done",
        "final": True,
        "chunks": [
            _stats_chunk(1, 0, 100, {0: [(10, 40)], 1: [(40, 50)]}),
            _stats_chunk(2, 1000, 1070, {0: [(1010, 50)]}),
        ],
        "audio": [_audio(500.0), _audio(500.0)],
    }

    metrics = session_stats_turn_metrics(turn)

    assert metrics["final"] is True
    assert metrics["generation_ms"] == 130.0  # 10..90 plus 1010..1060
    assert metrics["turn_span_ms"] == 170.0  # 0..100 plus 1000..1070
    assert metrics["audio_ms"] == 1000.0
    assert metrics["busy_rtf"] == 0.13
    assert metrics["stall_ms"] == 40.0
    assert metrics["stall_ratio"] == 0.04


@pytest.mark.parametrize(
    "turn",
    [
        # The server's verdict decides: not final means no numbers, even if every chunk looks done.
        {"status": "done", "final": False, "chunks": [_stats_chunk(1, 0, 10, {0: [(0, 5)]})], "audio": [_audio(1.0)]},
        {"status": "cancelled", "final": False, "chunks": [_stats_chunk(1, 0, None, {}, complete=False)]},
        # A final flag without chunk end times is not trusted either.
        {"status": "done", "final": True, "chunks": [_stats_chunk(1, 0, None, {})]},
    ],
)
def test_turn_metrics_are_withheld_until_final(turn: dict[str, object]) -> None:
    metrics = session_stats_turn_metrics({"epoch": 0, "turn_id": 0, **turn})
    assert metrics == {"final": False, "epoch": 0, "turn_id": 0}


def test_listen_turn_has_stall_time_but_no_ratios() -> None:
    turn = {
        "epoch": 0,
        "turn_id": 0,
        "status": "listen",
        "final": True,
        "chunks": [_stats_chunk(1, 0, 30, {0: [(10, 10)]})],
        "audio": [],
    }
    metrics = session_stats_turn_metrics(turn)
    assert (metrics["busy_rtf"], metrics["stall_ratio"], metrics["stall_ms"]) == (None, None, 20.0)


def test_an_audio_turn_missing_a_stages_spans_gets_no_numbers() -> None:
    turn = {
        "epoch": 0,
        "turn_id": 0,
        "status": "done",
        "final": True,
        "chunks": [_stats_chunk(1, 0, 100, {0: [(10, 40)], 1: [(40, 50)]})],
        "audio": [_audio(500.0)],
    }
    # Busy RTF without the vocoder's work would read low.
    assert session_stats_turn_metrics(turn, stage_ids=[0, 1, 2]) == {
        "final": False,
        "epoch": 0,
        "turn_id": 0,
        "missing_stage_ids": [2],
    }
    assert session_stats_turn_metrics(turn, stage_ids=[0, 1])["busy_rtf"] == 0.16
    # A listen turn needs no work from the speaking stages.
    listen = {**turn, "status": "listen", "audio": []}
    assert session_stats_turn_metrics(listen, stage_ids=[0, 1, 2])["final"] is True
