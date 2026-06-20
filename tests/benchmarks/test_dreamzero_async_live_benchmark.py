# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from benchmarks.dreamzero_async.compare_replays import summarize as summarize_pair
from benchmarks.dreamzero_async.live_benchmark import summarize_suite, write_suite_table

pytestmark = [pytest.mark.core_model, pytest.mark.benchmark, pytest.mark.cpu]


def _sync_summary(time_s: float, *, chunks: int = 2) -> dict:
    return {
        "action_chunk_count": chunks,
        "executed_rows": chunks * 24,
        "total_closed_loop_time_s": time_s,
        "idle_time_s": time_s - chunks * 24 / 15.0,
        "effective_control_idle_ratio": 0.7,
    }


def _async_summary(time_s: float, *, chunks: int = 2, underruns: int = 0, errors: int = 0) -> dict:
    return {
        "action_chunk_count": chunks,
        "executed_rows": chunks * 24,
        "total_elapsed_s": time_s,
        "bootstrap_latency_s": 4.5,
        "underruns": underruns,
        "server_error_count": errors,
    }


def _args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        host="127.0.0.1",
        port=8000,
        num_chunks=2,
        control_hz=15.0,
        order="async-first",
        repeats=2,
        warmups=1,
        video_dir=tmp_path / "assets",
    )


def test_summarize_pair_computes_speedup_and_idle_proxy():
    summary = summarize_pair(_sync_summary(12.0), _async_summary(9.0), control_hz=15.0)

    assert summary["gain"]["time_saved_s"] == pytest.approx(3.0)
    assert summary["gain"]["speedup"] == pytest.approx(1.333333)
    assert summary["dreamzero_async"]["idle_proxy_s"] == pytest.approx(5.8)


def test_summarize_suite_skips_warmups_and_accumulates_failures(tmp_path: Path):
    pairs = [
        summarize_pair(_sync_summary(100.0), _async_summary(80.0), control_hz=15.0),
        summarize_pair(_sync_summary(12.0), _async_summary(9.0, underruns=1), control_hz=15.0),
        summarize_pair(_sync_summary(10.0), _async_summary(8.0, errors=1), control_hz=15.0),
    ]
    for index, pair in enumerate(pairs):
        pair["run"] = {"index": index + 1, "warmup": index == 0, "order": "async-first"}

    suite = summarize_suite(pairs, _args(tmp_path))

    assert suite["summary"]["measured_repeats"] == 2
    assert suite["summary"]["sync_time_mean_s"] == pytest.approx(11.0)
    assert suite["summary"]["async_time_mean_s"] == pytest.approx(8.5)
    assert suite["summary"]["time_saved_mean_s"] == pytest.approx(2.5)
    assert suite["summary"]["async_underrun_total"] == 1
    assert suite["summary"]["async_server_error_total"] == 1


def test_write_suite_table_and_json_payload_are_stable(tmp_path: Path):
    pair = summarize_pair(_sync_summary(12.0), _async_summary(9.0), control_hz=15.0)
    pair["run"] = {"index": 1, "warmup": False, "order": "async-first"}
    suite = summarize_suite([pair], _args(tmp_path))
    table_path = tmp_path / "result_table.md"

    write_suite_table(table_path, suite)
    (tmp_path / "summary.json").write_text(json.dumps(suite, indent=2, sort_keys=True), encoding="utf-8")

    table = table_path.read_text(encoding="utf-8")
    assert "| Mean speedup | 1.333x |" in table
    assert json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))["summary"]["measured_repeats"] == 1
