# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from benchmarks.dreamzero_async.compare_replays import summarize as summarize_pair
from benchmarks.dreamzero_async.live_benchmark import (
    _async_command,
    _speedup_requirement_exit_code,
    _sync_command,
    summarize_suite,
    write_suite_table,
)
from benchmarks.dreamzero_async.suite_benchmark import (
    _benchmark_command,
    summarize_benchmark_suite,
    validate_config,
    write_comparison_table,
)

pytestmark = [pytest.mark.core_model, pytest.mark.benchmark, pytest.mark.cpu]


def _sync_summary(time_s: float, *, chunks: int = 2) -> dict:
    return {
        "action_chunk_count": chunks,
        "executed_rows": chunks * 24,
        "total_closed_loop_time_s": time_s,
        "idle_time_s": time_s - chunks * 24 / 15.0,
        "effective_control_idle_ratio": 0.7,
    }


def _async_summary(
    time_s: float,
    *,
    chunks: int = 2,
    underruns: int = 0,
    errors: int = 0,
    realtime: bool | None = None,
    non_sim_chunks: list[int] | None = None,
    deadline_misses: int = 0,
) -> dict:
    config = {} if realtime is None else {"realtime": realtime}
    return {
        "action_chunk_count": chunks,
        "executed_rows": chunks * 24,
        "total_elapsed_s": time_s,
        "bootstrap_latency_s": 4.5,
        "underruns": underruns,
        "deadline_miss_count": deadline_misses,
        "sim_conditioned_post_bootstrap_chunks": [] if non_sim_chunks else list(range(2, chunks + 1)),
        "non_sim_conditioned_post_bootstrap_chunks": non_sim_chunks or [],
        "server_error_count": errors,
        "config": config,
    }


def _args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        host="127.0.0.1",
        port=8000,
        python="python",
        sync_python=None,
        async_python=None,
        sync_path="/v1/realtime/robot/openpi",
        async_path="/v1/realtime/robot/dreamzero-async",
        num_chunks=2,
        control_hz=15.0,
        repeat_last_observation=False,
        realtime=False,
        bootstrap_timeout_s=180.0,
        chunk_timeout_s=10.0,
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
    assert summary["validity"]["speedup_claim_valid"] is True


def test_summarize_pair_marks_underrun_speedup_as_invalid():
    summary = summarize_pair(_sync_summary(12.0), _async_summary(9.0, underruns=24), control_hz=15.0)

    assert summary["gain"]["speedup"] == pytest.approx(1.333333)
    assert summary["validity"]["speedup_claim_valid"] is False
    assert "underruns" in summary["validity"]["reason"]


def test_summarize_pair_marks_slower_async_as_invalid_speedup():
    summary = summarize_pair(_sync_summary(12.0), _async_summary(14.0, realtime=True), control_hz=15.0)

    assert summary["gain"]["speedup"] == pytest.approx(0.857143)
    assert summary["validity"]["speedup_claim_valid"] is False
    assert "not faster than sync" in summary["validity"]["reason"]


def test_summarize_pair_rejects_non_realtime_strict_claim():
    summary = summarize_pair(_sync_summary(12.0), _async_summary(9.0, realtime=False), control_hz=15.0)

    assert summary["validity"]["speedup_claim_valid"] is False
    assert "realtime control timing" in summary["validity"]["reason"]


def test_summarize_pair_allows_waited_deadline_misses_with_full_coverage():
    summary = summarize_pair(
        _sync_summary(12.0),
        _async_summary(9.0, realtime=True, deadline_misses=1),
        control_hz=15.0,
    )

    assert summary["validity"]["speedup_claim_valid"] is True
    assert summary["dreamzero_async"]["deadline_miss_count"] == 1


def test_summarize_pair_rejects_non_sim_conditioned_post_bootstrap_chunks():
    summary = summarize_pair(
        _sync_summary(12.0),
        _async_summary(9.0, realtime=True, non_sim_chunks=[3], deadline_misses=1),
        control_hz=15.0,
    )

    assert summary["validity"]["speedup_claim_valid"] is False
    assert "post-bootstrap chunks were not sim-conditioned: [3]" in summary["validity"]["reason"]


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
    assert suite["summary"]["async_deadline_miss_total"] == 0
    assert suite["summary"]["async_non_sim_conditioned_post_bootstrap_total"] == 0
    assert suite["summary"]["async_server_error_total"] == 1
    assert suite["summary"]["valid_speedup_repeats"] == 0
    assert suite["summary"]["invalid_speedup_repeats"] == 2
    assert suite["summary"]["speedup_claim_valid"] is False


def test_write_suite_table_and_json_payload_are_stable(tmp_path: Path):
    pair = summarize_pair(_sync_summary(12.0), _async_summary(9.0), control_hz=15.0)
    pair["run"] = {"index": 1, "warmup": False, "order": "async-first"}
    suite = summarize_suite([pair], _args(tmp_path))
    table_path = tmp_path / "result_table.md"

    write_suite_table(table_path, suite)
    (tmp_path / "summary.json").write_text(json.dumps(suite, indent=2, sort_keys=True), encoding="utf-8")

    table = table_path.read_text(encoding="utf-8")
    assert "| Raw mean speedup | 1.333x |" in table
    assert "| Speedup claim valid | True |" in table
    assert json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))["summary"]["measured_repeats"] == 1


def test_speedup_requirement_exit_code_respects_validity(tmp_path: Path):
    valid_pair = summarize_pair(_sync_summary(12.0), _async_summary(9.0), control_hz=15.0)
    invalid_pair = summarize_pair(_sync_summary(12.0), _async_summary(9.0, underruns=24), control_hz=15.0)
    for pair in (valid_pair, invalid_pair):
        pair["run"] = {"index": 1, "warmup": False, "order": "async-first"}

    assert _speedup_requirement_exit_code(summarize_suite([valid_pair], _args(tmp_path)), require_valid=True) == 0
    assert _speedup_requirement_exit_code(summarize_suite([invalid_pair], _args(tmp_path)), require_valid=True) == 1
    assert _speedup_requirement_exit_code(summarize_suite([invalid_pair], _args(tmp_path)), require_valid=False) == 0


def test_summarize_benchmark_suite_builds_pr_style_comparison(tmp_path: Path):
    variants = [
        {
            "name": "baseline",
            "description": "baseline server",
            "summary": {
                "sync_time_mean_s": 14.0,
                "async_time_mean_s": 10.0,
                "time_saved_mean_s": 4.0,
                "speedup_mean": 1.4,
                "async_underrun_total": 0,
                "async_deadline_miss_total": 0,
                "async_non_sim_conditioned_post_bootstrap_total": 0,
                "async_server_error_total": 0,
                "measured_repeats": 2,
                "valid_speedup_repeats": 2,
                "invalid_speedup_repeats": 0,
                "speedup_claim_valid": True,
            },
        },
        {
            "name": "faster",
            "description": "optimized server",
            "summary": {
                "sync_time_mean_s": 7.0,
                "async_time_mean_s": 5.0,
                "time_saved_mean_s": 2.0,
                "speedup_mean": 1.4,
                "async_underrun_total": 1,
                "async_deadline_miss_total": 0,
                "async_non_sim_conditioned_post_bootstrap_total": 0,
                "async_server_error_total": 0,
                "measured_repeats": 2,
                "valid_speedup_repeats": 1,
                "invalid_speedup_repeats": 1,
                "speedup_claim_valid": False,
            },
        },
    ]

    suite = summarize_benchmark_suite(variants, baseline_name="baseline", environment={"gpu": ["A100"]})
    table_path = tmp_path / "result_table.md"
    write_comparison_table(table_path, suite)

    assert suite["baseline"] == "baseline"
    assert suite["comparison"][1]["async_vs_baseline_async"] == pytest.approx(2.0)
    assert "| faster | 7.000 | 5.000 | 1.400x | no | 2.000x | 1 | 0 |" in table_path.read_text(
        encoding="utf-8"
    )


def test_validate_suite_config_requires_explicit_variants():
    with pytest.raises(ValueError, match="at least one variant"):
        validate_config({"server": {}, "benchmark": {}, "variants": []})

    validate_config(
        {
            "server": {},
            "benchmark": {},
            "variants": [{"name": "current", "command": ["python", "-V"]}],
        }
    )


def test_suite_benchmark_command_preserves_replay_shape(tmp_path: Path):
    config = {
        "python": "python",
        "server": {"host": "127.0.0.1", "port": 8000},
        "benchmark": {
            "video_dir": "outputs/dreamzero/assets",
            "num_chunks": 15,
            "control_hz": 15.0,
            "warmups": 1,
            "repeats": 3,
            "order": "sync-first",
            "repeat_last_observation": True,
            "realtime": True,
            "require_valid_speedup": True,
            "chunk_timeout_s": 10,
            "sync_python": "/tmp/sync-python",
            "async_python": "/tmp/async-python",
        },
    }
    command = _benchmark_command(config, {"name": "bde"}, tmp_path / "out")

    assert "--num-chunks" in command
    assert command[command.index("--num-chunks") + 1] == "15"
    assert "--warmups" in command
    assert command[command.index("--warmups") + 1] == "1"
    assert "--repeats" in command
    assert command[command.index("--repeats") + 1] == "3"
    assert "--repeat-last-observation" in command
    assert "--realtime" in command
    assert "--require-valid-speedup" in command
    assert command[command.index("--chunk-timeout-s") + 1] == "10"
    assert command[command.index("--sync-python") + 1] == "/tmp/sync-python"
    assert command[command.index("--async-python") + 1] == "/tmp/async-python"


def test_live_benchmark_realtime_only_applies_to_async_client(tmp_path: Path):
    args = _args(tmp_path)
    args.realtime = True

    assert "--realtime" not in _sync_command(args, tmp_path / "sync")
    assert "--realtime" in _async_command(args, tmp_path / "async")


def test_live_benchmark_allows_separate_client_pythons(tmp_path: Path):
    args = _args(tmp_path)
    args.python = "server-python"
    args.sync_python = "sync-python"
    args.async_python = "async-python"

    assert _sync_command(args, tmp_path / "sync")[0] == "sync-python"
    assert _async_command(args, tmp_path / "async")[0] == "async-python"
