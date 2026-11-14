# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import pytest

from vllm_omni.metrics import OrchestratorAggregator
from vllm_omni.metrics.stats import RequestE2EStats, StageRequestStats, StageStats

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _get_request_entry(table: list[dict], request_id: str) -> dict:
    for entry in table:
        if entry.get("request_id") == request_id:
            return entry
    raise AssertionError(f"request_id={request_id} not found")


def test_orchestrator_aggregator_builds_summary() -> None:
    agg = OrchestratorAggregator(num_stages=2, log_stats=True, wall_start_ts=0.0, final_stage_id_for_e2e=1)
    agg.stage_first_ts[0] = 0.0
    agg.stage_last_ts[0] = 0.03
    agg.stage_first_ts[1] = 0.05
    agg.stage_last_ts[1] = 0.07

    agg.on_forward(0, 1, "r1", size_bytes=1024, tx_ms=5.0, used_shm=False)

    agg.on_stage_metrics(
        0,
        "r1",
        StageRequestStats(
            batch_id=1,
            batch_size=1,
            num_tokens_in=3,
            num_tokens_out=3,
            stage_gen_time_ms=30.0,
            rx_transfer_bytes=0,
            rx_decode_time_ms=0.0,
            rx_in_flight_time_ms=0.0,
            stage_stats=StageStats(),
        ),
    )
    agg.on_stage_metrics(
        1,
        "r1",
        StageRequestStats(
            batch_id=1,
            batch_size=1,
            num_tokens_in=0,
            num_tokens_out=4,
            stage_gen_time_ms=20.0,
            rx_transfer_bytes=1024,
            rx_decode_time_ms=5.0,
            rx_in_flight_time_ms=2.0,
            stage_stats=StageStats(),
        ),
    )
    agg.on_finalize_request(1, "r1", req_start_ts=0.0)

    summary = agg.build_and_log_summary()
    overall = summary["overall_summary"]
    assert overall["e2e_requests"] == 1

    stage_entry = _get_request_entry(summary["stage_table"], "r1")
    stage_ids = [row["stage_id"] for row in stage_entry["stages"]]
    assert stage_ids == [0, 1]

    transfer_entry = _get_request_entry(summary["trans_table"], "r1")
    assert transfer_entry["transfers"][0]["edge"] == "0->1"
    assert transfer_entry["transfers"][0]["size_kbytes"] == 1.0

    e2e_entry = _get_request_entry(summary["e2e_table"], "r1")
    assert e2e_entry["e2e_total_tokens"] == 10


def test_build_and_log_summary_e2e_only() -> None:
    agg = OrchestratorAggregator(num_stages=1, log_stats=True, wall_start_ts=0.0, final_stage_id_for_e2e=0)
    agg.e2e_events.append(
        RequestE2EStats(
            request_id="r",
            e2e_total_ms=10.0,
            e2e_total_tokens=5,
            transfers_total_time_ms=0.0,
            transfers_total_bytes=0,
        )
    )

    summary = agg.build_and_log_summary()
    e2e_entry = _get_request_entry(summary["e2e_table"], "r")
    assert e2e_entry["e2e_total_tokens"] == 5
    stage_entry = _get_request_entry(summary["stage_table"], "r")
    assert stage_entry["stages"] == []


def test_build_and_log_summary_multiple_requests() -> None:
    agg = OrchestratorAggregator(
        num_stages=2, log_stats=True, wall_start_ts=0.0, final_stage_id_for_e2e={"r1": 1, "r2": 0}
    )

    # Request r1 goes through both stages
    agg.on_stage_metrics(
        0,
        "r1",
        StageRequestStats(
            batch_id=1,
            batch_size=1,
            num_tokens_in=2,
            num_tokens_out=4,
            stage_gen_time_ms=10.0,
            rx_transfer_bytes=0,
            rx_decode_time_ms=0.0,
            rx_in_flight_time_ms=0.0,
            stage_stats=StageStats(),
        ),
    )
    agg.on_stage_metrics(
        1,
        "r1",
        StageRequestStats(
            batch_id=1,
            batch_size=1,
            num_tokens_in=4,
            num_tokens_out=5,
            stage_gen_time_ms=8.0,
            rx_transfer_bytes=0,
            rx_decode_time_ms=0.0,
            rx_in_flight_time_ms=0.0,
            stage_stats=StageStats(),
        ),
    )
    agg.on_finalize_request(1, "r1", req_start_ts=0.0)

    # Request r2 only goes through stage 0
    agg.on_stage_metrics(
        0,
        "r2",
        StageRequestStats(
            batch_id=2,
            batch_size=1,
            num_tokens_in=1,
            num_tokens_out=2,
            stage_gen_time_ms=12.0,
            rx_transfer_bytes=0,
            rx_decode_time_ms=0.0,
            rx_in_flight_time_ms=0.0,
            stage_stats=StageStats(),
        ),
    )
    agg.on_finalize_request(0, "r2", req_start_ts=0.0)

    summary = agg.build_and_log_summary()
    assert len(summary["stage_table"]) == 2
    assert {entry["request_id"] for entry in summary["e2e_table"]} == {"r1", "r2"}
    # Check that r1 has two stages and r2 has one
    r1_stage_entry = next(e for e in summary["stage_table"] if e["request_id"] == "r1")
    r2_stage_entry = next(e for e in summary["stage_table"] if e["request_id"] == "r2")
    assert len(r1_stage_entry["stages"]) == 2
    assert len(r2_stage_entry["stages"]) == 1


def test_log_timing_summary_skips_tables_unless_debug(monkeypatch) -> None:
    from vllm_omni.metrics import stats

    agg = OrchestratorAggregator(num_stages=1, log_stats=True, wall_start_ts=0.0, final_stage_id_for_e2e=0)
    agg.stage_first_ts[0] = 0.0
    agg.stage_last_ts[0] = 0.02
    agg.on_stage_metrics(
        0,
        "r1",
        StageRequestStats(
            batch_id=1,
            batch_size=1,
            num_tokens_in=3,
            num_tokens_out=3,
            stage_gen_time_ms=20.0,
            rx_transfer_bytes=0,
            rx_decode_time_ms=0.0,
            rx_in_flight_time_ms=0.0,
            stage_stats=StageStats(),
        ),
    )
    agg.on_finalize_request(0, "r1", req_start_ts=0.0)
    tables, lines = [], []

    def record_table(*args, **kwargs):
        tables.append(args)
        return ""

    monkeypatch.setattr(stats, "_format_table", record_table)
    monkeypatch.setattr(stats.logger, "info", lambda msg, *args: lines.append(msg % args))
    monkeypatch.setattr(stats.logger, "isEnabledFor", lambda level: False)
    agg.log_timing_summary()
    assert not tables
    assert len(lines) == 1 and lines[0].startswith("[OmniTiming] req=r1 total=")


_PHASE_FIELDS = ("vllm_queued_ms", "vllm_prefill_ms", "vllm_decode_ms", "vllm_num_preemptions")


def _phase_split_summary(**phase_split) -> dict:
    agg = OrchestratorAggregator(num_stages=2, log_stats=True, wall_start_ts=0.0, final_stage_id_for_e2e=1)
    for stage_id, split in ((0, phase_split), (1, {})):  # stage 1 stands in for a diffusion stage
        agg.on_stage_metrics(
            stage_id,
            "r1",
            StageRequestStats(
                batch_id=1,
                batch_size=1,
                num_tokens_in=0,
                num_tokens_out=4,
                stage_gen_time_ms=20.0,
                rx_transfer_bytes=0,
                rx_decode_time_ms=0.0,
                rx_in_flight_time_ms=0.0,
                stage_stats=StageStats(),
                **split,
            ),
        )
    agg.on_finalize_request(1, "r1", req_start_ts=0.0)
    return agg.build_and_log_summary()


def _logged_stage_table_rows(monkeypatch, **phase_split) -> dict[str, list[str]]:
    from vllm_omni.metrics import stats

    tables: list[str] = []
    monkeypatch.setattr(stats.logger, "isEnabledFor", lambda level: True)
    monkeypatch.setattr(stats.logger, "debug", lambda msg, *args: tables.append(msg % args))
    monkeypatch.setattr(stats.logger, "info", lambda *args, **kwargs: None)
    _phase_split_summary(**phase_split)
    table = next(table for table in tables if "[StageRequestStats [request_id=r1]]" in table)
    rows = {}
    for line in table.splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        rows[cells[0]] = cells[1:]
    return rows


def test_stage_table_summary_keeps_measured_zero_apart_from_missing_phase_split() -> None:
    summary = _phase_split_summary(vllm_queued_ms=3.0, vllm_prefill_ms=40.0, vllm_decode_ms=0.0, vllm_num_preemptions=0)

    ar_row, diffusion_row = _get_request_entry(summary["stage_table"], "r1")["stages"]
    assert {field: ar_row[field] for field in _PHASE_FIELDS} == {
        "vllm_queued_ms": 3.0,
        "vllm_prefill_ms": 40.0,
        "vllm_decode_ms": 0.0,
        "vllm_num_preemptions": 0,
    }
    assert all(diffusion_row[field] is None for field in _PHASE_FIELDS)


def test_logged_stage_table_renders_phase_split_rows(monkeypatch) -> None:
    rows = _logged_stage_table_rows(
        monkeypatch, vllm_queued_ms=3.0, vllm_prefill_ms=40.0, vllm_decode_ms=900.0, vllm_num_preemptions=2
    )

    assert {field: rows[field] for field in _PHASE_FIELDS} == {
        "vllm_queued_ms": ["3.000", "None"],
        "vllm_prefill_ms": ["40.000", "None"],
        "vllm_decode_ms": ["900.000", "None"],
        "vllm_num_preemptions": ["2", "None"],
    }


def test_logged_stage_table_has_no_phase_split_rows_when_unobserved(monkeypatch) -> None:
    rows = _logged_stage_table_rows(monkeypatch)

    assert "stage_gen_time_ms" in rows
    assert not set(_PHASE_FIELDS) & set(rows)


_INPUT_SPLIT_FIELDS = ("vllm_input_wait_ms", "vllm_ready_queued_ms")


def test_stage_table_summary_keeps_input_wait_split() -> None:
    summary = _phase_split_summary(
        vllm_queued_ms=3.0,
        vllm_prefill_ms=40.0,
        vllm_decode_ms=0.0,
        vllm_num_preemptions=0,
        vllm_input_wait_ms=2.25,
        vllm_ready_queued_ms=0.75,
    )

    ar_row, diffusion_row = _get_request_entry(summary["stage_table"], "r1")["stages"]
    assert {field: ar_row[field] for field in _INPUT_SPLIT_FIELDS} == {
        "vllm_input_wait_ms": 2.25,
        "vllm_ready_queued_ms": 0.75,
    }
    assert all(diffusion_row[field] is None for field in _INPUT_SPLIT_FIELDS)


def _stage_chunk(**split) -> StageRequestStats:
    return StageRequestStats(
        batch_id=1,
        batch_size=1,
        num_tokens_in=0,
        num_tokens_out=4,
        stage_gen_time_ms=20.0,
        rx_transfer_bytes=0,
        rx_decode_time_ms=0.0,
        rx_in_flight_time_ms=0.0,
        stage_stats=StageStats(),
        stage_id=1,
        **split,
    )


def test_one_row_per_stage_takes_input_split_from_the_chunk_with_the_phase_split() -> None:
    from vllm_omni.metrics.stats import _one_row_per_stage

    split = dict(
        vllm_queued_ms=3.0,
        vllm_prefill_ms=40.0,
        vllm_decode_ms=900.0,
        vllm_num_preemptions=0,
        vllm_input_wait_ms=2.0,
        vllm_ready_queued_ms=1.0,
    )
    # The template (first) chunk has no split; the finishing chunk has it.
    (row,) = _one_row_per_stage([_stage_chunk(), _stage_chunk(**split), _stage_chunk()])
    assert {field: getattr(row, field) for field in split} == split

    # A later split without the input fields does not keep an earlier input split.
    later = dict(split, vllm_queued_ms=None, vllm_input_wait_ms=None, vllm_ready_queued_ms=None)
    (row,) = _one_row_per_stage([_stage_chunk(**split), _stage_chunk(**later)])
    assert {field: getattr(row, field) for field in later} == later
