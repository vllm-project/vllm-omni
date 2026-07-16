from __future__ import annotations

import json

import pytest

from vllm_omni.metrics.concurrency_trace import (
    build_summary,
    emit_concurrency_trace,
    emit_stage_config_snapshot,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_trace_is_disabled_without_path(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("VLLM_OMNI_CONCURRENCY_TRACE_PATH", raising=False)
    monkeypatch.chdir(tmp_path)

    emit_concurrency_trace("stage_completed", request_id="request-1")

    assert list(tmp_path.iterdir()) == []


def test_trace_records_request_attribution_and_summarizes_metrics(monkeypatch, tmp_path) -> None:
    trace_path = tmp_path / "concurrency.jsonl"
    monkeypatch.setenv("VLLM_OMNI_CONCURRENCY_TRACE_PATH", str(trace_path))
    monkeypatch.setenv("VLLM_OMNI_CONCURRENCY_TRACE_RUN_ID", "run-1")

    emit_stage_config_snapshot(
        {
            "stages": [
                {
                    "stage_id": 0,
                    "devices": "0",
                    "gpu_memory_utilization": 0.7,
                    "max_model_len": 8192,
                    "max_num_seqs": 4,
                },
                {"stage_id": 1, "devices": "1", "max_num_seqs": 2},
            ]
        },
        "test",
    )
    emit_concurrency_trace(
        "stage_completed",
        request_id="request-1",
        stage_id=0,
        batch_size=2,
        stage_gen_time_ms=12.0,
    )
    emit_concurrency_trace("batch_composition_changed", stage_id=0, batch_size=2, scheduled_tokens=24)
    emit_concurrency_trace(
        "stage1_input_ready",
        request_id="request-1",
        batch_index=0,
        batch_size=2,
        has_tts_input=True,
    )
    emit_concurrency_trace("tts_slot_started", request_id="request-1", output_slot=0)
    emit_concurrency_trace("stage1_batch_started", batch_size=2)
    emit_concurrency_trace(
        "tts_slot_completed",
        request_id="request-1",
        output_slot=0,
        outcome="ok",
        waveform_samples=24000,
    )
    emit_concurrency_trace(
        "stage_completed",
        request_id="request-1",
        stage_id=1,
        batch_size=1,
        stage_gen_time_ms=8.0,
    )
    emit_concurrency_trace(
        "stage_postprocess_completed",
        request_id="request-1",
        stage_id=1,
        postprocess_time_ms=2.0,
    )
    emit_concurrency_trace("gpu_sample", gpu_index=0, gpu_utilization_pct=80, memory_used_bytes=1234)
    emit_concurrency_trace("gpu_sample", gpu_index=1, gpu_utilization_pct=60, memory_used_bytes=5678)
    emit_concurrency_trace(
        "prometheus_sample",
        metric="vllm_omni:num_requests_waiting",
        labels={"stage": "1"},
        value=3,
    )
    emit_concurrency_trace("request_completed", request_id="request-1", e2e_total_ms=50.0)

    records = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    summary = build_summary(records)

    assert all(record["run_id"] == "run-1" for record in records)
    assert summary["completed_requests"] == 1
    assert summary["completed_requests_per_s"] == pytest.approx(20.0)
    assert summary["stage_configurations"]["0"]["max_num_seqs"] == 4
    assert summary["stage_configurations"]["0"]["max_model_len"] == 8192
    assert summary["stage_configurations"]["0"]["gpu_memory_utilization"] == 0.7
    assert summary["stages"]["0"]["batch_sizes"] == [2]
    assert summary["stages"]["0"]["batch_composition_sizes"] == [2]
    assert summary["stages"]["0"]["batch_composition_scheduled_tokens"] == [24]
    assert summary["stages"]["0"]["max_observed_batch_size"] == 2
    assert summary["stages"]["1"]["batch_composition_sizes"] == []
    assert summary["stages"]["1"]["max_observed_batch_size"] == 2
    assert summary["stages"]["1"]["devices"] == [1]
    assert summary["stages"]["1"]["gpus"]["1"]["peak_memory_used_bytes"] == 5678
    assert summary["stages"]["1"]["postprocess_latency_ms"]["p50"] == 2.0
    assert summary["tts_output_outcomes"] == {"ok": 1}
    assert summary["gpus"]["0"]["peak_memory_used_bytes"] == 1234
    assert summary["queue"]["vllm_omni:num_requests_waiting|stage=1"]["max"] == 3.0
    assert summary["stage1_queue_delay_ms"]["p50"] >= 0.0
    assert "cross_stage_overlap_ms" in summary
