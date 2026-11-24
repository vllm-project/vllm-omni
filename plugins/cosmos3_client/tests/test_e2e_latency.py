# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import pytest

from vllm_omni_cosmos3_client.e2e_latency import build_e2e_report, percentile


def event(stage, start, end, trace="a"):
    return dict(stage=stage, start_s=start, end_s=end, trace_id=trace, clock_domain="boot")


def record(trace="a", horizon=1, warmup=False):
    return dict(trace_id=trace, execution_window=horizon, batch_size=1, clock_domain="boot", warmup=warmup)


def test_e2e_includes_preprocessing_execution_and_final_response():
    events = [
        event("client.render_observation", 0, 10),
        event("client.preprocess", 10, 11),
        event("client.request", 11, 15),
        event("client.execute_action", 13, 16),
        event("client.wait_final", 16, 17),
    ]
    result = build_e2e_report(events, [record()])
    assert result["p95_ms"] == 7000
    assert result["windows"][0]["wall_ms"] == 17000


def test_render_overlapping_work_is_not_subtracted_twice():
    events = [
        event("client.render_observation", 0, 10),
        event("client.render_observation", 2, 5),
        event("client.request", 6, 12),
        event("client.execute_action", 11, 13),
    ]
    window = build_e2e_report(events, [record()])["windows"][0]
    assert window["render_excluded_ms"] == 6000
    assert window["e2e_no_render_ms"] == 7000


def test_partial_and_warmup_windows_remain_visible_but_do_not_change_p95():
    events = []
    for trace, scale in [("a", 1), ("b", 10), ("c", 100)]:
        events.extend(
            [
                event("client.render_observation", 0, 1, trace),
                event("client.request", 1, 2 * scale, trace),
                event("client.execute_action", 2 * scale, 3 * scale, trace),
            ]
        )
    result = build_e2e_report(events, [record(), record("b", warmup=True), record("c", horizon=16)])
    assert result["p95_ms"] == 2000
    assert result["measured_windows"] == 1
    assert result["partial_windows"] == result["warmup_windows"] == 1
    assert len(result["windows"]) == 3


def test_clock_mismatch_and_missing_events_are_errors():
    rows = [
        event("client.render_observation", 0, 1),
        event("client.request", 1, 2),
        event("client.execute_action", 2, 3),
    ]
    rows[0]["clock_domain"] = "other-host"
    with pytest.raises(ValueError, match="clock domain"):
        build_e2e_report(rows, [record()])
    with pytest.raises(ValueError, match="Missing E2E timeline"):
        build_e2e_report([], [record()])


def test_quantile_is_computed_from_samples():
    assert percentile([0, 10, 20], 95) == 19
    assert percentile([], 95) is None
