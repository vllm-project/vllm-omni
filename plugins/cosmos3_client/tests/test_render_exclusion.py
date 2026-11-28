"""Measured render intervals are excluded without summing concurrent work twice."""

import pytest

from vllm_omni_cosmos3_client.render_exclusion import build_render_free_report, window_metrics


def event(stage, start, end, trace="one"):
    return dict(stage=stage, start_s=start, end_s=end, trace_id=trace)


def test_window_starts_after_render_and_counts_overlap_once():
    rows = [
        event("client.render_observation", 0, 10),
        event("client.request", 10, 14),
        event("client.wait_action", 10, 12),
        event("client.execute_action", 12, 16),
    ]
    metric = window_metrics(rows)
    assert metric["elapsed_no_render_s"] == 6
    assert metric["render_excluded_s"] == 10
    assert metric["executed_actions"] == 1
    assert metric["actions_per_s"] == 1 / 6


def test_duplicate_render_spans_are_subtracted_once():
    rows = [
        event("client.render_observation", 0, 10),
        event("client.render_observation", 2, 5),
        event("client.request", 10, 14),
        event("client.execute_action", 14, 16),
    ]
    record = dict(trace_id="one", batch_size=1, execution_window=1, warmup=False)
    metric = build_render_free_report(rows, [record], (0, 20), 1)
    assert metric["evaluation_no_render"]["elapsed_no_render_s"] == 10
    assert metric["evaluation_no_render"]["render_excluded_s"] == 10
    assert metric["complete_windows_no_render"]["elapsed_no_render_s"] == 6


def test_warmup_and_partial_windows_are_retained_but_not_in_steady_throughput():
    rows = [event("client.request", 2, 4), event("client.execute_action", 4, 5)]
    records = [dict(trace_id="one", batch_size=1, execution_window=16, warmup=False)]
    metric = build_render_free_report(rows, records, (0, 6), 1)
    assert metric["evaluation_no_render"]["actions_per_s"] == 1 / 6
    assert metric["complete_windows_no_render"]["windows"] == 0
    assert metric["complete_windows_no_render"]["actions_per_s"] is None


def test_whole_loop_excludes_dummy_and_render_but_keeps_real_warmup_and_partial_actions():
    rows = [
        event("client.dummy_stabilization", 1, 3, trace=None),
        event("client.render_observation", 3, 5),
        event("client.request", 5, 8),
        event("client.execute_action", 8, 10),
        event("client.dummy_stabilization", 11, 13, trace=None),
        event("client.render_observation", 13, 14),
        event("client.request", 14, 17),
        event("client.execute_action", 17, 19),
    ]
    records = [
        dict(trace_id="one", batch_size=1, execution_window=16, warmup=True),
        dict(trace_id="two", batch_size=1, execution_window=16, warmup=False),
    ]
    rows[5]["trace_id"] = "two"
    rows[6]["trace_id"] = "two"
    rows[7]["trace_id"] = "two"
    report = build_render_free_report(rows, records, (0, 20), 2, exclude_dummy_time=True, expected_dummy_intervals=2)
    assert report["evaluation_no_render"]["elapsed_no_render_s"] == 17
    clean = report["evaluation_no_render_no_dummy"]
    assert clean["render_excluded_s"] == 3
    assert clean["dummy_stabilization_s"] == 4
    assert clean["total_excluded_s"] == 7
    assert clean["elapsed_no_render_no_dummy_s"] == 13
    assert clean["executed_actions"] == 2
    assert clean["actions_per_s"] == 2 / 13
    assert report["complete_windows_no_render"]["windows"] == 0


def test_intersecting_render_and_dummy_are_excluded_once():
    rows = [event("client.render_observation", 1, 5), event("client.dummy_stabilization", 3, 7, trace=None)]
    result = build_render_free_report(rows, [], (0, 10), 2, exclude_dummy_time=True, expected_dummy_intervals=1)
    clean = result["evaluation_no_render_no_dummy"]
    assert clean["render_excluded_s"] == 4
    assert clean["dummy_stabilization_s"] == 4
    assert clean["total_excluded_s"] == 6
    assert clean["elapsed_no_render_no_dummy_s"] == 4


def test_refuse_no_dummy_metric_when_episode_stabilization_probe_is_missing():
    with pytest.raises(ValueError, match="Expected 2 dummy stabilization intervals, got 1"):
        build_render_free_report(
            [event("client.dummy_stabilization", 0, 1, trace=None)],
            [],
            (0, 10),
            0,
            exclude_dummy_time=True,
            expected_dummy_intervals=2,
        )
