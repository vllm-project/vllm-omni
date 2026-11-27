"""Throughput from shared-clock intervals, excluding camera rendering once."""

from collections import defaultdict


def union(intervals):
    result = []
    for start, end in sorted(intervals):
        if end < start:
            raise ValueError("Negative timing interval")
        if result and start <= result[-1][1]:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([start, end])
    return result


def seconds(intervals):
    return sum(end - start for start, end in union(intervals))


def intersection(left, right):
    return seconds((max(a, c), min(b, d)) for a, b in union(left) for c, d in union(right) if max(a, c) < min(b, d))


def window_metrics(events):
    rendering = [(e["start_s"], e["end_s"]) for e in events if e["stage"] == "client.render_observation"]
    active = [e for e in events if e["stage"] != "client.render_observation"]
    if not active:
        return None
    start, end = min(e["start_s"] for e in active), max(e["end_s"] for e in active)
    render_inside = intersection([(start, end)], rendering)
    execute = [(e["start_s"], e["end_s"]) for e in active if e["stage"] == "client.execute_action"]
    count = sum(len(e.get("env_slots", [0])) for e in active if e["stage"] == "client.execute_action")
    elapsed = end - start - render_inside
    return {
        "start_s": start,
        "end_s": end,
        "elapsed_no_render_s": elapsed,
        "render_excluded_s": seconds(rendering),
        "executed_actions": count,
        "client_execute_s": seconds(execute),
        "actions_per_s": count / elapsed if elapsed > 0 else None,
    }


def build_render_free_report(
    events,
    records,
    evaluation_interval,
    executed_actions,
    *,
    exclude_dummy_time=False,
    expected_dummy_intervals=None,
):
    groups = defaultdict(list)
    rendering = []
    stabilization = []
    for event in events:
        if event.get("trace_id"):
            groups[event["trace_id"]].append(event)
        if event["stage"] == "client.render_observation":
            rendering.append((event["start_s"], event["end_s"]))
        elif event["stage"] == "client.dummy_stabilization":
            stabilization.append((event["start_s"], event["end_s"]))
    elapsed = evaluation_interval[1] - evaluation_interval[0]
    excluded = intersection([evaluation_interval], rendering)
    effective = elapsed - excluded
    windows = []
    for request in records:
        metric = window_metrics(groups.get(request.get("trace_id"), []))
        if metric is None:
            continue
        expected = request["batch_size"] * (request.get("execution_window") or 16)
        if not request.get("warmup") and metric["executed_actions"] == expected:
            windows.append(metric)
    action_count = sum(w["executed_actions"] for w in windows)
    window_time = sum(w["elapsed_no_render_s"] for w in windows)
    report = {
        "scope": "Camera and complete render RPC intervals are excluded using interval unions; concurrent server/client work is not summed twice",
        "evaluation_no_render": {
            "elapsed_no_render_s": effective,
            "render_excluded_s": excluded,
            "executed_actions": executed_actions,
            "actions_per_s": executed_actions / effective if effective > 0 else None,
        },
        "complete_windows_no_render": {
            "windows": len(windows),
            "executed_actions": action_count,
            "elapsed_no_render_s": window_time,
            "actions_per_s": action_count / window_time if window_time > 0 else None,
        },
    }
    if exclude_dummy_time:
        if expected_dummy_intervals is not None and len(stabilization) != expected_dummy_intervals:
            raise ValueError(
                f"Expected {expected_dummy_intervals} dummy stabilization intervals, got {len(stabilization)}"
            )
        excluded_both = intersection([evaluation_interval], rendering + stabilization)
        elapsed_no_dummy = elapsed - excluded_both
        report["evaluation_no_render_no_dummy"] = {
            "scope": "Whole LIBERO task loop; exclude the union of render RPC and measured dummy stabilization spans",
            "elapsed_no_render_no_dummy_s": elapsed_no_dummy,
            "render_excluded_s": excluded,
            "dummy_stabilization_s": intersection([evaluation_interval], stabilization),
            "total_excluded_s": excluded_both,
            "timed_dummy_intervals": len(stabilization),
            "executed_actions": executed_actions,
            "actions_per_s": executed_actions / elapsed_no_dummy if elapsed_no_dummy > 0 else None,
        }
    return report
