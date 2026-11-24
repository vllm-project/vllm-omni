# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Complete control-window latency, measured on the client monotonic clock."""

import math
from collections import defaultdict

from .render_exclusion import intersection

SCHEMA = "cosmos3_e2e_no_render_v1"


def percentile(values, q):
    """Linear interpolation, equivalent to NumPy's default percentile method."""
    values = sorted(values)
    if not values:
        return None
    position = (len(values) - 1) * q / 100
    lo, hi = math.floor(position), math.ceil(position)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def build_e2e_report(events, records):
    grouped = defaultdict(list)
    for event in events:
        if event.get("trace_id"):
            grouped[event["trace_id"]].append(event)
    windows, seen = [], set()
    for record in records:
        trace = record.get("trace_id")
        if not trace or trace in seen:
            raise ValueError("E2E requires a unique trace_id for every request")
        seen.add(trace)
        rows = grouped.get(trace, [])
        if not rows:
            raise ValueError(f"Missing E2E timeline for {trace}")
        domain = record.get("clock_domain")
        if not domain or any(row.get("clock_domain") != domain for row in rows):
            raise ValueError(f"E2E monotonic clock domain mismatch for {trace}")
        for row in rows:
            if not (math.isfinite(row["start_s"]) and math.isfinite(row["end_s"])) or row["end_s"] < row["start_s"]:
                raise ValueError(f"Invalid E2E timing interval for {trace}")
            if row.get("outcome", "complete") != "complete":
                raise ValueError(f"Failed E2E interval for {trace}")
        render = [(r["start_s"], r["end_s"]) for r in rows if r["stage"] == "client.render_observation"]
        requests = [r for r in rows if r["stage"] == "client.request"]
        actions = [r for r in rows if r["stage"] == "client.execute_action"]
        if not render or not requests or not actions:
            raise ValueError(f"Incomplete E2E probes for {trace}")
        # Final-response collection and execution both belong to this window.
        start = min(r["start_s"] for r in rows)
        end = max(r["end_s"] for r in rows)
        # Keep work hidden behind camera RPCs: only exclusively rendering time
        # is removed. Never subtract all rendering from an overlapping window.
        work = [
            (r["start_s"], r["end_s"])
            for r in rows
            if r["stage"] in {"client.request", "client.preprocess", "client.execute_action"}
        ]
        render_s = intersection([(start, end)], render)
        overlap_s = intersection(render, work)
        excluded_s = render_s - overlap_s
        elapsed_s = end - start - excluded_s
        if elapsed_s <= 0:
            raise ValueError(f"Nonpositive E2E duration for {trace}")
        batch_size = int(record["batch_size"])
        horizon = int(record["execution_window"])
        if batch_size <= 0 or horizon <= 0:
            raise ValueError("E2E requires positive batch_size and execution_window")
        executed = sum(len(r.get("env_slots", [0])) for r in actions)
        expected = batch_size * horizon
        if executed > expected:
            raise ValueError(f"E2E action count exceeds execution window for {trace}")
        windows.append(
            dict(
                trace_id=trace,
                task_id=record.get("task_id"),
                episode=record.get("episode"),
                warmup=bool(record.get("warmup", False)),
                complete=executed == expected,
                executed_actions=executed,
                expected_actions=expected,
                wall_ms=(end - start) * 1000,
                render_ms=render_s * 1000,
                render_overlapped_ms=overlap_s * 1000,
                render_excluded_ms=excluded_s * 1000,
                e2e_no_render_ms=elapsed_s * 1000,
            )
        )
    measured = [w["e2e_no_render_ms"] for w in windows if w["complete"] and not w["warmup"]]
    return dict(
        schema=SCHEMA,
        scope="observation acquisition start to both window action execution and final response completion; remove exclusive rendering intervals only",
        percentile_method="linear",
        unit="ms",
        measured_windows=len(measured),
        warmup_windows=sum(w["warmup"] for w in windows),
        partial_windows=sum(not w["complete"] for w in windows),
        p50_ms=percentile(measured, 50),
        p95_ms=percentile(measured, 95),
        p99_ms=percentile(measured, 99),
        windows=windows,
    )
