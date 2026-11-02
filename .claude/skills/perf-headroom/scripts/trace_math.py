# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Device interval accounting and explicitly constrained DAG sensitivity, CPU only."""

import math
from collections import Counter, defaultdict
from graphlib import TopologicalSorter


def number(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name}: expected number")
    if not math.isfinite(value) or value < minimum:
        raise ValueError(f"{name}: invalid value {value}")
    return value


def union(intervals):
    merged = []
    for start, end in sorted(intervals):
        if not math.isfinite(start) or not math.isfinite(end) or end < start:
            raise ValueError("Invalid interval")
        if end == start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([start, end])
    return merged


def ledger(events, start, end):
    """events: (start_ms, end_ms, category), all from ONE device/timebase."""
    if not math.isfinite(start) or not math.isfinite(end) or end <= start:
        raise ValueError("Invalid accounting window")
    edges = defaultdict(Counter)
    spans = []
    for a, b, category in events:
        if not math.isfinite(a) or not math.isfinite(b) or b < a:
            raise ValueError("Invalid device event")
        a, b = max(a, start), min(b, end)
        if b <= a:
            continue
        spans.append((a, b))
        edges[a][category] += 1
        edges[b][category] -= 1
    merged = union(spans)
    totals, active, previous = Counter(), Counter(), start
    for timestamp in sorted(edges):
        if active:
            category = next(iter(active)) if len(active) == 1 else "cross_category_overlap"
            totals[category] += timestamp - previous
        active.update(edges[timestamp])
        active = +active
        previous = timestamp
    busy = math.fsum(b - a for a, b in merged)
    summed = math.fsum(b - a for a, b in spans)
    if merged:
        head, tail = merged[0][0] - start, end - merged[-1][1]
        span = merged[-1][1] - merged[0][0]
        internal = span - busy
    else:
        head, tail, span, internal = end - start, 0, 0, 0
    totals.update(request_head=head, request_tail=tail, internal_gap=internal)
    return {
        "window_ms": end - start,
        "summed_event_ms": summed,
        "busy_union_ms": busy,
        # Duplicate-counted event duration, not elapsed overlap or removable time.
        "overlap_excess_ms": max(0.0, summed - busy),
        "device_span_ms": span,
        "internal_gap_ms": internal,
        "internal_gap_pct": 100 * internal / span if span else None,
        "exclusive_ledger_ms": dict(totals),
        "has_device_events": bool(merged),
        "note": (
            "Event sum includes overlapping kernel/copy/set work. "
            "No-event time is observed, not proven removable; not unprofiled E2E."
        ),
    }


def trace_ledger(document, device, start_us=None, end_us=None, rules=()):
    events = document["traceEvents"] if isinstance(document, dict) else document
    selected = [
        e
        for e in events
        if e.get("cat") in {"kernel", "gpu_memcpy", "gpu_memset"}
        and e.get("ph") == "X"
        and str(e.get("args", {}).get("device")) == str(device)
    ]
    if not selected:
        raise ValueError("No device events for the requested device")
    for e in selected:
        number(e["dur"], "duration")
        if not math.isfinite(e["ts"]):
            raise ValueError("Invalid event timestamp")
    start = min(e["ts"] for e in selected) if start_us is None else start_us
    end = max(e["ts"] + e["dur"] for e in selected) if end_us is None else end_us
    classified = []
    for e in selected:
        matches = [r["label"] for r in rules if any(s in e["name"] for s in r["contains"])]
        label = matches[0] if len(matches) == 1 else "unattributed"
        classified.append((e["ts"] / 1000, (e["ts"] + e["dur"]) / 1000, label))
    result = ledger(classified, start / 1000, end / 1000)
    result.update(
        device=str(device),
        start_us=start,
        end_us=end,
        window_kind="explicit" if start_us is not None and end_us is not None else "device_event_span",
    )
    return result


def replay(graph):
    """All resource serialization/order must be explicit dependency edges."""
    if graph.get("resource_order_confirmed") is not True:
        raise ValueError("Replay requires explicit, checked resource-order edges")
    nodes = {n["id"]: n for n in graph["nodes"]}
    if not nodes or len(nodes) != len(graph["nodes"]):
        raise ValueError("Empty graph or duplicate node id")
    for node in nodes.values():
        number(node["ms"], "node duration")
        if any(dep not in nodes for dep in node["deps"]):
            raise ValueError("Missing dependency node")
    order = list(TopologicalSorter({k: n["deps"] for k, n in nodes.items()}).static_order())
    ends, paths = {}, {}
    for key in order:
        deps = nodes[key]["deps"]
        parent = max(deps, key=ends.__getitem__) if deps else None
        ends[key] = (ends[parent] if parent is not None else 0) + nodes[key]["ms"]
        paths[key] = (paths[parent] if parent is not None else []) + [key]
    # End at completion, including zero-duration joins that tie their producer.
    has_successor = {dep for node in nodes.values() for dep in node["deps"]}
    last = max((key for key in order if key not in has_successor), key=ends.__getitem__)
    return {
        "predicted_ms": ends[last],
        "critical_path": paths[last],
        "model": "fixed_resource_order_DAG",
        "hard_lower_bound": False,
        "limitations": "No inferred dependencies, new contention, or cache-locality model.",
    }
