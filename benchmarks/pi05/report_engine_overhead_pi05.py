# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Aggregate a π0.5 engine-overhead result tree into before/after tables.

Expected layout (written by ``run_engine_overhead_matrix.sh``)::

    <root>/<variant>/<dtype>/standalone_run<N>.json
    <root>/<variant>/<dtype>/engine_<uni|mp>_run<N>.json + trace_engine_<uni|mp>_run<N>/
    <root>/<variant>/<dtype>/e2e/{client.jsonl,trace/}
    <root>/<variant>/<dtype>/e2e_notrace/client.jsonl        (optional)
    <root>/<variant>/<dtype>/*.npz                            (action chunks)

``variant`` is ``before`` (the reference tree) or ``after`` (the tree under
test); one of them is enough for a profile. Every latency below is warm; the
first request of a server or engine process is reported separately as cold.

Trace rows are the ``trace-<pid>.tsv`` files of ``vllm_omni.utils.latency_trace``
(one per process) and, for OpenPI E2E, the ``--requests-out`` rows of
``e2e_latency_pi05.py``. All timestamps are ``CLOCK_MONOTONIC`` nanoseconds
from one host, so they share one timeline.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

VARIANTS = ("before", "after")

# Causal order of marks along one BS=1 request. A segment is the time between
# two consecutive marks that are both present, so segments always sum to the
# span between the first and last mark of the request.
CHAIN = tuple(
    """
    cli_start cli_sent ws_received ws_unpacked serving_request_built engine_generate_enter engine_enqueued
    orch_dequeued orch_state_ready orch_stage_submitted client_dispatch_start diffusion_preprocessed
    diffusion_scheduler_enqueued busy_scheduled worker_msg_received worker_execute_start runner_start
    pipeline_forward_start model_start model_end pipeline_forward_end runner_end worker_execute_end busy_executed
    busy_emitted diffusion_stream_item diffusion_output_dequeued diffusion_postprocessed client_output_queued
    orch_output_polled orch_output_dispatched orch_output_routed frontend_output_wakeup engine_output_received
    engine_result_processed engine_generate_yield serving_engine_done serving_infer_done ws_reply_packed
    ws_reply_sent cli_received cli_done
    """.split()
)
# Layer boundaries.
LAYERS = {
    "model": ("model_start", "model_end"),
    "pipeline": ("pipeline_forward_start", "pipeline_forward_end"),
    "engine": ("engine_generate_enter", "engine_generate_yield"),
    "e2e": ("cli_start", "cli_done"),
}

_SERVER_ID = re.compile(r"^robot-(?P<session>.+)-(?P<n>\d+)$")


# Engine-path segments, in request order. Each entry sums the listed
# consecutive-mark segments; marks that a backend does not emit are skipped by
# ``group_ms`` and their time lands in the neighbouring pair.
GROUPS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("client", "client pack + send", ("cli_start", "cli_sent")),
    ("server", "socket in + websocket receive", ("cli_sent", "ws_received")),
    ("server", "request unpack (msgpack)", ("ws_received", "ws_unpacked")),
    ("server", "serving request build", ("ws_unpacked", "engine_generate_enter")),
    ("engine", "request enqueue (input processing + queue put)", ("engine_generate_enter", "engine_enqueued")),
    ("engine", "queue wait (frontend -> orchestrator)", ("engine_enqueued", "orch_dequeued")),
    ("engine", "orchestrator construction", ("orch_dequeued", "orch_state_ready")),
    ("engine", "stage submission", ("orch_state_ready", "client_dispatch_start")),
    ("engine", "diffusion engine admission", ("client_dispatch_start", "diffusion_scheduler_enqueued")),
    ("engine", "scheduler wait", ("diffusion_scheduler_enqueued", "busy_scheduled")),
    ("engine", "executor dispatch (to worker)", ("busy_scheduled", "worker_execute_start")),
    ("engine", "runner setup", ("worker_execute_start", "pipeline_forward_start")),
    ("pipeline", "pipeline: pre-model (inputs)", ("pipeline_forward_start", "model_start")),
    ("pipeline", "pipeline: model", ("model_start", "model_end")),
    ("pipeline", "pipeline: post-model (outputs)", ("model_end", "pipeline_forward_end")),
    ("engine", "runner teardown", ("pipeline_forward_end", "worker_execute_end")),
    ("engine", "executor return (from worker)", ("worker_execute_end", "busy_executed")),
    ("engine", "scheduler update + emit", ("busy_executed", "diffusion_stream_item")),
    ("engine", "output-ready wait", ("diffusion_stream_item", "diffusion_output_dequeued")),
    ("engine", "diffusion postprocess", ("diffusion_output_dequeued", "client_output_queued")),
    ("engine", "output pickup (poll / wake-up)", ("client_output_queued", "orch_output_polled")),
    ("engine", "output routing", ("orch_output_polled", "orch_output_routed")),
    ("engine", "frontend handoff", ("orch_output_routed", "engine_output_received")),
    ("engine", "response handling", ("engine_output_received", "engine_generate_yield")),
    ("server", "serving reply build + pack", ("engine_generate_yield", "ws_reply_packed")),
    ("server", "websocket send + socket out", ("ws_reply_packed", "cli_received")),
    ("client", "client decode", ("cli_received", "cli_done")),
)


def load_marks(trace_dir: str) -> dict[str, dict[str, int]]:
    """Return ``{request_id: {mark: t_ns}}`` with internal ids folded into external ids."""
    marks: dict[str, dict[str, int]] = defaultdict(dict)
    aliases: dict[str, str] = {}
    for path in sorted(glob.glob(os.path.join(trace_dir, "trace-*.tsv"))):
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                parts = line.rstrip("\n").split("\t")
                if len(parts) != 3:
                    continue
                rid, name, t_ns = parts[0], parts[1], int(parts[2])
                if name.startswith("@alias:"):
                    aliases[name[len("@alias:") :]] = rid
                    continue
                # Keep the first occurrence: a mark can repeat only on retries.
                marks[rid].setdefault(name, t_ns)
    merged: dict[str, dict[str, int]] = defaultdict(dict)
    for rid, row in marks.items():
        target = aliases.get(rid, rid)
        for name, t_ns in row.items():
            merged[target].setdefault(name, t_ns)
    return dict(merged)


def load_client_rows(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def join_client(marks: dict[str, dict[str, int]], client_rows: list[dict]) -> list[dict]:
    """Attach client timestamps to the server request served for each client row.

    The server names requests ``robot-<session>-<n>`` with ``n`` increasing in
    arrival order, and the BS=1 client sends one session serially, so the i-th
    client row of a session is the i-th server request of that session.
    """
    by_session: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for rid in marks:
        match = _SERVER_ID.match(rid)
        if match:
            by_session[match.group("session")].append((int(match.group("n")), rid))
    for items in by_session.values():
        items.sort()
    joined = []
    for row in client_rows:
        server_ids = by_session.get(row["session_id"], [])
        if row["index"] >= len(server_ids):
            raise RuntimeError(f"No server trace for session {row['session_id']} request {row['index']}")
        rid = server_ids[row["index"]][1]
        timeline = dict(marks[rid])
        for key in ("cli_start", "cli_sent", "cli_received", "cli_done"):
            timeline[key] = row[key]
        joined.append(
            {**{k: row[k] for k in ("views", "round", "index", "session_id")}, "request_id": rid, "t": timeline}
        )
    return joined


def layer_ms(timeline: dict[str, int], layer: str) -> float | None:
    start, end = LAYERS[layer]
    if start not in timeline or end not in timeline:
        return None
    return (timeline[end] - timeline[start]) / 1e6


def quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = q * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def group_ms(timeline: dict[str, int], start: str, end: str) -> float | None:
    """Time between ``start`` and ``end``, widening to the nearest chain marks present."""
    chain = CHAIN
    i, j = chain.index(start), chain.index(end)
    while i >= 0 and chain[i] not in timeline:
        i -= 1
    while j < len(chain) and chain[j] not in timeline:
        j += 1
    if i < 0 or j >= len(chain) or i >= j:
        return None
    return (timeline[chain[j]] - timeline[chain[i]]) / 1e6


def request_metrics(timeline: dict[str, int]) -> dict[str, float]:
    out: dict[str, float] = {}
    for _, label, (start, end) in GROUPS:
        if start in timeline or end in timeline:
            # Skip groups that lie entirely outside this run's marks (e.g. websocket groups in engine mode).
            present = [name for name in CHAIN if name in timeline]
            if CHAIN.index(end) <= CHAIN.index(present[0]) or CHAIN.index(start) >= CHAIN.index(present[-1]):
                continue
            value = group_ms(timeline, start, end)
            if value is not None:
                out[label] = value
    for layer in LAYERS:
        value = layer_ms(timeline, layer)
        if value is not None:
            out[f"layer:{layer}"] = value
    if "layer:engine" in out and "layer:pipeline" in out:
        out["engine - pipeline"] = out["layer:engine"] - out["layer:pipeline"]
    if "layer:pipeline" in out and "layer:model" in out:
        out["pipeline - model"] = out["layer:pipeline"] - out["layer:model"]
    if "layer:e2e" in out and "layer:engine" in out:
        out["e2e - engine"] = out["layer:e2e"] - out["layer:engine"]
    if "layer:e2e" in out and "layer:model" in out:
        out["e2e - model"] = out["layer:e2e"] - out["layer:model"]
    return out


def load_engine_run(result_path: str) -> dict:
    with open(result_path, encoding="utf-8") as handle:
        result = json.load(handle)
    trace_dir = os.path.join(os.path.dirname(result_path), "trace_" + os.path.basename(result_path)[: -len(".json")])
    marks = load_marks(trace_dir) if os.path.isdir(trace_dir) else {}
    rows = []
    for request in result.get("requests", []):
        timeline = marks.get(request["request_id"])
        rows.append(
            {
                "views": request["views"],
                "wall_ms": request["latency_ms"],
                "metrics": request_metrics(timeline) if timeline else {},
            }
        )
    return {"result": result, "rows": rows}


def load_e2e(directory: str) -> list[dict]:
    client_rows = load_client_rows(os.path.join(directory, "client.jsonl"))
    trace_dir = os.path.join(directory, "trace")
    if os.path.isdir(trace_dir) and glob.glob(os.path.join(trace_dir, "trace-*.tsv")):
        joined = join_client(load_marks(trace_dir), client_rows)
    else:
        joined = [
            {
                **{k: r[k] for k in ("views", "round", "index", "session_id")},
                "t": {k: r[k] for k in ("cli_start", "cli_done")},
            }
            for r in client_rows
        ]
    warmup = {(row["session_id"], row["index"]) for row in client_rows if row["warmup"]}
    first = min(joined, key=lambda row: row["t"]["cli_start"])
    rows = []
    for row in joined:
        if row is not first and (row["session_id"], row["index"]) in warmup:
            continue
        rows.append(
            {
                "views": row["views"],
                "round": row["round"],
                "cold": row is first,
                "wall_ms": (row["t"]["cli_done"] - row["t"]["cli_start"]) / 1e6,
                "metrics": request_metrics(row["t"]),
            }
        )
    return rows


def collect(root: str) -> dict:
    data: dict = {}
    for variant in VARIANTS:
        for dtype_dir in sorted(glob.glob(os.path.join(root, variant, "*"))):
            if not os.path.isdir(dtype_dir):
                continue
            dtype = os.path.basename(dtype_dir)
            entry = data.setdefault(dtype, {}).setdefault(variant, {"standalone": [], "engine": {"uni": [], "mp": []}})
            for path in sorted(glob.glob(os.path.join(dtype_dir, "standalone_run*.json"))):
                with open(path, encoding="utf-8") as handle:
                    entry["standalone"].append(json.load(handle))
            for backend in ("uni", "mp"):
                for path in sorted(glob.glob(os.path.join(dtype_dir, f"engine_{backend}_run*.json"))):
                    entry["engine"][backend].append(load_engine_run(path))
            for name in ("e2e", "e2e_notrace"):
                directory = os.path.join(dtype_dir, name)
                if os.path.exists(os.path.join(directory, "client.jsonl")):
                    entry[name] = load_e2e(directory)
            entry["npz"] = sorted(glob.glob(os.path.join(dtype_dir, "*.npz")))
    return data


def runs_p50(per_run_values: list[list[float]]) -> dict:
    """Median of per-run p50 and its run-to-run range."""
    medians = [quantile(values, 0.5) for values in per_run_values if values]
    pooled = [value for values in per_run_values for value in values]
    if not medians:
        return {"n": 0}
    return {
        "runs": len(medians),
        "p50": quantile(medians, 0.5),
        "run_min": min(medians),
        "run_max": max(medians),
        "pooled_p50": quantile(pooled, 0.5),
        "pooled_p95": quantile(pooled, 0.95),
        "n": len(pooled),
    }


def fmt(value: float | None, digits: int = 1) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def cell(summary: dict, digits: int = 1) -> str:
    if not summary.get("n"):
        return "n/a"
    return f"{summary['p50']:.{digits}f} [{summary['run_min']:.{digits}f}-{summary['run_max']:.{digits}f}]"


def layer_table(entry: dict, views_list: list[int]) -> tuple[list[str], dict]:
    lines = [
        "| Views | Model-only | In-process pipeline | `uni` engine | `mp` engine | OpenPI E2E p50 / p95 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    summary: dict = {}
    for views in views_list:
        key = str(views)
        model = runs_p50([[run["views"][key]["layers"]["model_only_ms"]["p50_ms"]] for run in entry["standalone"]])
        pipeline = runs_p50([[run["views"][key]["layers"]["pipeline_ms"]["p50_ms"]] for run in entry["standalone"]])
        engines = {}
        for backend in ("uni", "mp"):
            engines[backend] = runs_p50(
                [[row["wall_ms"] for row in run["rows"] if row["views"] == views] for run in entry["engine"][backend]]
            )
        e2e_rows = [row for row in entry.get("e2e", []) if row["views"] == views and not row["cold"]]
        rounds = sorted({row["round"] for row in e2e_rows})
        e2e = runs_p50([[row["wall_ms"] for row in e2e_rows if row["round"] == r] for r in rounds])
        summary[key] = {
            "model_only": model,
            "pipeline": pipeline,
            "engine_uni": engines["uni"],
            "engine_mp": engines["mp"],
            "e2e": e2e,
        }
        e2e_cell = "n/a" if not e2e.get("n") else f"{e2e['pooled_p50']:.1f} / {e2e['pooled_p95']:.1f}"
        lines.append(
            f"| {views} | {cell(model)} | {cell(pipeline)} | {cell(engines['uni'])} "
            f"| {cell(engines['mp'])} | {e2e_cell} |"
        )
    return lines, summary


def pooled_metric(rows_by_run: list[list[dict]], name: str) -> dict:
    return runs_p50([[row["metrics"][name] for row in rows if name in row["metrics"]] for rows in rows_by_run])


def path_rows(entry: dict, path: str) -> list[list[dict]]:
    """Warm requests of one path, one list per independent run / client round."""
    if path == "e2e":
        rows = [row for row in entry.get("e2e", []) if not row["cold"]]
        return [[row for row in rows if row["round"] == r] for r in sorted({row["round"] for row in rows})]
    return [run["rows"] for run in entry["engine"][path]]


def verdict(before: dict, after: dict) -> str:
    if not before.get("n") or not after.get("n"):
        return ""
    if after["run_max"] < before["run_min"]:
        return "lower"
    if after["run_min"] > before["run_max"]:
        return "higher"
    return "within run-to-run"


def comparison_table(
    dtype_data: dict, path: str, names: list[tuple[str, str]], digits: int = 3
) -> tuple[list[str], dict]:
    lines = [
        "| Segment | Before p50 [runs] | Before p95 | After p50 [runs] | After p95 | Δ p50 | Verdict |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    out: dict = {}
    before_rows = (
        path_rows(dtype_data.get("before", {"engine": {"uni": [], "mp": []}}), path) if "before" in dtype_data else []
    )
    after_rows = (
        path_rows(dtype_data.get("after", {"engine": {"uni": [], "mp": []}}), path) if "after" in dtype_data else []
    )
    for label, name in names:
        before = pooled_metric(before_rows, name)
        after = pooled_metric(after_rows, name)
        if not before.get("n") and not after.get("n"):
            continue
        delta = (after["p50"] - before["p50"]) if before.get("n") and after.get("n") else None
        out[name] = {"before": before, "after": after, "delta_p50": delta, "verdict": verdict(before, after)}
        lines.append(
            f"| {label} | {cell(before, digits)} | {fmt(before.get('pooled_p95'), digits)} | {cell(after, digits)} | "
            f"{fmt(after.get('pooled_p95'), digits)} | {'n/a' if delta is None else f'{delta:+.{digits}f}'} "
            f"| {verdict(before, after)} |"
        )
    return lines, out


def correctness(dtype_data: dict) -> tuple[list[str], dict]:
    """Bit-exact comparison of every saved action chunk, before vs after and across layers."""
    arrays: dict[str, dict[str, np.ndarray]] = {}
    for variant in VARIANTS:
        merged: dict[str, np.ndarray] = {}
        consistent = True
        for path in dtype_data.get(variant, {}).get("npz", []):
            with np.load(path) as handle:
                for key in handle.files:
                    name = f"{os.path.basename(path).split('_run')[0].split('_round')[0]}:{key}"
                    value = handle[key]
                    if name in merged and not np.array_equal(merged[name], value):
                        consistent = False
                    merged.setdefault(name, value)
        arrays[variant] = merged
        arrays[variant + "_consistent"] = consistent  # type: ignore[assignment]
    lines = []
    result: dict = {}
    for variant in VARIANTS:
        if arrays.get(variant):
            result[f"{variant}_repeat_runs_identical"] = bool(arrays[variant + "_consistent"])
            lines.append(
                f"- `{variant}`: repeated runs/rounds return identical action chunks: "
                f"**{bool(arrays[variant + '_consistent'])}**"
            )
    if arrays.get("before") and arrays.get("after"):
        shared = sorted(set(arrays["before"]) & set(arrays["after"]))
        mismatched = [name for name in shared if not np.array_equal(arrays["before"][name], arrays["after"][name])]
        max_abs = max((float(np.abs(arrays["before"][n] - arrays["after"][n]).max()) for n in shared), default=0.0)
        result.update({"compared": len(shared), "mismatched": mismatched, "max_abs_diff": max_abs})
        lines.append(
            f"- before vs after: {len(shared)} action chunks compared, "
            f"**{len(shared) - len(mismatched)} bit-identical**, "
            f"max |Δ| = {max_abs:.3g}"
        )
    for variant in VARIANTS:
        merged = arrays.get(variant) or {}
        by_views: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
        for name, value in merged.items():
            source, key = name.split(":", 1)
            if source.startswith("e2e"):
                continue  # the websocket client draws a new observation per request
            by_views[key[-1]][name] = value
        same = all(
            np.array_equal(value, next(iter(group.values()))) for group in by_views.values() for value in group.values()
        )
        if by_views:
            result[f"{variant}_layers_identical"] = bool(same)
            lines.append(
                f"- `{variant}`: model-only, pipeline, `uni` and `mp` engine outputs identical per view count: "
                f"**{bool(same)}**"
            )
    return lines, result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", help="Result tree of one GPU.")
    parser.add_argument("--title", default=None)
    parser.add_argument("--out-md")
    parser.add_argument("--out-json")
    args = parser.parse_args()

    data = collect(args.root)
    report: dict = {}
    lines: list[str] = [f"## {args.title or os.path.basename(os.path.normpath(args.root))}", ""]
    for dtype, dtype_data in data.items():
        report[dtype] = {}
        any_entry = dtype_data.get("after") or dtype_data.get("before")
        views_list = sorted({int(v) for run in any_entry["standalone"] for v in run["views"]}) or [1, 2, 3]
        environment = (any_entry["standalone"] or [{}])[0].get("environment", {})
        lines += [f"### {dtype}", ""]
        if environment:
            lines += [
                f"GPU `{environment.get('gpu')}` ({environment.get('machine')}), torch {environment.get('torch')}, "
                f"vLLM {environment.get('vllm')}, transformers {environment.get('transformers')}, "
                f"Python {environment.get('python')}.",
                "",
            ]
        for variant in VARIANTS:
            if variant not in dtype_data:
                continue
            table, summary = layer_table(dtype_data[variant], views_list)
            report[dtype][f"layers_{variant}"] = summary
            lines += [
                f"**Layers, `{variant}`** — warm p50 in ms, median of the independent runs `[min-max across runs]`.",
                "",
            ]
            lines += table + [""]
            cold = [row["wall_ms"] for row in dtype_data[variant].get("e2e", []) if row["cold"]]
            firsts = [
                run["result"]["views"][str(views_list[0])]["first_request_ms"]
                for backend in ("uni", "mp")
                for run in dtype_data[variant]["engine"][backend]
            ]
            lines += [
                f"Cold first request: OpenPI E2E {fmt(cold[0] if cold else None)} ms; engine processes "
                f"{fmt(min(firsts) if firsts else None)}-{fmt(max(firsts) if firsts else None)} ms.",
                "",
            ]
        nested = [
            ("model (same requests)", "layer:model"),
            ("pipeline (same requests)", "layer:pipeline"),
            ("engine (same requests)", "layer:engine"),
            ("OpenPI E2E (same requests)", "layer:e2e"),
            ("**engine - pipeline**", "engine - pipeline"),
            ("pipeline - model", "pipeline - model"),
            ("E2E - engine", "e2e - engine"),
            ("**E2E - model (all non-model time)**", "e2e - model"),
        ]
        segment_names = [(f"{label}", label) for _, label, _ in GROUPS]
        for path, title in (
            ("e2e", "OpenPI E2E (`vllm serve`, `mp` executor)"),
            ("mp", "`mp` engine (`Omni.generate`)"),
            ("uni", "`uni` engine (`Omni.generate`)"),
        ):
            table, out = comparison_table(dtype_data, path, nested)
            if len(table) > 2:
                report[dtype][f"nested_{path}"] = out
                lines += (
                    [f"**Per-request layers, {title}** — ms, each row from the same traced requests.", ""]
                    + table
                    + [""]
                )
            table, out = comparison_table(dtype_data, path, segment_names)
            if len(table) > 2:
                report[dtype][f"segments_{path}"] = out
                lines += [f"**Segments, {title}** — ms.", ""] + table + [""]
        overhead = {}
        for variant in VARIANTS:
            entry = dtype_data.get(variant, {})
            if "e2e_notrace" in entry and "e2e" in entry:
                traced = [row["wall_ms"] for row in entry["e2e"] if not row["cold"]]
                plain = [row["wall_ms"] for row in entry["e2e_notrace"] if not row["cold"]]
                overhead[variant] = {
                    name: {"n": len(v), "p50": quantile(v, 0.5), "p95": quantile(v, 0.95)}
                    for name, v in (("traced", traced), ("untraced", plain))
                }
                lines += [
                    f"Trace overhead check (`{variant}`): OpenPI E2E p50 {quantile(plain, 0.5):.2f} ms untraced vs "
                    f"{quantile(traced, 0.5):.2f} ms traced "
                    f"(p95 {quantile(plain, 0.95):.2f} vs {quantile(traced, 0.95):.2f}).",
                    "",
                ]
        report[dtype]["trace_overhead"] = overhead
        correctness_lines, correctness_result = correctness(dtype_data)
        report[dtype]["correctness"] = correctness_result
        lines += ["**Output check**", ""] + correctness_lines + [""]

    text = "\n".join(lines)
    if args.out_md:
        with open(args.out_md, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")
    print(text)


if __name__ == "__main__":
    main()
