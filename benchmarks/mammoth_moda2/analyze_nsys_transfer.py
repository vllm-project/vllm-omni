#!/usr/bin/env python3
"""Validate worker coverage and summarize MammothModa2 Nsight transfers."""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


STAGE_PID_RE = re.compile(r"StageEngineCoreProc_stage(?P<stage>\d+)_replica\d+ pid=(?P<pid>\d+)")
STAGE_GPU_RE = re.compile(r"Stage (?P<stage>\d+) logical-to-physical device mapping: \d+->(?P<gpu>\d+)")
DEVICE_MEMORY_KIND = 2
HOST_MEMORY_KINDS = {0, 1}


def _os_pid(global_id: int | None) -> int | None:
    """Nsight encodes global PID/TID as ``(os_pid << 32) | local_id``."""
    return None if global_id is None else int(global_id) >> 32


def _read_expected_workers(log_path: Path) -> tuple[dict[int, set[int]], dict[int, int]]:
    text = log_path.read_text(errors="replace")
    workers: dict[int, set[int]] = defaultdict(set)
    for match in STAGE_PID_RE.finditer(text):
        workers[int(match["stage"])].add(int(match["pid"]))
    gpus = {int(match["stage"]): int(match["gpu"]) for match in STAGE_GPU_RE.finditer(text)}
    if not workers or not gpus:
        raise ValueError(f"Missing stage PID or physical-GPU mapping in {log_path}")
    return dict(workers), gpus


def _direction(src_kind: int | None, dst_kind: int | None) -> str:
    if src_kind == DEVICE_MEMORY_KIND and dst_kind in HOST_MEMORY_KINDS:
        return "DtoH"
    if src_kind in HOST_MEMORY_KINDS and dst_kind == DEVICE_MEMORY_KIND:
        return "HtoD"
    if src_kind == DEVICE_MEMORY_KIND and dst_kind == DEVICE_MEMORY_KIND:
        return "DtoD"
    return "other"


def _pid_filter(column: str, pids: set[int]) -> tuple[str, tuple[int, ...]]:
    placeholders = ",".join("?" for _ in pids)
    return f"({column} >> 32) IN ({placeholders})", tuple(sorted(pids))


def _stage_summary(con: sqlite3.Connection, pids: set[int]) -> dict[str, Any]:
    pid_filter, params = _pid_filter("globalPid", pids)
    copies = con.execute(
        "SELECT globalPid, deviceId, start, end, bytes, srcKind, dstKind "
        f"FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE {pid_filter}", params
    ).fetchall()
    kernels = con.execute(
        "SELECT globalPid, deviceId, COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL "
        f"WHERE {pid_filter} GROUP BY globalPid, deviceId", params
    ).fetchall()
    groups: dict[str, dict[str, int]] = defaultdict(lambda: {"count": 0, "bytes": 0, "duration_ns": 0})
    largest: dict[str, list[dict[str, int | None]]] = defaultdict(list)
    for global_pid, device, start, end, size, src_kind, dst_kind in copies:
        name = _direction(src_kind, dst_kind)
        duration = max(0, int(end) - int(start))
        groups[name]["count"] += 1
        groups[name]["bytes"] += int(size)
        groups[name]["duration_ns"] += duration
        largest[name].append({"pid": _os_pid(global_pid), "device": int(device), "bytes": int(size), "duration_ns": duration})

    tid_filter, tid_params = _pid_filter("r.globalTid", pids)
    apis = con.execute(
        "SELECT r.globalTid, s.value, COUNT(*), SUM(r.end - r.start) "
        "FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON r.nameId = s.id "
        f"WHERE {tid_filter} AND (s.value LIKE 'cudaMemcpy%' OR "
        "s.value LIKE 'cuda%Synchronize%' OR s.value LIKE 'cudaStreamWait%') "
        "GROUP BY r.globalTid, s.value ORDER BY s.value", tid_params
    ).fetchall()
    return {
        "captured_pids": sorted({_os_pid(row[0]) for row in copies} | {_os_pid(row[0]) for row in kernels}),
        "devices": sorted({int(row[1]) for row in copies} | {int(row[1]) for row in kernels}),
        "kernel_count": sum(int(row[2]) for row in kernels),
        "memory_operations": dict(sorted(groups.items())),
        "largest_memory_operations": {
            name: sorted(rows, key=lambda row: int(row["bytes"]), reverse=True)[:10]
            for name, rows in sorted(largest.items())
        },
        "cuda_api": [
            {"pid": _os_pid(pid), "api": name, "count": int(count), "duration_ns": int(duration or 0)}
            for pid, name, count, duration in apis
        ],
    }


def _request_end_d2h(con: sqlite3.Connection, pids: set[int]) -> dict[str, int]:
    tid_filter, tid_params = _pid_filter("globalTid", pids)
    ranges = con.execute(
        "SELECT start, end FROM NVTX_EVENTS WHERE "
        "COALESCE(text, (SELECT value FROM StringIds WHERE id = textId)) "
        "= 'mammoth_moda2:ar2dit_full_payload_d2h' "
        f"AND {tid_filter} AND end IS NOT NULL", tid_params
    ).fetchall()
    pid_filter, pid_params = _pid_filter("globalPid", pids)
    result = {"range_count": len(ranges), "copy_count": 0, "bytes": 0, "duration_ns": 0}
    for start, end in ranges:
        rows = con.execute(
            "SELECT start, end, bytes FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE "
            f"{pid_filter} AND srcKind = ? AND dstKind IN (?, ?) AND start < ? AND end > ?",
            pid_params + (DEVICE_MEMORY_KIND, *sorted(HOST_MEMORY_KINDS), end, start),
        ).fetchall()
        for copy_start, copy_end, size in rows:
            result["copy_count"] += 1
            result["bytes"] += int(size)
            result["duration_ns"] += max(0, int(copy_end) - int(copy_start))
    return result


def _analyze_case(results_dir: Path, label: str) -> tuple[dict[str, Any], bool]:
    workers, expected_gpus = _read_expected_workers(results_dir / f"{label}_profile.log")
    sqlite_path = results_dir / f"nsys_{label}.sqlite"
    if not sqlite_path.is_file():
        raise FileNotFoundError(f"Missing Nsight SQLite export: {sqlite_path}")
    stages: dict[str, Any] = {}
    valid = True
    with sqlite3.connect(sqlite_path) as con:
        for stage, pids in sorted(workers.items()):
            summary = _stage_summary(con, pids)
            summary["expected_pids"] = sorted(pids)
            summary["expected_gpu"] = expected_gpus.get(stage)
            summary["missing_pids"] = sorted(pids - set(summary["captured_pids"]))
            summary["request_end_payload_d2h"] = _request_end_d2h(con, pids)
            stages[str(stage)] = summary
            valid = valid and not summary["missing_pids"] and summary["expected_gpu"] in summary["devices"]
    return {"label": label, "stages": stages}, valid


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output: dict[str, Any] = {"methodology": {"comparison": "worker PID and physical GPU matched"}}
    valid = True
    for label in ("baseline", "optimized"):
        output[label], case_valid = _analyze_case(args.results_dir, label)
        valid = valid and case_valid
    layouts = {
        label: {stage: len(summary["expected_pids"]) for stage, summary in output[label]["stages"].items()}
        for label in ("baseline", "optimized")
    }
    output["comparison_layout"] = layouts
    valid = valid and layouts["baseline"] == layouts["optimized"] == {"0": 1, "1": 1}
    path = args.output or args.results_dir / "nsys_transfer_attribution.json"
    path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))
    print(f"Wrote {path}")
    if not valid:
        print("Nsight worker coverage is incomplete or GPU mapping is wrong. Do not compare transfer totals.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
