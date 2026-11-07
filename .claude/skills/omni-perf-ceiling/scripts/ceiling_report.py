#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Build a performance-ceiling ledger from `vllm bench serve --omni` cells.

Each ``--cell`` is ``NAME=RESULT_JSON[:BENCH_LOG[:GPU_CSV]]``:

- RESULT_JSON: ``--save-result`` output of one concurrency cell.
- BENCH_LOG: stdout of the same run with ``--print-stage`` (stage timing).
- GPU_CSV: ``nvidia-smi --query-gpu=timestamp,utilization.gpu,... --format=csv``
  sampled on the serving device during the run.

``--pmon NAME=PMON_TXT:SERVER_LOG`` attributes the cell's GPU busy share to
stages from ``nvidia-smi pmon -s u`` samples (MPS off), mapping PIDs to stages
through ``StageEngineCoreProc_stage<N>`` lines in the server log. Samples are
trimmed to the span between the first and last non-idle sample.

Optional physical bounds need ``--peaks`` (``measure_peaks.py`` output):

- ``--decode-bound NAME=WEIGHT_GIB,TOKENS_PER_AUDIO_S``: bandwidth floor for an
  autoregressive stage at batch = cell concurrency (weights only, KV ignored).
- ``--compute-bound NAME=TFLOP_PER_AUDIO_S,DTYPE``: FLOP floor for a compute
  stage at the measured peak of DTYPE (bf16, fp16, tf32, fp32).
"""

import argparse
import csv
import datetime as dt
import json
import sys
from pathlib import Path

import regex as re

STAGE_HEADER = re.compile(r"=+\s*Stage (\d+) \(([^)]*)\)\s*=+")
STAGE_GEN_MEAN = re.compile(r"Mean stage_gen_time \(ms\):\s+([\d.]+)")
STAGE_PID = re.compile(r"StageEngineCoreProc_stage(\d+)_replica\d+ pid=(\d+)")


def stage_pids(server_log: Path) -> dict[int, str]:
    pids: dict[int, str] = {}
    for line in server_log.read_text(errors="replace").splitlines():
        if match := STAGE_PID.search(line):
            pids[int(match.group(2))] = match.group(1)
    return pids


def pmon_stage_shares(pmon_path: Path, server_log: Path) -> dict[str, float]:
    """Mean SM share per stage over the active span of 1 Hz pmon samples."""
    pids = stage_pids(server_log)
    samples: list[dict[str, float]] = []
    current: dict[str, float] = {}
    seen: set[int] = set()
    for line in pmon_path.read_text(errors="replace").splitlines():
        fields = line.split()
        if line.startswith("#") or len(fields) < 4 or not fields[1].isdigit():
            continue
        pid = int(fields[1])
        # One line per process per interval; a repeated PID starts the next sample.
        if pid in seen:
            samples.append(current)
            current, seen = {}, set()
        seen.add(pid)
        stage = pids.get(pid)
        if stage is not None:
            # "-" means no reading in this interval.
            sm = float(fields[3]) / 100.0 if fields[3].isdigit() else 0.0
            current[stage] = current.get(stage, 0.0) + sm
    if seen:
        samples.append(current)
    active = [i for i, sample in enumerate(samples) if sum(sample.values()) > 0]
    if not active:
        return {}
    window = samples[active[0] : active[-1] + 1]
    stages = sorted({stage for sample in window for stage in sample})
    return {stage: sum(sample.get(stage, 0.0) for sample in window) / len(window) for stage in stages}


def parse_stage_times(log_path: Path) -> dict[str, float]:
    """Mean per-request stage_gen_time (ms) per stage from a --print-stage log."""
    stages: dict[str, float] = {}
    current = None
    for line in log_path.read_text(errors="replace").splitlines():
        if header := STAGE_HEADER.search(line):
            current = f"{header.group(1)}:{header.group(2)}"
            continue
        if current and (mean := STAGE_GEN_MEAN.search(line)):
            stages[current] = float(mean.group(1))
            current = None
    return stages


def _parse_smi_time(raw: str) -> dt.datetime | None:
    for fmt in ("%Y/%m/%d %H:%M:%S.%f", "%Y/%m/%d %H:%M:%S"):
        try:
            return dt.datetime.strptime(raw.strip(), fmt)
        except ValueError:
            continue
    return None


def gpu_busy_share(csv_path: Path, window: tuple[dt.datetime, dt.datetime] | None) -> tuple[float, int]:
    """Mean utilization.gpu share inside the measured window, and sample count."""
    values = []
    with csv_path.open() as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            return 0.0, 0
        util_col = next(i for i, name in enumerate(header) if "utilization.gpu" in name)
        for row in reader:
            if len(row) <= util_col:
                continue
            ts = _parse_smi_time(row[0])
            if window and ts and not (window[0] <= ts <= window[1]):
                continue
            try:
                values.append(float(row[util_col].strip().rstrip("%").strip()) / 100.0)
            except ValueError:
                continue
    return (sum(values) / len(values) if values else 0.0), len(values)


def measured_window(result: dict) -> tuple[dt.datetime, dt.datetime] | None:
    """Bench result ``date`` is written when the measured window ends."""
    try:
        end = dt.datetime.strptime(result["date"], "%Y%m%d-%H%M%S")
    except (KeyError, ValueError):
        return None
    start = end - dt.timedelta(seconds=float(result["duration"]))
    slack = dt.timedelta(seconds=2)
    return start + slack, end - slack


def percentile(result: dict, name: str, p: float) -> float | None:
    value = result.get(f"p{p:g}_{name}")
    return None if value is None else float(value)


def analyse_cell(name: str, spec: str) -> dict:
    parts = spec.split(":")
    result = json.loads(Path(parts[0]).read_text())
    concurrency = int(result["max_concurrency"])
    throughput = float(result["audio_throughput"])
    mean_audio_s = float(result["mean_audio_duration_s"])
    mean_e2el_s = float(result["mean_e2el_ms"]) / 1000.0

    cell = {
        "name": name,
        "concurrency": concurrency,
        "completed": result["completed"],
        "failed": result["failed"],
        "duration_s": float(result["duration"]),
        "audio_throughput": throughput,
        "mean_audio_rtf": float(result["mean_audio_rtf"]),
        "p50_audio_rtf": float(result["median_audio_rtf"]),
        "p99_audio_rtf": percentile(result, "audio_rtf", 99.0),
        "mean_audio_ttfp_ms": float(result.get("mean_audio_ttfp_ms", 0.0)),
        "mean_audio_duration_s": mean_audio_s,
        # Ratio of means; E2EL ends at the last audio packet for audio requests.
        "aggregate_rtf": mean_e2el_s / mean_audio_s if mean_audio_s else None,
        "little_rtf": concurrency / throughput if throughput else None,
    }
    if cell["aggregate_rtf"] and cell["little_rtf"]:
        cell["little_residual"] = cell["little_rtf"] / cell["aggregate_rtf"] - 1.0

    if len(parts) > 1 and parts[1]:
        stage_ms = parse_stage_times(Path(parts[1]))
        cell["stage_gen_ms"] = stage_ms
        cell["stage_share_of_audio"] = {k: v / (mean_audio_s * 1000.0) for k, v in stage_ms.items()}
    if len(parts) > 2 and parts[2]:
        busy, samples = gpu_busy_share(Path(parts[2]), measured_window(result))
        cell["gpu_busy_share"] = busy
        cell["gpu_samples"] = samples
        if throughput and busy:
            # GPU-seconds per audio-second across all stages on this device.
            cell["g_total"] = busy / throughput
            # Same work with zero idle time.
            cell["rtf_floor_no_idle"] = concurrency * busy / throughput
    return cell


def parse_bound(raw: str) -> tuple[str, list[str]]:
    name, _, values = raw.partition("=")
    return name, values.split(",")


def physical_bounds(cell: dict, peaks: dict, decode: list[str], compute: list[str], num_gpus: int) -> dict:
    bounds = {}
    bw = float(peaks["copy_bandwidth_gbs"]) * 1e9
    batch = cell["concurrency"]
    for raw in decode:
        name, (weight_gib, tokens_per_s) = parse_bound(raw)
        step_s = float(weight_gib) * 2**30 / bw
        bounds[name] = step_s / batch * float(tokens_per_s)
    for raw in compute:
        name, (tflop_per_s, dtype) = parse_bound(raw)
        bounds[name] = float(tflop_per_s) / float(peaks["matmul_tflops"][dtype])
    if bounds:
        total = sum(bounds.values())
        cell["g_bound"] = bounds
        cell["g_bound_total"] = total
        cell["rtf_bound"] = batch * total / num_gpus
    return cell


def fmt(value, digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def render(cells: list[dict]) -> str:
    lines = [
        "| Cell | C | Done/Fail | Mean RTF | P50 | P99 | Aggregate RTF | C/X | Little residual "
        "| Audio X (s/s) | GPU busy | RTF if no idle | RTF bound |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for c in cells:
        lines.append(
            f"| {c['name']} | {c['concurrency']} | {c['completed']}/{c['failed']} | {fmt(c['mean_audio_rtf'])} "
            f"| {fmt(c['p50_audio_rtf'])} | {fmt(c['p99_audio_rtf'])} | {fmt(c['aggregate_rtf'])} "
            f"| {fmt(c['little_rtf'])} | {fmt(c.get('little_residual'), 3)} | {fmt(c['audio_throughput'], 2)} "
            f"| {fmt(c.get('gpu_busy_share'), 2)} | {fmt(c.get('rtf_floor_no_idle'))} | {fmt(c.get('rtf_bound'))} |"
        )
    for c in cells:
        if shares := c.get("stage_share_of_audio"):
            lines.append("")
            lines.append(f"Stage gen time / audio duration ({c['name']}):")
            slowest = max(shares, key=shares.get)
            for stage, share in shares.items():
                mark = "  <- slowest" if stage == slowest else ""
                lines.append(f"  stage {stage}: {share:.3f}{mark}")
        if shares := c.get("pmon_stage_share"):
            lines.append(f"Per-stage SM share and g_i from pmon ({c['name']}):")
            for stage, share in shares.items():
                g = c["pmon_stage_g"][stage]
                lines.append(f"  stage {stage}: SM share {share:.2f}, g {g:.4f} GPU-s per audio-s")
            lines.append(f"  idle {max(0.0, 1.0 - sum(shares.values())):.2f}")
        if bounds := c.get("g_bound"):
            lines.append(f"Physical g bound at batch {c['concurrency']} ({c['name']}):")
            for stage, g in bounds.items():
                lines.append(f"  {stage}: {g:.4f} GPU-s per audio-s")
            if "g_total" in c:
                lines.append(f"  measured g_total {c['g_total']:.4f} vs bound {c['g_bound_total']:.4f}")
    warnings = [
        f"{c['name']}: Little residual {c['little_residual']:+.1%}; concurrency was not held"
        for c in cells
        if abs(c.get("little_residual") or 0.0) > 0.10
    ]
    warnings += [f"{c['name']}: {c['failed']} failed requests" for c in cells if c["failed"]]
    for c in cells:
        pmon_sum = sum((c.get("pmon_stage_share") or {}).values())
        busy = c.get("gpu_busy_share")
        if pmon_sum and busy and abs(pmon_sum / busy - 1.0) > 0.2:
            warnings.append(
                f"{c['name']}: pmon stage sum {pmon_sum:.2f} vs GPU busy {busy:.2f}; "
                "pmon dropped readings, use stage proportions only"
            )
    if warnings:
        lines.append("")
        lines.extend(f"WARNING {w}" for w in warnings)
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cell", action="append", required=True, help="NAME=RESULT_JSON[:BENCH_LOG[:GPU_CSV]]")
    parser.add_argument("--pmon", action="append", default=[], help="NAME=PMON_TXT:SERVER_LOG")
    parser.add_argument("--peaks", help="measure_peaks.py JSON")
    parser.add_argument("--decode-bound", action="append", default=[], help="NAME=WEIGHT_GIB,TOKENS_PER_AUDIO_S")
    parser.add_argument("--compute-bound", action="append", default=[], help="NAME=TFLOP_PER_AUDIO_S,DTYPE")
    parser.add_argument("--num-gpus", type=int, default=1, help="devices shared by all stages")
    parser.add_argument("--json", help="write the ledger to this path")
    args = parser.parse_args()

    cells = []
    for raw in args.cell:
        name, _, spec = raw.partition("=")
        cells.append(analyse_cell(name, spec))
    by_name = {c["name"]: c for c in cells}
    for raw in args.pmon:
        name, _, spec = raw.partition("=")
        pmon_path, server_log = spec.split(":")
        cell = by_name[name]
        shares = pmon_stage_shares(Path(pmon_path), Path(server_log))
        cell["pmon_stage_share"] = shares
        cell["pmon_stage_g"] = {stage: share / cell["audio_throughput"] for stage, share in shares.items()}
    if args.peaks:
        peaks = json.loads(Path(args.peaks).read_text())
        cells = [physical_bounds(c, peaks, args.decode_bound, args.compute_bound, args.num_gpus) for c in cells]

    print(render(cells))
    if args.json:
        Path(args.json).write_text(json.dumps(cells, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
