#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Build a performance-ceiling ledger from `vllm bench serve --omni` cells.

Each ``--cell`` is ``NAME=RESULT_JSON[:BENCH_LOG[:UTIL]]``:

- RESULT_JSON: ``--save-result`` output of one concurrency cell.
- BENCH_LOG: stdout of the same run with ``--print-stage`` (stage timing).
- UTIL: device utilisation sampled on the serving device during the run, either
  ``nvidia-smi --query-gpu=timestamp,utilization.gpu,... --format=csv`` or
  ``npu-smi info watch -s abmn``. The format is detected from the header.

Per-stage attribution, in order of preference:

- ``--stage-util NAME=STAGE:UTIL``: one device per stage, one sample file per
  device. Works on both CUDA and Ascend.
- ``--pmon NAME=PMON_TXT:SERVER_LOG``: CUDA only. Attributes the cell's busy
  share from ``nvidia-smi pmon -s u`` samples (MPS off), mapping PIDs to stages
  through ``StageEngineCoreProc_stage<N>`` lines in the server log. Samples are
  trimmed to the span between the first and last non-idle sample. Ascend has no
  per-process AI Core counter, so use ``--stage-util`` there.

Optional physical bounds need ``--peaks`` (``measure_peaks.py`` output). All
terms are at batch = cell concurrency. ``@STAGE`` joins a term to that stage's
measured g (``--stage-util`` or ``--pmon``) to report its % of SOL; terms
sharing a NAME add up within each class:

- ``--decode-bound NAME[@STAGE]=WEIGHT_GIB,TOKENS_PER_AUDIO_S[,KV_GIB_PER_SEQ]``:
  bandwidth floor for an autoregressive stage, reading weights once per step
  plus each sequence's KV cache.
- ``--compute-bound NAME[@STAGE]=TFLOP_PER_AUDIO_S,DTYPE[,GIB_PER_AUDIO_S]``:
  FLOP floor at the measured peak of DTYPE (bf16, fp16, tf32, hf32, fp32), and
  optionally the bytes the stage must move.
- ``--launch-bound NAME[@STAGE]=KERNELS_PER_STEP,STEPS_PER_AUDIO_S,eager|graph``:
  launch floor from the measured per-kernel latency; STEPS_PER_AUDIO_S is per
  sequence, so batching divides it.

The bounds cover kernel work and launch latency only, not host scheduling,
inter-stage transfer or queueing.
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
NPU_COLUMN = re.compile(r"[A-Za-z][A-Za-z0-9 _]*\([^)]*\)")
SMI_TIME_PREFIX = re.compile(r"\d{4}/\d{2}/\d{2} ")


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


def _nvidia_smi_samples(lines: list[str], column: str) -> list[tuple[dt.datetime | None, float]]:
    reader = csv.reader(lines)
    header = next(reader, None)
    if not header:
        return []
    util_col = next(i for i, name in enumerate(header) if column in name)
    samples = []
    for row in reader:
        if len(row) <= util_col:
            continue
        try:
            value = float(row[util_col].strip().rstrip("%").strip()) / 100.0
        except ValueError:
            continue
        samples.append((_parse_smi_time(row[0]), value))
    return samples


def _npu_smi_samples(lines: list[str], column: str) -> list[tuple[dt.datetime | None, float]]:
    """Parse ``npu-smi info watch`` output, with or without a prepended timestamp.

    ``npu-smi`` emits fixed-width columns and no timestamp, so a sampler that
    prefixes ``<date>, `` per line (see references/npu-ceiling.md) is what makes
    window trimming possible; without it every sample is kept.
    """
    header_index = next(i for i, line in enumerate(lines) if "AI Core(%)" in line)
    header = lines[header_index]
    has_time = bool(SMI_TIME_PREFIX.match(header))
    # Columns are fixed width and some names contain a space ("AI Core(%)"), so
    # names are read as "<label>(<unit>)" runs rather than split on whitespace.
    names = NPU_COLUMN.findall(header)
    util_col = names.index(column)
    samples = []
    for line in lines[header_index + 1 :]:
        stamp, _, payload = line.partition(", ") if has_time else ("", "", line)
        fields = payload.split()
        if len(fields) <= util_col or not fields[util_col].replace(".", "", 1).isdigit():
            continue
        samples.append((_parse_smi_time(stamp) if has_time else None, float(fields[util_col]) / 100.0))
    return samples


def device_busy_share(
    util_path: Path, window: tuple[dt.datetime, dt.datetime] | None, npu_column: str
) -> tuple[float, int, bool]:
    """Mean busy share in the measured window, sample count, and whether trimmed."""
    lines = util_path.read_text(errors="replace").splitlines()
    if any("AI Core(%)" in line for line in lines):
        samples = _npu_smi_samples(lines, npu_column)
    else:
        samples = _nvidia_smi_samples(lines, "utilization.gpu")
    # Untimestamped samples cannot be trimmed; the sampler must already be
    # scoped to the measured window.
    trimmed = bool(window) and bool(samples) and all(ts is not None for ts, _ in samples)
    values = [v for ts, v in samples if not (window and ts) or window[0] <= ts <= window[1]]
    return (sum(values) / len(values) if values else 0.0), len(values), trimmed


def measured_window(result: dict) -> tuple[dt.datetime, dt.datetime] | None:
    """Bench result ``date`` is written when the measured window ends."""
    try:
        end = dt.datetime.strptime(result["date"], "%Y%m%d-%H%M%S")
    except (KeyError, ValueError):
        return None
    start = end - dt.timedelta(seconds=float(result["duration"]))
    slack = dt.timedelta(seconds=2)
    return start + slack, end - slack


def measured_window_of(cell: dict) -> tuple[dt.datetime, dt.datetime] | None:
    window = cell.get("window")
    return None if window is None else (dt.datetime.fromisoformat(window[0]), dt.datetime.fromisoformat(window[1]))


def percentile(result: dict, name: str, p: float) -> float | None:
    value = result.get(f"p{p:g}_{name}")
    return None if value is None else float(value)


def analyse_cell(name: str, spec: str, npu_column: str) -> dict:
    parts = spec.split(":")
    result = json.loads(Path(parts[0]).read_text())
    concurrency = int(result["max_concurrency"])
    throughput = float(result["audio_throughput"])
    mean_audio_s = float(result["mean_audio_duration_s"])
    mean_e2el_s = float(result["mean_e2el_ms"]) / 1000.0

    window = measured_window(result)
    cell = {
        "name": name,
        "window": None if window is None else [window[0].isoformat(), window[1].isoformat()],
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
        busy, samples, trimmed = device_busy_share(Path(parts[2]), window, npu_column)
        cell["gpu_busy_share"] = busy
        cell["gpu_samples"] = samples
        cell["util_trimmed"] = trimmed
        if throughput and busy:
            # GPU-seconds per audio-second across all stages on this device.
            cell["g_total"] = busy / throughput
            # Same work with zero idle time.
            cell["rtf_floor_no_idle"] = concurrency * busy / throughput
    return cell


BOUND_CLASSES = ("compute", "memory", "launch")


def parse_bound(raw: str) -> tuple[str, str | None, list[str]]:
    """``NAME[@STAGE]=V1,V2,...`` into name, stage id (or None) and values."""
    key, _, values = raw.partition("=")
    name, _, stage = key.partition("@")
    return name, stage or None, values.split(",")


def measured_stage_g(cell: dict) -> dict[str, float]:
    """Measured device-s per audio-s by stage id; one device per stage beats pmon."""
    return cell.get("stage_device_g") or cell.get("pmon_stage_g") or {}


def physical_bounds(
    cell: dict, peaks: dict, decode: list[str], compute: list[str], launch: list[str], num_gpus: int
) -> dict:
    """Per-stage compute / memory / launch floors at batch = cell concurrency.

    Device time per audio second is ``max(compute, memory)`` (roofline). A stage
    runs its steps in one process, so its serial time per audio second is
    ``max(device, launch)``. Audio throughput is bounded both by the shared
    devices (``N / sum device``) and by the slowest stage's serial time.
    """
    bw = float(peaks["copy_bandwidth_gbs"]) * 1e9
    batch = cell["concurrency"]
    terms: dict[str, dict] = {}

    def term(name: str, stage: str | None) -> dict:
        entry = terms.setdefault(name, {"stage": None, **dict.fromkeys(BOUND_CLASSES, 0.0)})
        entry["stage"] = stage or entry["stage"]
        return entry

    for raw in decode:
        name, stage, values = parse_bound(raw)
        weight_gib, tokens_per_s = float(values[0]), float(values[1])
        kv_gib_per_seq = float(values[2]) if len(values) > 2 else 0.0
        step_s = (weight_gib + batch * kv_gib_per_seq) * 2**30 / bw
        term(name, stage)["memory"] += step_s / batch * tokens_per_s
    for raw in compute:
        name, stage, values = parse_bound(raw)
        entry = term(name, stage)
        entry["compute"] += float(values[0]) / float(peaks["matmul_tflops"][values[1]])
        if len(values) > 2:
            entry["memory"] += float(values[2]) * 2**30 / bw
    for raw in launch:
        name, stage, (kernels_per_step, steps_per_audio_s, mode) = parse_bound(raw)
        alpha_s = float(peaks["launch_latency_us"][mode]) * 1e-6
        term(name, stage)["launch"] += alpha_s * float(kernels_per_step) * float(steps_per_audio_s) / batch
    if not terms:
        return cell

    measured = measured_stage_g(cell)
    for entry in terms.values():
        entry["device"] = max(entry["compute"], entry["memory"])
        entry["serial"] = max(entry["device"], entry["launch"])
        entry["bound"] = max(BOUND_CLASSES, key=entry.get)
        entry["measured_g"] = measured.get(entry["stage"]) if entry["stage"] is not None else None
        entry["pct_of_sol"] = entry["device"] / entry["measured_g"] if entry["measured_g"] and entry["device"] else None
    device_total = sum(entry["device"] for entry in terms.values())
    limits = {f"stage {name} serial": 1.0 / entry["serial"] for name, entry in terms.items() if entry["serial"]}
    if device_total:
        limits["shared device"] = num_gpus / device_total
    binding = min(limits, key=limits.get)

    cell["bound_terms"] = terms
    cell["g_bound"] = {name: entry["device"] for name, entry in terms.items()}
    cell["g_bound_total"] = device_total
    cell["x_bound"] = limits[binding]
    cell["bound_binding"] = binding
    cell["rtf_bound"] = batch / limits[binding]
    if cell.get("aggregate_rtf"):
        cell["pct_of_sol"] = cell["rtf_bound"] / cell["aggregate_rtf"]
    return cell


def fmt(value, digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def fmt_pct(value) -> str:
    return "-" if value is None else f"{value:.0%}"


def render(cells: list[dict]) -> str:
    lines = [
        "| Cell | C | Done/Fail | Mean RTF | P50 | P99 | Aggregate RTF | C/X | Little residual "
        "| Audio X (s/s) | GPU busy | RTF if no idle | RTF bound | % of SOL |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for c in cells:
        lines.append(
            f"| {c['name']} | {c['concurrency']} | {c['completed']}/{c['failed']} | {fmt(c['mean_audio_rtf'])} "
            f"| {fmt(c['p50_audio_rtf'])} | {fmt(c['p99_audio_rtf'])} | {fmt(c['aggregate_rtf'])} "
            f"| {fmt(c['little_rtf'])} | {fmt(c.get('little_residual'), 3)} | {fmt(c['audio_throughput'], 2)} "
            f"| {fmt(c.get('gpu_busy_share'), 2)} | {fmt(c.get('rtf_floor_no_idle'))} | {fmt(c.get('rtf_bound'))} "
            f"| {fmt_pct(c.get('pct_of_sol'))} |"
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
        if shares := c.get("stage_device_share"):
            lines.append(f"Per-stage busy share and g_i from one device per stage ({c['name']}):")
            for stage, share in shares.items():
                g = c["stage_device_g"][stage]
                lines.append(f"  stage {stage}: busy {share:.2f}, g {g:.4f} device-s per audio-s")
            binding = max(shares, key=shares.get)
            lines.append(
                f"  binding stage {binding}; X bound by it: {1.0 / c['stage_device_g'][binding]:.2f} audio-s/s"
            )
        if terms := c.get("bound_terms"):
            lines.append(f"Physical bound at batch {c['concurrency']} ({c['name']}), seconds per audio-s:")
            for name, e in terms.items():
                stage = "" if e["stage"] is None else f" (stage {e['stage']})"
                measured = (
                    ""
                    if e["measured_g"] is None
                    else f"; measured g {e['measured_g']:.4f}, {fmt_pct(e['pct_of_sol'])} of SOL"
                )
                lines.append(
                    f"  {name}{stage}: compute {e['compute']:.4f}, memory {e['memory']:.4f}, "
                    f"launch {e['launch']:.4f} -> {e['bound']} bound{measured}"
                )
            if "g_total" in c:
                lines.append(f"  measured g_total {c['g_total']:.4f} vs device bound {c['g_bound_total']:.4f}")
            lines.append(
                f"  audio X bound {c['x_bound']:.2f} audio-s/s ({c['bound_binding']}); "
                f"RTF bound {c['rtf_bound']:.3f}, measured aggregate RTF at {fmt_pct(c.get('pct_of_sol'))} of SOL"
            )
    warnings = [
        f"{c['name']}: Little residual {c['little_residual']:+.1%}; concurrency was not held"
        for c in cells
        if abs(c.get("little_residual") or 0.0) > 0.10
    ]
    warnings += [f"{c['name']}: {c['failed']} failed requests" for c in cells if c["failed"]]
    warnings += [
        f"{c['name']}: utilisation samples carry no timestamp; busy share covers the whole sample "
        "file, not the measured window"
        for c in cells
        if c.get("gpu_samples") and not c.get("util_trimmed")
    ]
    warnings += [
        f"{c['name']}: RTF bound {c['rtf_bound']:.3f} is above measured aggregate RTF {c['aggregate_rtf']:.3f} "
        f"({c['bound_binding']}); a bound input overcounts"
        for c in cells
        if (c.get("pct_of_sol") or 0.0) > 1.05
    ]
    for c in cells:
        pmon_sum = sum((c.get("pmon_stage_share") or {}).values())
        busy = c.get("gpu_busy_share")
        if pmon_sum and busy and abs(pmon_sum / busy - 1.0) > 0.2:
            warnings.append(
                f"{c['name']}: pmon stage sum {pmon_sum:.2f} vs GPU busy {busy:.2f}; "
                "pmon dropped readings, use stage proportions only"
            )
        for name, e in (c.get("bound_terms") or {}).items():
            if e["pct_of_sol"] and e["pct_of_sol"] > 1.05:
                warnings.append(
                    f"{c['name']}: {name} measured g {e['measured_g']:.4f} is below its bound "
                    f"{e['device']:.4f}; the work count is too high, a peak too low, or attribution undercounts"
                )
    if warnings:
        lines.append("")
        lines.extend(f"WARNING {w}" for w in warnings)
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cell", action="append", required=True, help="NAME=RESULT_JSON[:BENCH_LOG[:UTIL]]")
    parser.add_argument("--pmon", action="append", default=[], help="NAME=PMON_TXT:SERVER_LOG (CUDA only)")
    parser.add_argument(
        "--stage-util", action="append", default=[], help="NAME=STAGE:UTIL, one device per stage (CUDA or Ascend)"
    )
    parser.add_argument(
        "--npu-util-column",
        default="AI Core(%)",
        help="column read from npu-smi info watch output; 'NPU Util(%%)' is the closer analogue of utilization.gpu",
    )
    parser.add_argument("--peaks", help="measure_peaks.py JSON")
    parser.add_argument(
        "--decode-bound",
        action="append",
        default=[],
        help="NAME[@STAGE]=WEIGHT_GIB,TOKENS_PER_AUDIO_S[,KV_GIB_PER_SEQ]",
    )
    parser.add_argument(
        "--compute-bound", action="append", default=[], help="NAME[@STAGE]=TFLOP_PER_AUDIO_S,DTYPE[,GIB_PER_AUDIO_S]"
    )
    parser.add_argument(
        "--launch-bound",
        action="append",
        default=[],
        help="NAME[@STAGE]=KERNELS_PER_STEP,STEPS_PER_AUDIO_S,eager|graph (needs launch_latency_us in --peaks)",
    )
    parser.add_argument("--num-gpus", type=int, default=1, help="devices shared by all stages")
    parser.add_argument("--json", help="write the ledger to this path")
    args = parser.parse_args()

    cells = []
    for raw in args.cell:
        name, _, spec = raw.partition("=")
        cells.append(analyse_cell(name, spec, args.npu_util_column))
    by_name = {c["name"]: c for c in cells}
    for raw in args.stage_util:
        name, _, spec = raw.partition("=")
        stage, _, util_path = spec.partition(":")
        if not util_path:
            parser.error(f"--stage-util {raw!r} must be NAME=STAGE:UTIL")
        cell = by_name[name]
        busy, _, _ = device_busy_share(Path(util_path), measured_window_of(cell), args.npu_util_column)
        cell.setdefault("stage_device_share", {})[stage] = busy
        cell.setdefault("stage_device_g", {})[stage] = busy / cell["audio_throughput"]
    for raw in args.pmon:
        name, _, spec = raw.partition("=")
        pmon_path, server_log = spec.split(":")
        cell = by_name[name]
        shares = pmon_stage_shares(Path(pmon_path), Path(server_log))
        cell["pmon_stage_share"] = shares
        cell["pmon_stage_g"] = {stage: share / cell["audio_throughput"] for stage, share in shares.items()}
    if args.peaks:
        peaks = json.loads(Path(args.peaks).read_text())
        if args.launch_bound and not peaks.get("launch_latency_us"):
            parser.error("--launch-bound needs launch_latency_us in --peaks; rerun measure_peaks.py")
        cells = [
            physical_bounds(c, peaks, args.decode_bound, args.compute_bound, args.launch_bound, args.num_gpus)
            for c in cells
        ]

    print(render(cells))
    if args.json:
        Path(args.json).write_text(json.dumps(cells, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
