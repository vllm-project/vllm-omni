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

Optional physical bounds need ``--peaks`` (``measure_peaks.py`` output) and
work counts, which ``count_reference_work.py`` takes from the checkpoint's
reference implementation and prints as the flags below. All
terms are at batch = cell concurrency. ``@STAGE`` joins a term to that stage's
measured g (``--stage-util`` or ``--pmon``) to report its % of SOL; terms
sharing a NAME add up within each class:

- ``--decode-bound NAME[@STAGE]=WEIGHT_GIB,TOKENS_PER_AUDIO_S[,KV_GIB_PER_SEQ[,GFLOP_PER_TOKEN,DTYPE]]``:
  roofline of an autoregressive stage, reading weights once per step plus each
  sequence's KV cache, against each token's FLOPs.
- ``--compute-bound NAME[@STAGE]=TFLOP_PER_AUDIO_S,DTYPE[,GIB_PER_AUDIO_S]``:
  FLOP floor at the measured peak of DTYPE (bf16, fp16, tf32, hf32, fp32), and
  optionally the bytes the stage must move.
- ``--launch-bound NAME[@STAGE]=KERNELS_PER_STEP,STEPS_PER_AUDIO_S,eager|graph``:
  launch floor from the measured per-kernel latency; STEPS_PER_AUDIO_S is per
  sequence, so batching divides it.

Per-request terms, ``NAME[@STAGE]=TFLOP,DTYPE[,WEIGHT_GIB[,KERNELS,MODE]]``, where
STAGE is the first stage the term gates (default 0):

- ``--fixed-bound``: work every request adds beyond its audio (Thinker prefill,
  encoders). It lengthens the critical path and adds device work.
- ``--fill-bound``: latency before STAGE can start whose work is already in
  the per-audio terms (first Talker chunk before Code2Wav). ``@out`` marks the
  last stage's work on the first packet (first Code2Wav chunk).

Each request then satisfies ``RTF_r >= a / d_r + b``. The report prints the
aggregate RTF floor (``a / mean(d) + b``), the mean RTF floor (per request), and
the measured ``a`` and ``b`` solved from mean and aggregate RTF. The same terms
bound first-token latency (terms gating stage 0) and first-packet latency (all
terms), compared with the bench's mean TTFT and AUDIO_TTFP.

Audits: ``--kv-capacity STAGE=KV_CACHE_TOKENS,TOKENS_PER_REQUEST[,BLOCK]``
warns when a stage's KV cache cannot hold C requests; ``--series A,B,...``
lists cells of one configuration and warns when throughput, aggregate RTF,
TTFT or TTFP falls as C rises.

Two readings sit beside the bound and never replace it:

- ``--efficiency-target compute=F,memory=F``: the shared-device floor with each
  row priced at a fraction of the measured peak (``F <= 1``). It is a
  projection of one utilization target, not a lower bound.
- ``--profiled NAME=SECONDS_PER_AUDIO_S``: a profiled duration of the
  implementation the row names. The implementation floor is
  ``max(roofline, profiled)``; it constrains that implementation only.

Rows of one stage are summed roofline by roofline, so each flag must cover
kernels no other flag of that stage covers. The bounds cover kernel work and
launch latency only, not host scheduling, inter-stage transfer or queueing.
"""

import argparse
import csv
import datetime as dt
import json
import math
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
) -> tuple[float | None, int, bool]:
    """Mean busy share (None without samples), sample count, and whether trimmed."""
    lines = util_path.read_text(errors="replace").splitlines()
    if any("AI Core(%)" in line for line in lines):
        samples = _npu_smi_samples(lines, npu_column)
    else:
        samples = _nvidia_smi_samples(lines, "utilization.gpu")
    # Untimestamped samples cannot be trimmed; the sampler must already be
    # scoped to the measured window.
    trimmed = bool(window) and bool(samples) and all(ts is not None for ts, _ in samples)
    values = [v for ts, v in samples if not (window and ts) or window[0] <= ts <= window[1]]
    return (sum(values) / len(values) if values else None), len(values), trimmed


def measured_window(result: dict) -> tuple[dt.datetime, dt.datetime] | None:
    """Bench result ``date`` is written when the measured window ends."""
    try:
        end = dt.datetime.strptime(result["date"], "%Y%m%d-%H%M%S")
    except (KeyError, ValueError):
        return None
    start = end - dt.timedelta(seconds=float(result["duration"]))
    slack = dt.timedelta(seconds=2)
    # Keep the full measured span when trimming would empty or invert it.
    if end - start <= 2 * slack:
        return start, end
    return start + slack, end - slack


def measured_window_of(cell: dict) -> tuple[dt.datetime, dt.datetime] | None:
    window = cell.get("window")
    return None if window is None else (dt.datetime.fromisoformat(window[0]), dt.datetime.fromisoformat(window[1]))


def percentile(result: dict, name: str, p: float) -> float | None:
    value = result.get(f"p{p:g}_{name}")
    return None if value is None else float(value)


def request_audio_durations(result: dict) -> list[float]:
    """Per-request audio seconds from ``--save-detailed`` stage metrics."""
    durations = []
    for metrics in result.get("request_stage_metrics") or []:
        audio_s = max((float(stage.get("audio_duration_s") or 0.0) for stage in (metrics or {}).values()), default=0.0)
        if audio_s > 0:
            durations.append(audio_s)
    return durations


def fixed_rate_split(mean_rtf: float, aggregate_rtf: float, mean_audio_s: float, durations: list[float]) -> dict:
    """Solve ``T_r = a + b * d_r`` from the mean and aggregate RTF.

    ``mean RTF = b + a * mean(1/d)`` and ``aggregate RTF = b + a / mean(d)``; only
    the spread of ``1/d`` across requests separates ``a`` from ``b``.
    """
    inv_mean = sum(1.0 / d for d in durations) / len(durations)
    spread = inv_mean - 1.0 / mean_audio_s
    if spread <= 0.02 * inv_mean:
        return {}
    fixed_s = (mean_rtf - aggregate_rtf) / spread
    return {"measured_fixed_s": fixed_s, "measured_rate": aggregate_rtf - fixed_s / mean_audio_s}


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
        "mean_ttft_ms": float(result.get("mean_ttft_ms") or 0.0),
        "p50_ttft_ms": float(result.get("median_ttft_ms") or 0.0),
        "mean_audio_ttfp_ms": float(result.get("mean_audio_ttfp_ms") or 0.0),
        "p50_audio_ttfp_ms": float(result.get("median_audio_ttfp_ms") or 0.0),
        "mean_audio_duration_s": mean_audio_s,
        # Ratio of means; E2EL ends at the last audio packet for audio requests.
        "aggregate_rtf": mean_e2el_s / mean_audio_s if mean_audio_s else None,
        "little_rtf": concurrency / throughput if throughput else None,
    }
    if cell["aggregate_rtf"] and cell["little_rtf"]:
        cell["little_residual"] = cell["little_rtf"] / cell["aggregate_rtf"] - 1.0
    if durations := request_audio_durations(result):
        cell["request_audio_s"] = durations
        cell["audio_coverage"] = len(durations) / max(1, int(result["completed"]))
        if cell["aggregate_rtf"]:
            cell.update(fixed_rate_split(cell["mean_audio_rtf"], cell["aggregate_rtf"], mean_audio_s, durations))

    if len(parts) > 1 and parts[1]:
        stage_ms = parse_stage_times(Path(parts[1]))
        cell["stage_gen_ms"] = stage_ms
        cell["stage_share_of_audio"] = {k: v / (mean_audio_s * 1000.0) for k, v in stage_ms.items()}
    if len(parts) > 2 and parts[2]:
        busy, samples, trimmed = device_busy_share(Path(parts[2]), window, npu_column)
        cell["gpu_busy_share"] = busy
        cell["gpu_samples"] = samples
        cell["util_trimmed"] = trimmed
        if throughput and busy is not None:
            # GPU-seconds per audio-second across all stages on this device.
            cell["g_total"] = busy / throughput
            # Same work with zero idle time.
            cell["rtf_floor_no_idle"] = concurrency * busy / throughput
    return cell


BOUND_CLASSES = ("compute", "memory", "launch")
OUTPUT_STAGE = "out"


def parse_bound(raw: str) -> tuple[str, str | None, list[str]]:
    """``NAME[@STAGE]=V1,V2,...`` into name, stage id (or None) and values."""
    key, _, values = raw.partition("=")
    name, _, stage = key.partition("@")
    return name, stage or None, values.split(",")


def measured_stage_g(cell: dict) -> dict[str, float | None]:
    """Measured device-s per audio-s by stage id; one device per stage beats pmon."""
    return cell.get("stage_device_g") or cell.get("pmon_stage_g") or {}


def alpha_s(peaks: dict, mode: str) -> float:
    latency = peaks.get("launch_latency_us") or {}
    if latency.get(mode) is None:
        raise SystemExit(f"launch_latency_us[{mode!r}] missing from --peaks; rerun measure_peaks.py")
    return float(latency[mode]) * 1e-6


def stage_order(stage: str | None) -> float:
    """Position a term gates; ``out`` is the first audio packet, after every stage."""
    if stage == OUTPUT_STAGE:
        return math.inf
    return int(stage) if stage and stage.isdigit() else 0


def request_terms(raws: list[str], peaks: dict, bw: float, batch: int, adds_work: bool) -> list[dict]:
    """Per-request work on the critical path: ``TFLOP,DTYPE[,WEIGHT_GIB[,KERNELS,MODE]]``.

    One step reads its weights once for the whole batch, so the work a request
    adds to the shared device divides the weight term by the batch, while its
    own latency does not. ``adds_work=False`` is a fill term whose work is
    already counted per audio second.
    """
    terms = []
    for raw in raws:
        name, stage, values = parse_bound(raw)
        flops_s = float(values[0]) / float(peaks["matmul_tflops"][values[1]])
        bytes_s = float(values[2]) * 2**30 / bw if len(values) > 2 and values[2] else 0.0
        launch_s = alpha_s(peaks, values[4]) * float(values[3]) if len(values) > 4 else 0.0
        terms.append(
            {
                "name": name,
                "stage": stage,
                "kind": "fixed" if adds_work else "fill",
                "work": max(flops_s, bytes_s / batch) if adds_work else 0.0,
                "latency": max(flops_s, bytes_s, launch_s),
            }
        )
    return terms


def stage_terms(specs: dict[str, list[str]], peaks: dict, bw: float, batch: int) -> dict[str, dict]:
    """Per-stage seconds per audio second at ``batch``.

    Every flag is one row: a group of kernels that runs on the stage's stream
    without overlapping the stage's other rows. The device floor is the sum of
    each row's roofline, ``sum max(compute, memory)``, which is at least the
    stage-level ``max(sum compute, sum memory)``. Rows must not share kernels.
    """
    terms: dict[str, dict] = {}

    def row(name: str, stage: str | None, compute: float, memory: float) -> dict:
        entry = terms.setdefault(name, {"stage": None, "device": 0.0, "rows": [], **dict.fromkeys(BOUND_CLASSES, 0.0)})
        entry["stage"] = stage or entry["stage"]
        entry["rows"].append((compute, memory))
        entry["compute"] += compute
        entry["memory"] += memory
        entry["device"] += max(compute, memory)
        return entry

    for raw in specs["decode"]:
        name, stage, values = parse_bound(raw)
        weight_gib, tokens_per_s = float(values[0]), float(values[1])
        kv_gib_per_seq = float(values[2]) if len(values) > 2 and values[2] else 0.0
        step_s = (weight_gib + batch * kv_gib_per_seq) * 2**30 / bw
        # Each sequence's token FLOPs are paid whatever the batch.
        flops_s = (
            float(values[3]) * 1e-3 / float(peaks["matmul_tflops"][values[4]]) * tokens_per_s
            if len(values) > 4
            else 0.0
        )
        row(name, stage, flops_s, step_s / batch * tokens_per_s)
    for raw in specs["compute"]:
        name, stage, values = parse_bound(raw)
        flops_s = float(values[0]) / float(peaks["matmul_tflops"][values[1]])
        bytes_s = float(values[2]) * 2**30 / bw if len(values) > 2 else 0.0
        row(name, stage, flops_s, bytes_s)
    for raw in specs["launch"]:
        name, stage, (kernels_per_step, steps_per_audio_s, mode) = parse_bound(raw)
        launch_s = alpha_s(peaks, mode) * float(kernels_per_step) * float(steps_per_audio_s) / batch
        row(name, stage, 0.0, 0.0)["launch"] += launch_s
    for entry in terms.values():
        entry["serial"] = max(entry["device"], entry["launch"])
        if entry["launch"] > entry["device"]:
            entry["bound"] = "launch"
        else:
            entry["bound"] = "compute" if entry["compute"] >= entry["memory"] else "memory"
    return terms


def efficiency_target(raw: str | None) -> dict[str, float] | None:
    """``compute=F,memory=F`` with each fraction in (0, 1]."""
    if not raw:
        return None
    parsed = {}
    for part in raw.split(","):
        key, _, value = part.partition("=")
        try:
            fraction = float(value)
        except ValueError:
            fraction = 0.0
        if key not in ("compute", "memory") or not 0.0 < fraction <= 1.0:
            raise SystemExit(f"--efficiency-target {raw!r}: each entry is compute=F or memory=F with 0 < F <= 1")
        parsed[key] = fraction
    return parsed


def scaled_device(entry: dict, efficiency: dict[str, float]) -> float:
    """Row time at a fraction of the measured peak, taken per row before summing."""
    total = 0.0
    for compute, memory in entry["rows"]:
        total += max(compute / efficiency.get("compute", 1.0), memory / efficiency.get("memory", 1.0))
    return total


def physical_bounds(
    cell: dict,
    peaks: dict,
    specs: dict[str, list[str]],
    num_gpus: int,
    efficiency: dict[str, float] | None = None,
    profiled: dict[str, float] | None = None,
) -> dict:
    """End-to-end RTF, TTFT and TTFP floors for one cell.

    Per audio second, each stage's device time is the sum of its rows'
    rooflines and its serial time in its own process is ``max(device, launch)``.
    Per request, fixed terms (prefill) add work and latency, fill terms (first
    chunk before a downstream stage, or before the first packet for ``@out``)
    add latency only. Each RTF floor has the form ``RTF_r >= a / d_r + b``:

    - shared device: ``a = C * W / N``, ``b = C * sum device(C) / N``
      (work conservation; aggregate only above C = 1);
    - stage j throughput: ``a = 0``, ``b = C * serial_j(C)`` (one process runs
      at most C sequences per step; aggregate only above C = 1);
    - stage j path: ``a = sum of request latencies gating stages <= j``,
      ``b = serial_j(1)`` (one request's critical path; a step never runs
      faster than at batch 1, so it holds per request at any C).

    Aggregate RTF is bounded by the largest ``a / mean(d) + b``, mean RTF by
    the per-request mean of the largest path floor. TTFT is bounded by the
    latencies gating stage 0, TTFP by every per-request latency.
    """
    bw = float(peaks["copy_bandwidth_gbs"]) * 1e9
    batch = cell["concurrency"]
    terms = stage_terms(specs, peaks, bw, batch)
    single = terms if batch == 1 else stage_terms(specs, peaks, bw, 1)
    per_request = request_terms(specs["fixed"], peaks, bw, batch, True) + request_terms(
        specs["fill"], peaks, bw, batch, False
    )
    if not terms and not per_request:
        return cell

    measured = measured_stage_g(cell)
    for name, entry in terms.items():
        entry["serial_1"] = single[name]["serial"]
        entry["measured_g"] = measured.get(entry["stage"]) if entry["stage"] is not None else None
        entry["pct_of_sol"] = entry["device"] / entry["measured_g"] if entry["measured_g"] and entry["device"] else None
    device_total = sum(entry["device"] for entry in terms.values())
    work_s = sum(t["work"] for t in per_request)

    floors: dict[str, dict] = {}

    def floor(key: str, fixed: float, rate: float, per_request_scope: bool) -> None:
        floors[key] = {"fixed_s": fixed, "rate": rate, "per_request": per_request_scope or batch == 1}

    if device_total or work_s:
        floor("shared device", batch * work_s / num_gpus, batch * device_total / num_gpus, False)
    for name, entry in terms.items():
        order = stage_order(entry["stage"])
        gate_s = sum(t["latency"] for t in per_request if stage_order(t["stage"]) <= order)
        if batch > 1:
            floor(f"stage {name} throughput", 0.0, batch * entry["serial"], False)
        floor(f"stage {name} path", gate_s, entry["serial_1"], True)
    if per_request:
        floor("first packet", sum(t["latency"] for t in per_request), 0.0, True)

    mean_d = cell["mean_audio_duration_s"]
    aggregate = {key: f["fixed_s"] / mean_d + f["rate"] for key, f in floors.items()}
    binding = max(aggregate, key=aggregate.get)
    per_request_floors = [(f["fixed_s"], f["rate"]) for f in floors.values() if f["per_request"]]
    durations = cell.get("request_audio_s") or []
    if per_request_floors and durations:
        cell["rtf_bound_mean"] = sum(
            max(fixed / d + rate for fixed, rate in per_request_floors) for d in durations
        ) / len(durations)

    cell["bound_terms"] = terms
    cell["request_terms"] = per_request
    cell["bound_floors"] = floors
    cell["g_bound"] = {name: entry["device"] for name, entry in terms.items()}
    cell["g_bound_total"] = device_total
    cell["bound_binding"] = binding
    cell["rtf_bound"] = aggregate[binding]
    cell["x_bound"] = batch / aggregate[binding]
    if cell.get("aggregate_rtf"):
        cell["pct_of_sol"] = cell["rtf_bound"] / cell["aggregate_rtf"]
    if cell.get("rtf_bound_mean"):
        cell["pct_of_sol_mean"] = cell["rtf_bound_mean"] / cell["mean_audio_rtf"]

    ttft_terms = [t for t in per_request if stage_order(t["stage"]) == 0]
    if ttft_terms:
        cell["ttft_terms"] = [t["name"] for t in ttft_terms]
        cell["ttft_bound_ms"] = sum(t["latency"] for t in ttft_terms) * 1000.0
        if cell.get("mean_ttft_ms"):
            cell["ttft_pct_of_sol"] = cell["ttft_bound_ms"] / cell["mean_ttft_ms"]
    if per_request:
        cell["ttfp_terms"] = [t["name"] for t in per_request]
        cell["ttfp_bound_ms"] = sum(t["latency"] for t in per_request) * 1000.0
        if cell.get("mean_audio_ttfp_ms"):
            cell["ttfp_pct_of_sol"] = cell["ttfp_bound_ms"] / cell["mean_audio_ttfp_ms"]

    if efficiency and terms:
        device_target = sum(scaled_device(entry, efficiency) for entry in terms.values())
        # Fixed and fill terms stay at the full measured peak: the target says
        # how the streaming rows move, and launch time is not a peak fraction.
        cell["efficiency_projection"] = {
            "compute": efficiency.get("compute", 1.0),
            "memory": efficiency.get("memory", 1.0),
            "device_s": device_target,
            "rtf": (batch * work_s / num_gpus) / mean_d + batch * device_target / num_gpus,
        }
    for name, seconds in (profiled or {}).items():
        if name not in terms:
            raise SystemExit(f"--profiled names {name!r}, which no bound row has")
        terms[name]["profiled_s"] = seconds
        terms[name]["implementation_s"] = max(terms[name]["device"], seconds)
    return cell


def kv_capacity(raws: list[str]) -> dict[str, int]:
    """``STAGE=KV_CACHE_TOKENS,TOKENS_PER_REQUEST[,BLOCK]``: requests one stage's KV cache holds.

    vLLM allocates whole blocks, so a request takes ``ceil(tokens / block)``
    blocks; ``KV_CACHE_TOKENS`` is the ``GPU KV cache size`` startup log line.
    """
    capacity = {}
    for raw in raws:
        stage, _, values = raw.partition("=")
        fields = values.split(",")
        kv_tokens, tokens = int(float(fields[0])), float(fields[1])
        block = int(fields[2]) if len(fields) > 2 else 16
        capacity[stage] = kv_tokens // (math.ceil(tokens / block) * block)
    return capacity


def series_warnings(cells: list[dict], series: list[str]) -> list[str]:
    """Laws a concurrency sweep of one configuration must obey.

    Raising C in a closed loop cannot lower aggregate RTF, audio throughput or
    first-packet latency beyond noise; a violation means the cells differ in
    more than C (warm-up, cache state, configuration) or the run thrashed.
    """
    by_name = {c["name"]: c for c in cells}
    warnings = []
    for raw in series:
        names = raw.split(",")
        missing = [n for n in names if n not in by_name]
        if missing:
            raise SystemExit(f"--series {raw!r}: unknown cells {missing}")
        sweep = sorted((by_name[n] for n in names), key=lambda c: c["concurrency"])
        for low, high in zip(sweep, sweep[1:]):
            pair = f"{low['name']} (C {low['concurrency']}) -> {high['name']} (C {high['concurrency']})"
            for key, label in (
                ("audio_throughput", "audio throughput"),
                ("aggregate_rtf", "aggregate RTF"),
                ("mean_ttft_ms", "mean TTFT"),
                ("mean_audio_ttfp_ms", "mean TTFP"),
            ):
                a, b = low.get(key), high.get(key)
                if a and b and b < 0.95 * a:
                    warnings.append(f"{pair}: {label} fell from {a:.3f} to {b:.3f}; the cells are not one sweep")
    return warnings


def fmt(value, digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def fmt_pct(value) -> str:
    if value is None:
        return "-"
    return f"{value:.1%}" if value < 0.1 else f"{value:.0%}"


def render_latency(cells: list[dict]) -> list[str]:
    if not any(c.get("mean_ttft_ms") or c.get("mean_audio_ttfp_ms") for c in cells):
        return []
    lines = [
        "",
        "| Cell | C | Mean TTFT (ms) | P50 TTFT | TTFT bound | TTFT % of SOL | Mean TTFP (ms) | P50 TTFP "
        "| TTFP bound | TTFP % of SOL |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for c in cells:
        lines.append(
            f"| {c['name']} | {c['concurrency']} | {fmt(c['mean_ttft_ms'], 1)} | {fmt(c['p50_ttft_ms'], 1)} "
            f"| {fmt(c.get('ttft_bound_ms'), 1)} | {fmt_pct(c.get('ttft_pct_of_sol'))} "
            f"| {fmt(c['mean_audio_ttfp_ms'], 1)} | {fmt(c['p50_audio_ttfp_ms'], 1)} "
            f"| {fmt(c.get('ttfp_bound_ms'), 1)} | {fmt_pct(c.get('ttfp_pct_of_sol'))} |"
        )
    return lines


def render(cells: list[dict], extra_warnings: list[str]) -> str:
    lines = [
        "| Cell | C | Done/Fail | Mean RTF | P50 | P99 | Aggregate RTF | C/X | Little residual "
        "| Audio X (s/s) | GPU busy | RTF if no idle | Aggregate RTF bound | % of SOL | Mean RTF bound "
        "| Mean % of SOL |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for c in cells:
        lines.append(
            f"| {c['name']} | {c['concurrency']} | {c['completed']}/{c['failed']} | {fmt(c['mean_audio_rtf'])} "
            f"| {fmt(c['p50_audio_rtf'])} | {fmt(c['p99_audio_rtf'])} | {fmt(c['aggregate_rtf'])} "
            f"| {fmt(c['little_rtf'])} | {fmt(c.get('little_residual'), 3)} | {fmt(c['audio_throughput'], 2)} "
            f"| {fmt(c.get('gpu_busy_share'), 2)} | {fmt(c.get('rtf_floor_no_idle'))} | {fmt(c.get('rtf_bound'))} "
            f"| {fmt_pct(c.get('pct_of_sol'))} | {fmt(c.get('rtf_bound_mean'))} | {fmt_pct(c.get('pct_of_sol_mean'))} |"
        )
    lines += render_latency(cells)
    for c in cells:
        if "measured_fixed_s" in c:
            lines.append("")
            lines.append(
                f"Fixed / per-audio-second split ({c['name']}, T = a + b * d): "
                f"a {c['measured_fixed_s'] * 1000:.0f} ms per request, b {c['measured_rate']:.3f} s per audio-s; "
                f"a is {c['measured_fixed_s'] / c['mean_audio_duration_s'] / c['aggregate_rtf']:.0%} of aggregate RTF"
            )
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
                lines.append(f"  stage {stage}: busy {fmt(share, 2)}, g {fmt(g, 4)} device-s per audio-s")
            positive_g = {stage: g for stage, g in c["stage_device_g"].items() if g is not None and g > 0}
            if positive_g:
                binding = max(positive_g, key=lambda stage: positive_g[stage])
                lines.append(f"  binding stage {binding}; X bound by it: {1.0 / positive_g[binding]:.2f} audio-s/s")
            else:
                lines.append("  X bound unavailable: no positive measured stage time")
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
                    f"device {e['device']:.4f}, launch {e['launch']:.4f} -> {e['bound']} bound; "
                    f"serial at batch 1 {e['serial_1']:.4f}{measured}"
                )
                if "profiled_s" in e:
                    source = "profiled" if e["profiled_s"] >= e["device"] else "roofline"
                    lines.append(
                        f"    profiled {e['profiled_s']:.4f}; implementation floor "
                        f"{e['implementation_s']:.4f} ({source})"
                    )
            if "g_total" in c:
                lines.append(f"  measured g_total {c['g_total']:.4f} vs device bound {c['g_bound_total']:.4f}")
        if proj := c.get("efficiency_projection"):
            lines.append(
                f"Efficiency projection ({c['name']}, compute {proj['compute']:.2f} of peak, "
                f"memory {proj['memory']:.2f} of peak): shared-device aggregate RTF {proj['rtf']:.3f} "
                f"against the bound {c['rtf_bound']:.3f} and measured {fmt(c.get('aggregate_rtf'))}; "
                f"per-audio device {proj['device_s']:.4f} s vs bound {c['g_bound_total']:.4f} s"
            )
        for t in c.get("request_terms") or []:
            if t["stage"] == OUTPUT_STAGE:
                stage = " (gates first packet)"
            else:
                stage = "" if t["stage"] is None else f" (gates stage {t['stage']})"
            lines.append(
                f"  per request, {t['kind']} {t['name']}{stage}: latency {t['latency'] * 1000:.1f} ms, "
                f"device work {t['work'] * 1000:.2f} ms"
            )
        if floors := c.get("bound_floors"):
            lines.append(
                f"End-to-end floors RTF >= a / d + b ({c['name']}, mean d {c['mean_audio_duration_s']:.2f} s):"
            )
            for key, f in floors.items():
                mark = "  <- binding" if key == c["bound_binding"] else ""
                scope = "per request" if f["per_request"] else "aggregate only"
                lines.append(f"  {key} ({scope}): a {f['fixed_s'] * 1000:.1f} ms, b {f['rate']:.4f}{mark}")
            fixed_floor = max(f["fixed_s"] for f in floors.values())
            rate_floor = max(f["rate"] for f in floors.values())
            lines.append(
                f"  aggregate RTF bound {c['rtf_bound']:.3f} (audio X <= {c['x_bound']:.2f} audio-s/s), "
                f"measured at {fmt_pct(c.get('pct_of_sol'))} of SOL; "
                f"mean RTF bound {fmt(c.get('rtf_bound_mean'))}"
            )
            if "measured_fixed_s" in c:
                lines.append(
                    f"  measured a {c['measured_fixed_s'] * 1000:.0f} ms vs largest floor a "
                    f"{fixed_floor * 1000:.1f} ms; measured b {c['measured_rate']:.3f} vs largest floor b "
                    f"{rate_floor:.4f}"
                )
        terms_by_name = {t["name"]: t for t in c.get("request_terms") or []}
        for metric, key, measured in (("TTFT", "ttft", "mean_ttft_ms"), ("TTFP", "ttfp", "mean_audio_ttfp_ms")):
            if f"{key}_bound_ms" not in c:
                continue
            parts = " + ".join(f"{n} {terms_by_name[n]['latency'] * 1000:.1f}" for n in c[f"{key}_terms"])
            lines.append(
                f"{metric} bound ({c['name']}): {parts} = {c[f'{key}_bound_ms']:.1f} ms; "
                f"measured mean {fmt(c.get(measured), 1)} ms, {fmt_pct(c.get(f'{key}_pct_of_sol'))} of SOL"
            )
        for stage, capacity in (c.get("kv_capacity") or {}).items():
            lines.append(f"KV capacity ({c['name']}): stage {stage} holds {capacity} requests at C {c['concurrency']}")
    warnings = list(extra_warnings)
    warnings += [
        f"{c['name']}: Little residual {c['little_residual']:+.1%}; concurrency was not held"
        for c in cells
        if abs(c.get("little_residual") or 0.0) > 0.10
    ]
    warnings += [f"{c['name']}: {c['failed']} failed requests" for c in cells if c["failed"]]
    for c in cells:
        if c.get("gpu_samples") == 0:
            warnings.append(f"{c['name']}: no utilisation samples in the measured window; busy share is unavailable")
        for stage, samples in (c.get("stage_device_samples") or {}).items():
            if not samples:
                warnings.append(
                    f"{c['name']}: stage {stage} has no utilisation samples in the measured window; "
                    "busy share and g_i are unavailable"
                )
            elif not c["stage_util_trimmed"][stage]:
                warnings.append(
                    f"{c['name']}: stage {stage} utilisation samples could not be trimmed to the measured "
                    "window; check the sample timestamps and result date"
                )
        for metric, key, measured in (("TTFT", "ttft", "mean_ttft_ms"), ("TTFP", "ttfp", "mean_audio_ttfp_ms")):
            if (c.get(f"{key}_pct_of_sol") or 0.0) > 1.05:
                warnings.append(
                    f"{c['name']}: {metric} bound {c[f'{key}_bound_ms']:.1f} ms is above measured mean "
                    f"{c[measured]:.1f} ms; a per-request term overcounts"
                )
        if c["mean_ttft_ms"] and c["mean_audio_ttfp_ms"] and c["mean_audio_ttfp_ms"] < c["mean_ttft_ms"]:
            warnings.append(
                f"{c['name']}: mean TTFP {c['mean_audio_ttfp_ms']:.1f} ms is below mean TTFT "
                f"{c['mean_ttft_ms']:.1f} ms; audio cannot precede the first Thinker token, check the timers"
            )
        for stage, capacity in (c.get("kv_capacity") or {}).items():
            if capacity < c["concurrency"]:
                warnings.append(
                    f"{c['name']}: stage {stage} KV cache holds {capacity} requests, below C {c['concurrency']}; "
                    "the cell measures queueing or preemption, not compute"
                )
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
    warnings += [
        f"{c['name']}: only {c['audio_coverage']:.0%} of requests carry stage audio durations; "
        "the fixed / per-audio split and mean RTF bound use that subset"
        for c in cells
        if c.get("audio_coverage", 1.0) < 0.9
    ]
    warnings += [
        f"{c['name']}: measured split gives a {c['measured_fixed_s'] * 1000:.0f} ms, b {c['measured_rate']:.3f}; "
        "T = a + b * d does not fit, so do not read the split"
        for c in cells
        if "measured_fixed_s" in c and (c["measured_fixed_s"] < 0 or c["measured_rate"] < 0)
    ]
    for c in cells:
        pmon_sum = sum((c.get("pmon_stage_share") or {}).values())
        busy = c.get("gpu_busy_share")
        if pmon_sum and busy and abs(pmon_sum / busy - 1.0) > 0.2:
            warnings.append(
                f"{c['name']}: pmon stage sum {pmon_sum:.2f} vs GPU busy {busy:.2f}; "
                "pmon dropped readings, use stage proportions only"
            )
        proj = c.get("efficiency_projection")
        if proj and c.get("aggregate_rtf") and proj["rtf"] > 1.05 * c["aggregate_rtf"]:
            warnings.append(
                f"{c['name']}: efficiency projection {proj['rtf']:.3f} is above measured aggregate RTF "
                f"{c['aggregate_rtf']:.3f}; the cell already beats this utilization target"
            )
        for name, e in (c.get("bound_terms") or {}).items():
            if e.get("profiled_s") is not None and e["device"] and e["profiled_s"] < 0.95 * e["device"]:
                warnings.append(
                    f"{c['name']}: {name} profiled {e['profiled_s']:.4f} s per audio-s is below its roofline "
                    f"{e['device']:.4f}; the work count is too high, the peak too low, or the profile covers less work"
                )
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
        help="NAME[@STAGE]=WEIGHT_GIB,TOKENS_PER_AUDIO_S[,KV_GIB_PER_SEQ[,GFLOP_PER_TOKEN,DTYPE]]",
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
    parser.add_argument(
        "--fixed-bound",
        action="append",
        default=[],
        help="NAME[@STAGE]=TFLOP,DTYPE[,WEIGHT_GIB[,KERNELS,MODE]] per request, adding work (prefill, encoders)",
    )
    parser.add_argument(
        "--fill-bound",
        action="append",
        default=[],
        help="NAME[@STAGE]=TFLOP,DTYPE[,WEIGHT_GIB[,KERNELS,MODE]] per request on the critical path before STAGE "
        "(or before the first packet for @out), whose work is already in the per-audio terms (first chunk)",
    )
    parser.add_argument(
        "--kv-capacity",
        action="append",
        default=[],
        help="STAGE=KV_CACHE_TOKENS,TOKENS_PER_REQUEST[,BLOCK]; warns when a stage's KV cache holds fewer than C",
    )
    parser.add_argument(
        "--series",
        action="append",
        default=[],
        help="NAME,NAME,...: cells of one configuration at rising C; checks the sweep is monotone",
    )
    parser.add_argument(
        "--efficiency-target",
        help="compute=F,memory=F with 0 < F <= 1: project the shared-device floor at that fraction of the "
        "measured peak. A projection, printed beside the bound, never instead of it",
    )
    parser.add_argument(
        "--profiled",
        action="append",
        default=[],
        help="NAME=SECONDS_PER_AUDIO_S at this cell's batch: a profiled duration of that row's implementation. "
        "The implementation floor is max(roofline, profiled) and does not enter the bound",
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
        busy, samples, trimmed = device_busy_share(Path(util_path), measured_window_of(cell), args.npu_util_column)
        cell.setdefault("stage_device_share", {})[stage] = busy
        cell.setdefault("stage_device_samples", {})[stage] = samples
        cell.setdefault("stage_util_trimmed", {})[stage] = trimmed
        cell.setdefault("stage_device_g", {})[stage] = (
            busy / cell["audio_throughput"] if busy is not None and cell["audio_throughput"] else None
        )
    for raw in args.pmon:
        name, _, spec = raw.partition("=")
        pmon_path, server_log = spec.split(":")
        cell = by_name[name]
        shares = pmon_stage_shares(Path(pmon_path), Path(server_log))
        cell["pmon_stage_share"] = shares
        cell["pmon_stage_g"] = {stage: share / cell["audio_throughput"] for stage, share in shares.items()}
    if args.peaks:
        peaks = json.loads(Path(args.peaks).read_text())
        specs = {
            "decode": args.decode_bound,
            "compute": args.compute_bound,
            "launch": args.launch_bound,
            "fixed": args.fixed_bound,
            "fill": args.fill_bound,
        }
        profiled = {}
        for raw in args.profiled:
            name, _, value = raw.partition("=")
            if not value:
                parser.error(f"--profiled {raw!r} must be NAME=SECONDS_PER_AUDIO_S")
            profiled[name] = float(value)
        cells = [
            physical_bounds(c, peaks, specs, args.num_gpus, efficiency_target(args.efficiency_target), profiled)
            for c in cells
        ]

    capacity = kv_capacity(args.kv_capacity)
    for cell in cells:
        if capacity:
            cell["kv_capacity"] = capacity
    print(render(cells, series_warnings(cells, args.series)))
    if args.json:
        Path(args.json).write_text(json.dumps(cells, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
