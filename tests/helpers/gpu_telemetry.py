# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""GPU clock / utilization / throttle telemetry for perf CI cases.

Samples ``nvidia-smi`` in a background thread while a benchmark case runs and
returns a compact summary that is written next to the latency metrics. The goal
is to make a perf-gate failure classifiable after the fact: a case that regressed
while the GPU sat at max clock with ~95% utilization is a code change; one that
regressed with clocks pinned but utilization at 55% is host-bound on that node.

Pure stdlib. On machines without ``nvidia-smi`` (CPU, ROCm, NPU) the summary is
``{"available": False, "reason": ...}`` and nothing else happens.

Env:
    DFX_PERF_GPU_TELEMETRY_INTERVAL  seconds between samples (default 2.0, 0 disables)
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any

ENV_INTERVAL = "DFX_PERF_GPU_TELEMETRY_INTERVAL"
DEFAULT_INTERVAL_S = 2.0
# Upper bound for one nvidia-smi call; stop() waits at least this long so no query outlives the case.
QUERY_TIMEOUT_S = 10.0

# utilization.gpu >= this counts as "the GPU was doing work" for the busy stats.
BUSY_UTIL_PCT = 50

# Newer drivers (>= 555) renamed the field; older ones only know the old name.
_THROTTLE_FIELDS = ("clocks_event_reasons.active", "clocks_throttle_reasons.active")

# NVML nvmlClocksThrottleReasons_t bits. 0x1 (gpu_idle) is normal between requests.
THROTTLE_REASON_BITS: dict[int, str] = {
    0x1: "gpu_idle",
    0x2: "applications_clocks_setting",
    0x4: "sw_power_cap",
    0x8: "hw_slowdown",
    0x10: "sync_boost",
    0x20: "sw_thermal_slowdown",
    0x40: "hw_thermal_slowdown",
    0x80: "hw_power_brake_slowdown",
    0x100: "display_clock_setting",
}
_BENIGN_MASK = 0x1

_BASE_FIELDS = ("index", "clocks.sm", "clocks.max.sm", "clocks.mem", "power.draw", "temperature.gpu", "utilization.gpu")


def decode_throttle_reasons(mask: int) -> list[str]:
    """Return the NVML reason names set in *mask*, excluding the benign gpu_idle bit."""
    return [name for bit, name in sorted(THROTTLE_REASON_BITS.items()) if mask & bit and bit != _BENIGN_MASK]


def _to_float(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        return None


@dataclass
class GpuSample:
    index: int
    sm_mhz: float | None
    sm_max_mhz: float | None
    mem_mhz: float | None
    power_w: float | None
    temp_c: float | None
    util_pct: float | None
    throttle_mask: int | None


def parse_sample_line(line: str, *, has_throttle: bool) -> GpuSample | None:
    """Parse one ``--format=csv,noheader,nounits`` row. Returns None for malformed rows."""
    parts = [p.strip() for p in line.split(",")]
    expected = len(_BASE_FIELDS) + (1 if has_throttle else 0)
    if len(parts) != expected:
        return None
    try:
        index = int(parts[0])
    except ValueError:
        return None
    mask: int | None = None
    if has_throttle:
        try:
            mask = int(parts[7], 16)
        except ValueError:
            mask = None
    return GpuSample(
        index=index,
        sm_mhz=_to_float(parts[1]),
        sm_max_mhz=_to_float(parts[2]),
        mem_mhz=_to_float(parts[3]),
        power_w=_to_float(parts[4]),
        temp_c=_to_float(parts[5]),
        util_pct=_to_float(parts[6]),
        throttle_mask=mask,
    )


def _stats(values: list[float]) -> dict[str, float]:
    return {"min": min(values), "mean": sum(values) / len(values), "max": max(values)}


def summarize_samples(samples: list[GpuSample]) -> dict[str, Any]:
    """Per-GPU summary of a sample list. Busy stats use samples with util >= BUSY_UTIL_PCT."""
    per_gpu: dict[int, list[GpuSample]] = {}
    for s in samples:
        per_gpu.setdefault(s.index, []).append(s)

    out: dict[str, Any] = {}
    for index, rows in sorted(per_gpu.items()):
        util = [r.util_pct for r in rows if r.util_pct is not None]
        busy = [r for r in rows if r.util_pct is not None and r.util_pct >= BUSY_UTIL_PCT]
        util_busy = [u for u in util if u >= BUSY_UTIL_PCT]
        sm_busy = [r.sm_mhz for r in busy if r.sm_mhz is not None]
        mem_busy = [r.mem_mhz for r in busy if r.mem_mhz is not None]
        sm_max = next((r.sm_max_mhz for r in rows if r.sm_max_mhz), None)
        power = [r.power_w for r in rows if r.power_w is not None]
        temp = [r.temp_c for r in rows if r.temp_c is not None]

        reasons: dict[str, int] = {}
        for r in rows:
            if r.throttle_mask is None:
                continue
            for name in decode_throttle_reasons(r.throttle_mask):
                reasons[name] = reasons.get(name, 0) + 1

        entry: dict[str, Any] = {
            "samples": len(rows),
            "busy_samples": len(busy),
            "util_pct": _stats(util) if util else None,
            "util_pct_busy_mean": (sum(util_busy) / len(util_busy)) if util_busy else None,
            "sm_mhz_busy": _stats(sm_busy) if sm_busy else None,
            "sm_max_mhz": sm_max,
            "sm_ratio_busy_min": (min(sm_busy) / sm_max) if sm_busy and sm_max else None,
            "mem_mhz_busy": _stats(mem_busy) if mem_busy else None,
            "power_w_max": max(power) if power else None,
            "temp_c_max": max(temp) if temp else None,
            "throttle_reasons": reasons,
        }
        out[str(index)] = entry
    return out


def format_summary_line(summary: dict[str, Any]) -> str:
    """One log line per case, e.g. for the pytest console."""
    if not summary.get("available"):
        return f"gpu_telemetry: unavailable ({summary.get('reason', 'unknown')})"
    parts = []
    for index, g in summary.get("gpus", {}).items():
        sm = g.get("sm_mhz_busy")
        util = g.get("util_pct_busy_mean")
        ratio = g.get("sm_ratio_busy_min")
        reasons = g.get("throttle_reasons") or {}
        sm_txt = f"sm {sm['min']:.0f}/{sm['mean']:.0f}/{sm['max']:.0f}MHz" if sm else "sm n/a"
        ratio_txt = f" ({ratio:.2f}x max)" if ratio is not None else ""
        util_txt = f"util_busy {util:.0f}%" if util is not None else "util n/a"
        busy_txt = f"{g['busy_samples']}/{g['samples']} busy"
        throttle_txt = ",".join(f"{k}:{v}" for k, v in sorted(reasons.items())) or "none"
        parts.append(f"gpu{index}: {sm_txt}{ratio_txt} {util_txt} {busy_txt} throttle={throttle_txt}")
    return "gpu_telemetry: " + " | ".join(parts)


def _visible_device_arg() -> list[str]:
    """Restrict nvidia-smi to CUDA_VISIBLE_DEVICES if it is a plain index list."""
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not raw:
        return []
    ids = [x.strip() for x in raw.split(",")]
    if all(x.isdigit() for x in ids):
        return ["-i", ",".join(ids)]
    return []  # UUIDs / MIG ids: let nvidia-smi report everything


def interval_from_env(default: float = DEFAULT_INTERVAL_S) -> float:
    raw = os.environ.get(ENV_INTERVAL)
    if raw is None or raw.strip() == "":
        return default
    value = _to_float(raw)
    if value is None or not math.isfinite(value) or value < 0:
        return default
    return value


@dataclass
class GpuTelemetrySampler:
    """Background ``nvidia-smi`` sampler. Use as a context manager around a benchmark case.

    ``summary()`` is safe to call after ``__exit__``; before that it reflects samples so far.
    """

    interval_s: float = field(default_factory=interval_from_env)
    _samples: list[GpuSample] = field(default_factory=list, init=False)
    _stop: threading.Event = field(default_factory=threading.Event, init=False)
    _thread: threading.Thread | None = field(default=None, init=False)
    _throttle_field: str | None = field(default=None, init=False)
    _available: bool = field(default=False, init=False)
    _reason: str = field(default="", init=False)
    _started_at: float = field(default=0.0, init=False)
    _ended_at: float = field(default=0.0, init=False)

    # -- probing --------------------------------------------------------------

    def _run_query(self, fields: str) -> str | None:
        cmd = ["nvidia-smi", *_visible_device_arg(), f"--query-gpu={fields}", "--format=csv,noheader,nounits"]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=QUERY_TIMEOUT_S)
        except (OSError, subprocess.SubprocessError):
            return None
        if proc.returncode != 0:
            return None
        return proc.stdout

    def _probe(self) -> None:
        if self.interval_s <= 0:
            self._reason = f"disabled ({ENV_INTERVAL}=0)"
            return
        if shutil.which("nvidia-smi") is None:
            self._reason = "nvidia-smi not found"
            return
        for cand in _THROTTLE_FIELDS:
            out = self._run_query(",".join((*_BASE_FIELDS, cand)))
            if out and any(parse_sample_line(ln, has_throttle=True) for ln in out.splitlines()):
                self._throttle_field = cand
                self._available = True
                return
        out = self._run_query(",".join(_BASE_FIELDS))
        if out and any(parse_sample_line(ln, has_throttle=False) for ln in out.splitlines()):
            self._available = True  # clocks/util only, no throttle bitmask on this driver
            return
        self._reason = "nvidia-smi query failed"

    # -- sampling -------------------------------------------------------------

    def _query_fields(self) -> str:
        fields = list(_BASE_FIELDS)
        if self._throttle_field:
            fields.append(self._throttle_field)
        return ",".join(fields)

    def _sample_once(self) -> None:
        out = self._run_query(self._query_fields())
        if not out:
            return
        has_throttle = self._throttle_field is not None
        for ln in out.splitlines():
            sample = parse_sample_line(ln, has_throttle=has_throttle)
            if sample is not None:
                self._samples.append(sample)

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._sample_once()
            self._stop.wait(self.interval_s)

    def start(self) -> GpuTelemetrySampler:
        self._probe()
        self._started_at = time.time()
        if self._available:
            self._thread = threading.Thread(target=self._loop, name="gpu-telemetry", daemon=True)
            self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            # A query may be mid-flight; wait long enough for it to finish or time out.
            self._thread.join(timeout=QUERY_TIMEOUT_S + self.interval_s + 1.0)
            self._thread = None
        self._ended_at = time.time()

    def __enter__(self) -> GpuTelemetrySampler:
        return self.start()

    def __exit__(self, *_: object) -> None:
        self.stop()

    # -- output ---------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        if not self._available:
            return {"available": False, "reason": self._reason or "not started"}
        end = self._ended_at or time.time()
        return {
            "available": True,
            "interval_s": self.interval_s,
            "duration_s": round(end - self._started_at, 1) if self._started_at else None,
            "throttle_field": self._throttle_field,
            "gpus": summarize_samples(list(self._samples)),
        }


__all__ = [
    "BUSY_UTIL_PCT",
    "ENV_INTERVAL",
    "GpuSample",
    "GpuTelemetrySampler",
    "QUERY_TIMEOUT_S",
    "decode_throttle_reasons",
    "format_summary_line",
    "interval_from_env",
    "parse_sample_line",
    "summarize_samples",
]
