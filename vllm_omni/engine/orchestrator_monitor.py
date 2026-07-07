"""Optional orchestrator window monitor (diagnostic only).

Enable with:
  --enable-orch-monitor

Optional output path override:
  export VLLM_OMNI_ORCH_MONITOR_PATH=/path/to/monitor.json

Each 1s window records:
  - duration_s, loop_idle, loop_active (orchestrator poll-loop counts)
  - per-replica outputs_queue_size (MP client outputs_queue backlog)
  - per-replica inflight (requests currently bound/routed to the replica)

See docs/contributing/profiling.md (Orchestrator Monitor).
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Protocol, TypedDict

from vllm.logger import init_logger

logger = init_logger(__name__)

_DEFAULT_OUTPUT_PREFIX = "vllm_omni_orch_monitor"
_WINDOW_S = 1.0

ReplicaSample = tuple[int, int]
ReplicaSampler = Callable[[], dict[str, ReplicaSample]]


class _ReplicaSeries(TypedDict):
    outputs_queue_size: list[int]
    inflight: list[int]


def replica_key(stage_id: int, replica_id: int) -> str:
    return f"stage={stage_id},replica={replica_id}"


def _resolve_output_path() -> str:
    env_path = os.environ.get("VLLM_OMNI_ORCH_MONITOR_PATH")
    if env_path:
        return env_path
    timestamp = time.strftime("%m%d%H%M", time.localtime())
    return str(Path.cwd() / f"{_DEFAULT_OUTPUT_PREFIX}_{timestamp}.json")


class OrchestratorMonitorBase(Protocol):
    def register_replica(self, stage_id: int, replica_id: int) -> None: ...

    def note_loop(self, *, idle: bool) -> None: ...

    def flush(self) -> None: ...


class _NullOrchestratorMonitor:
    def register_replica(self, stage_id: int, replica_id: int) -> None:
        return

    def note_loop(self, *, idle: bool) -> None:
        return

    def flush(self) -> None:
        return


_NULL_ORCH_MONITOR = _NullOrchestratorMonitor()


def create_orch_monitor(
    *,
    enabled: bool,
    replica_sampler: ReplicaSampler,
) -> OrchestratorMonitorBase:
    if not enabled:
        return _NULL_ORCH_MONITOR
    return OrchestratorMonitor(replica_sampler=replica_sampler)


class OrchestratorMonitor:
    def __init__(self, *, replica_sampler: ReplicaSampler) -> None:
        self._output_path = _resolve_output_path()
        self._flushed = False
        self._replica_sampler = replica_sampler
        self._window_start_mono = time.monotonic()
        self._loop_idle = 0
        self._loop_active = 0
        self._duration_s: list[float] = []
        self._loop_idle_buf: list[int] = []
        self._loop_active_buf: list[int] = []
        self._replicas: dict[str, _ReplicaSeries] = {}

    def register_replica(self, stage_id: int, replica_id: int) -> None:
        self._ensure_replica(replica_key(stage_id, replica_id))

    def note_loop(self, *, idle: bool) -> None:
        now_mono = time.monotonic()
        self._roll_window_if_needed(now_mono)
        if idle:
            self._loop_idle += 1
        else:
            self._loop_active += 1

    def flush(self) -> None:
        if self._flushed:
            return
        self._roll_window(time.monotonic())
        self._flushed = True
        payload = {
            "configured_window_s": _WINDOW_S,
            "windows": {
                "duration_s": self._duration_s,
                "loop_idle": self._loop_idle_buf,
                "loop_active": self._loop_active_buf,
            },
            "replicas": dict(sorted(self._replicas.items())),
        }
        try:
            output_path = Path(self._output_path).expanduser()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, sort_keys=True)
                f.write("\n")
            self._log_summary(str(output_path))
        except OSError:
            logger.exception("[OrchestratorMonitor] Failed to write log to %s", self._output_path)

    def _ensure_replica(self, key: str) -> None:
        if key in self._replicas:
            return
        # Backfill zeros for windows recorded before this replica appeared.
        n = len(self._duration_s)
        self._replicas[key] = {
            "outputs_queue_size": [0] * n,
            "inflight": [0] * n,
        }

    def _roll_window_if_needed(self, now_mono: float) -> None:
        if now_mono - self._window_start_mono >= _WINDOW_S:
            self._roll_window(now_mono)

    def _roll_window(self, now_mono: float) -> None:
        try:
            sampled = self._replica_sampler()
        except Exception:
            logger.exception("[OrchestratorMonitor] replica sampling failed")
            sampled = {}
        self._duration_s.append(max(now_mono - self._window_start_mono, 1e-9))
        self._loop_idle_buf.append(self._loop_idle)
        self._loop_active_buf.append(self._loop_active)
        for key, series in self._replicas.items():
            outputs_qsize, inflight = sampled.get(key, (0, 0))
            series["outputs_queue_size"].append(outputs_qsize)
            series["inflight"].append(inflight)

        self._loop_idle = 0
        self._loop_active = 0
        self._window_start_mono = now_mono

    def _log_summary(self, output_path: str) -> None:
        loop_idle = sum(self._loop_idle_buf)
        loop_active = sum(self._loop_active_buf)
        loop_total = loop_idle + loop_active
        loop_active_pct = (float(loop_active) / float(loop_total) * 100.0) if loop_total else 0.0
        logger.info(
            "[OrchestratorMonitor] wrote %s windows=%d duration=%.3fs loop_active=%.1f%%",
            output_path,
            len(self._duration_s),
            sum(self._duration_s),
            loop_active_pct,
        )
        for key, series in sorted(self._replicas.items()):
            queue_values = series["outputs_queue_size"]
            inflight_values = series["inflight"]
            queue_avg = sum(queue_values) / float(len(queue_values)) if queue_values else 0.0
            queue_max = max(queue_values) if queue_values else 0
            inflight_avg = sum(inflight_values) / float(len(inflight_values)) if inflight_values else 0.0
            inflight_max = max(inflight_values) if inflight_values else 0
            logger.info(
                "[OrchestratorMonitor] %s outputs_queue_avg=%.2f outputs_queue_max=%d "
                "inflight_avg=%.2f inflight_max=%d",
                key,
                queue_avg,
                queue_max,
                inflight_avg,
                inflight_max,
            )
