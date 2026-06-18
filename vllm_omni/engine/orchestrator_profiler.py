"""Optional orchestrator window profiler (diagnostic only).

Enable with:
  --enable-orch-profiler
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from vllm.logger import init_logger

logger = init_logger(__name__)

_DEFAULT_OUTPUT_PREFIX = "vllm_omni_orch_profile"
_WINDOW_S = 1.0


def _default_output_path() -> str:
    timestamp = time.strftime("%m%d%H%M", time.localtime())
    return str(Path.cwd() / f"{_DEFAULT_OUTPUT_PREFIX}_{timestamp}.json")


@dataclass
class _ReplicaWindow:
    queue_size_sum: float = 0.0
    queue_size_samples: int = 0
    queue_size_max: int = 0
    outputs: int = 0
    finished_outputs: int = 0


class OrchestratorProfiler:
    def __init__(
        self,
        *,
        enabled: bool = False,
    ) -> None:
        self._enabled = bool(enabled)
        self._output_path = _default_output_path()
        self._dumped = False
        self._started_at_wall = time.time()
        self._window_start_wall = self._started_at_wall
        self._window_start_mono = time.monotonic()
        self._loop_iterations = 0
        self._loop_idle_iterations = 0
        self._active_requests = 0
        self._replica_windows: dict[str, _ReplicaWindow] = {}
        self._replica_keys: set[str] = set()
        self._windows: dict[str, list[float | int]] = {
            "start_ts": [],
            "end_ts": [],
            "duration_s": [],
            "active_requests": [],
            "loop_iterations": [],
            "loop_idle": [],
            "loop_running": [],
        }
        self._replicas: dict[str, dict[str, list[float | int]]] = {}

    @property
    def enabled(self) -> bool:
        return self._enabled

    @staticmethod
    def _replica_key(stage_id: int, replica_id: int) -> str:
        return f"stage={stage_id},replica={replica_id}"

    def register_replica(self, stage_id: int, replica_id: int) -> None:
        if not self._enabled:
            return
        self._ensure_replica(self._replica_key(stage_id, replica_id))

    def note_loop(self, *, idle: bool, active_requests: int) -> None:
        if not self._enabled:
            return
        self._roll_window_if_needed(time.monotonic())
        self._loop_iterations += 1
        if idle:
            self._loop_idle_iterations += 1
        self._active_requests = int(active_requests)

    def record_queue(self, stage_id: int, replica_id: int, queue_size: int) -> None:
        if not self._enabled:
            return
        self._roll_window_if_needed(time.monotonic())
        key = self._replica_key(stage_id, replica_id)
        self._ensure_replica(key)
        window = self._replica_windows.setdefault(key, _ReplicaWindow())
        queue_size = max(int(queue_size), 0)
        window.queue_size_sum += float(queue_size)
        window.queue_size_samples += 1
        window.queue_size_max = max(window.queue_size_max, queue_size)

    def record_outputs(self, stage_id: int, replica_id: int, outputs: list[object]) -> None:
        if not self._enabled:
            return
        self._roll_window_if_needed(time.monotonic())
        key = self._replica_key(stage_id, replica_id)
        self._ensure_replica(key)
        window = self._replica_windows.setdefault(key, _ReplicaWindow())
        window.outputs += len(outputs)
        window.finished_outputs += sum(1 for output in outputs if bool(getattr(output, "finished", False)))

    def _ensure_replica(self, key: str) -> None:
        if key in self._replica_keys:
            return
        self._replica_keys.add(key)
        window_count = len(self._windows["start_ts"])
        self._replicas[key] = {
            "queue_size_avg": [0.0] * window_count,
            "queue_size_max": [0] * window_count,
            "outputs": [0] * window_count,
            "finished_outputs": [0] * window_count,
        }

    def _roll_window_if_needed(self, now_mono: float) -> None:
        if now_mono - self._window_start_mono >= _WINDOW_S:
            self._roll_window(now_mono)

    def _roll_window(self, now_mono: float) -> None:
        end_wall = time.time()
        duration_s = max(now_mono - self._window_start_mono, 1e-9)
        self._windows["start_ts"].append(self._window_start_wall)
        self._windows["end_ts"].append(end_wall)
        self._windows["duration_s"].append(duration_s)
        self._windows["active_requests"].append(self._active_requests)
        self._windows["loop_iterations"].append(self._loop_iterations)
        self._windows["loop_idle"].append(self._loop_idle_iterations)
        self._windows["loop_running"].append(self._loop_iterations - self._loop_idle_iterations)

        for key in sorted(self._replica_keys):
            metrics = self._replicas[key]
            window = self._replica_windows.get(key, _ReplicaWindow())
            queue_avg = window.queue_size_sum / float(window.queue_size_samples) if window.queue_size_samples else 0.0
            metrics["queue_size_avg"].append(queue_avg)
            metrics["queue_size_max"].append(window.queue_size_max)
            metrics["outputs"].append(window.outputs)
            metrics["finished_outputs"].append(window.finished_outputs)

        self._replica_windows.clear()
        self._loop_iterations = 0
        self._loop_idle_iterations = 0
        self._window_start_wall = end_wall
        self._window_start_mono = now_mono

    def dump(self) -> None:
        if not self._enabled:
            return
        output_path = self._output_path
        if self._dumped:
            return
        self._roll_window(time.monotonic())
        self._dumped = True
        payload = {
            "started_at": self._started_at_wall,
            "ended_at": time.time(),
            "configured_window_s": _WINDOW_S,
            "windows": self._windows,
            "replicas": self._replicas,
        }
        summary = self._build_summary()
        try:
            parent = Path(output_path).expanduser().parent
            parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, sort_keys=True)
                f.write("\n")
            self._log_summary(output_path, summary)
        except OSError:
            logger.exception("[OrchestratorProfiler] Failed to write profile to %s", output_path)

    def flush(self) -> None:
        self.dump()

    def _build_summary(self) -> dict[str, float | int]:
        duration_s = sum(float(x) for x in self._windows["duration_s"])
        loop_iterations = sum(int(x) for x in self._windows["loop_iterations"])
        loop_idle = sum(int(x) for x in self._windows["loop_idle"])
        loop_running = sum(int(x) for x in self._windows["loop_running"])
        active_values = [int(x) for x in self._windows["active_requests"]]
        return {
            "windows": len(self._windows["duration_s"]),
            "duration_s": duration_s,
            "loop_iterations": loop_iterations,
            "loop_idle": loop_idle,
            "loop_running": loop_running,
            "loop_running_pct": (float(loop_running) / float(loop_iterations) * 100.0) if loop_iterations else 0.0,
            "active_requests_avg": (sum(active_values) / float(len(active_values))) if active_values else 0.0,
            "active_requests_max": max(active_values) if active_values else 0,
        }

    def _log_summary(self, output_path: str, summary: dict[str, float | int]) -> None:
        logger.info(
            "[OrchestratorProfiler] wrote %s windows=%d duration=%.3fs loop_running=%.1f%% "
            "active_avg=%.2f active_max=%d",
            output_path,
            int(summary["windows"]),
            float(summary["duration_s"]),
            float(summary["loop_running_pct"]),
            float(summary["active_requests_avg"]),
            int(summary["active_requests_max"]),
        )
        for key, metrics in sorted(self._replicas.items()):
            queue_avg_values = [float(x) for x in metrics["queue_size_avg"]]
            queue_max_values = [int(x) for x in metrics["queue_size_max"]]
            outputs = sum(int(x) for x in metrics["outputs"])
            finished_outputs = sum(int(x) for x in metrics["finished_outputs"])
            queue_avg = sum(queue_avg_values) / float(len(queue_avg_values)) if queue_avg_values else 0.0
            queue_max = max(queue_max_values) if queue_max_values else 0
            logger.info(
                "[OrchestratorProfiler] %s queue_avg=%.2f queue_max=%d outputs=%d finished_outputs=%d",
                key,
                queue_avg,
                queue_max,
                outputs,
                finished_outputs,
            )


def get_orchestrator_profiler(
    *,
    enabled: bool = False,
) -> OrchestratorProfiler:
    return OrchestratorProfiler(enabled=enabled)
