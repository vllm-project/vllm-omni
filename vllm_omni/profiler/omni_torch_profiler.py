# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import os
import subprocess
from datetime import datetime
from typing import Any, Literal

import torch
from typing_extensions import override
from vllm.config import ProfilerConfig
from vllm.config.profiler import _is_uri_path
from vllm.logger import init_logger
from vllm.profiler.wrapper import WorkerProfiler

logger = init_logger(__name__)

# NPU has its custom profiler
TorchProfilerActivity = Literal["CPU", "CUDA", "XPU", "NPU", "MUSA"]
TorchProfilerActivityMap = {
    "CPU": torch.profiler.ProfilerActivity.CPU,
    "CUDA": torch.profiler.ProfilerActivity.CUDA,
    "XPU": torch.profiler.ProfilerActivity.XPU,
    "MUSA": torch.profiler.ProfilerActivity.CUDA,
}


class OmniTorchProfilerWrapper(WorkerProfiler):
    """Base torch profiler wrapper with platform-agnostic functionality.

    Provides common profiler features:
    - Custom trace file naming with stage/rank info
    - Background gzip compression via subprocess
    - Returns trace file paths from get_results() for orchestrator collection

    Subclasses can override hook methods for platform-specific behavior:
    - _get_default_activities(): Return default activities for the platform
    - _create_profiler(): Create platform-specific profiler instance
    - _on_stop_hook(): Handle platform-specific post-stop logic
    """

    def __init__(
        self,
        profiler_config: ProfilerConfig,
        worker_name: str,
        local_rank: int,
        activities: list[TorchProfilerActivity] | None = None,
    ) -> None:
        super().__init__(profiler_config)

        if activities is None:
            activities = self._get_default_activities()

        self.local_rank = local_rank
        self.profiler_config = profiler_config
        self._worker_name = worker_name
        self._trace_dir = profiler_config.torch_profiler_dir
        self._use_gzip = profiler_config.torch_profiler_use_gzip
        self._trace_filename: str | None = None
        self._trace_path: str | None = None
        self._table_path: str | None = None

        self._activities = activities
        self._session_dir: str | None = None
        self._artifact_paths: dict[str, str | None] = {}
        self._memory_history_enabled = False
        self._memory_history_backend: str | None = None
        self._memory_history_module = None

        if local_rank in (None, 0):
            logger.info_once(
                "Omni torch profiling enabled. Traces will be saved to: %s",
                self._trace_dir,
                scope="local",
            )

        self.dump_cpu_time_total = "CPU" in activities and len(activities) == 1
        self.profiler = self._create_profiler(profiler_config, activities)

    def _rank(self) -> int:
        return 0 if self.local_rank is None else self.local_rank

    def _get_default_activities(self) -> list[TorchProfilerActivity]:
        """Get default activities for this platform.

        Override in subclasses to provide platform-specific defaults.
        """
        return ["CPU", "CUDA"]

    def _create_profiler(
        self,
        profiler_config: ProfilerConfig,
        activities: list[TorchProfilerActivity],
    ):
        """Create the profiler instance.

        Override in subclasses for platform-specific profiler creation.
        """
        return torch.profiler.profile(
            activities=[TorchProfilerActivityMap[a] for a in activities],
            record_shapes=profiler_config.torch_profiler_record_shapes,
            profile_memory=profiler_config.torch_profiler_with_memory,
            with_stack=profiler_config.torch_profiler_with_stack,
            with_flops=profiler_config.torch_profiler_with_flops,
            on_trace_ready=self._on_trace_ready,
        )

    def set_trace_filename(self, filename: str) -> None:
        """Set the trace filename before starting profiling.

        Args:
            filename: Base filename without extension or rank suffix.
                      e.g. "stage_0_llm_1234567890"
                      Can also be a full path (e.g. from diffusion engine).
        """
        self._trace_filename = filename
        self._session_dir = None
        self._ensure_session_dir()

    def _ensure_session_dir(self) -> str:
        """Create one timestamped directory for this profiling run."""
        if self._session_dir is not None:
            os.makedirs(self._session_dir, exist_ok=True)
            return self._session_dir

        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        base_name = self._trace_filename or self._worker_name

        if os.path.dirname(base_name):
            parent_dir = os.path.dirname(base_name)
            leaf_name = os.path.basename(base_name)
            session_name = f"{ts}_{leaf_name}"
            self._session_dir = os.path.join(parent_dir, session_name)
        else:
            session_name = f"{ts}_{base_name}"
            self._session_dir = os.path.join(self._trace_dir, session_name)

        os.makedirs(self._session_dir, exist_ok=True)
        self._artifact_paths["session_dir"] = self._session_dir
        return self._session_dir

    def _artifact_path(self, stem: str, suffix: str) -> str:
        """Build artifact path under the session directory."""
        return os.path.join(
            self._ensure_session_dir(),
            f"{stem}_rank{self._rank()}{suffix}",
        )

    def _write_text_artifact(self, name: str, content: str) -> str:
        path = self._artifact_path(name, ".txt")
        with open(path, "w") as f:
            f.write(content)
        self._artifact_paths[name] = path
        return path

    def _has_cuda_like_activity(self) -> bool:
        return any(a in self._activities for a in ("CUDA", "MUSA"))

    def _get_time_sort_key(self) -> str:
        if self._has_cuda_like_activity():
            return "self_cuda_time_total"
        return "self_cpu_time_total"

    def _on_trace_ready(self, prof) -> None:
        """Custom trace handler: export chrome trace with omni naming."""
        rank = self._rank()

        json_file = self._artifact_path("trace", ".json")

        try:
            prof.export_chrome_trace(json_file)
            logger.info("[Rank %s] Trace exported to %s", rank, json_file)

            if self._use_gzip:
                try:
                    subprocess.Popen(["gzip", "-f", json_file])
                    logger.info(
                        "[Rank %s] Triggered background compression for %s",
                        rank,
                        json_file,
                    )
                    self._trace_path = f"{json_file}.gz"
                except Exception as compress_err:
                    logger.warning(
                        "[Rank %s] Background gzip failed to start: %s",
                        rank,
                        compress_err,
                    )
                    self._trace_path = json_file
            else:
                self._trace_path = json_file

            self._artifact_paths["trace"] = self._trace_path

        except Exception as e:
            logger.warning("[Rank %s] Failed to export trace: %s", rank, e)

    def _try_enable_memory_history(self) -> None:
        """Enable backend-specific memory history for snapshot analysis."""
        if not self.profiler_config.torch_profiler_with_memory:
            return

        backend_name, memory_module = self._resolve_memory_history_backend()
        if backend_name is None or memory_module is None:
            return

        record_memory_history = getattr(memory_module, "_record_memory_history", None)
        if record_memory_history is None:
            logger.info(
                "[Rank %s] %s memory history is not supported on this platform",
                self._rank(),
                backend_name,
            )
            return

        try:
            record_memory_history(
                enabled="all",
                context="all",
                stacks="python",
                max_entries=100000,
                clear_history=True,
            )
            self._memory_history_enabled = True
            self._memory_history_backend = backend_name
            self._memory_history_module = memory_module
            logger.info("[Rank %s] %s memory history enabled", self._rank(), backend_name)
        except Exception as e:
            logger.warning(
                "[Rank %s] Failed to enable %s memory history: %s",
                self._rank(),
                backend_name,
                e,
            )

    def _try_dump_memory_snapshot(self) -> None:
        """Dump a backend-specific memory snapshot into the session directory."""
        if not self._memory_history_enabled:
            return

        try:
            if self._memory_history_module is None or self._memory_history_backend is None:
                return

            dump_snapshot = getattr(self._memory_history_module, "_dump_snapshot", None)
            if dump_snapshot is None:
                logger.info(
                    "[Rank %s] %s memory snapshot is not supported on this platform",
                    self._rank(),
                    self._memory_history_backend,
                )
                return

            snapshot_file = self._artifact_path("memory_snapshot", ".pickle")
            dump_snapshot(snapshot_file)
            self._artifact_paths["memory_snapshot"] = snapshot_file
            logger.info(
                "[Rank %s] %s memory snapshot dumped to %s",
                self._rank(),
                self._memory_history_backend,
                snapshot_file,
            )
        except Exception as e:
            logger.warning(
                "[Rank %s] Failed to dump %s memory snapshot: %s",
                self._rank(),
                self._memory_history_backend,
                e,
            )
        finally:
            try:
                if self._memory_history_module is not None:
                    disable_memory_history = getattr(
                        self._memory_history_module,
                        "_record_memory_history",
                        None,
                    )
                    if disable_memory_history is not None:
                        disable_memory_history(enabled=None)
            except Exception:
                pass
            self._memory_history_enabled = False
            self._memory_history_backend = None
            self._memory_history_module = None

    def _resolve_memory_history_backend(self) -> tuple[str | None, Any]:
        """Resolve the memory backend that supports history/snapshot APIs."""
        backend_specs = [
            ("CUDA", self._has_cuda_like_activity(), getattr(torch, "cuda", None)),
            ("NPU", "NPU" in self._activities, getattr(torch, "npu", None)),
            ("XPU", "XPU" in self._activities, getattr(torch, "xpu", None)),
            ("MUSA", "MUSA" in self._activities, getattr(torch, "musa", None)),
        ]

        for backend_name, enabled, device_module in backend_specs:
            if not enabled or device_module is None:
                continue

            is_available = getattr(device_module, "is_available", None)
            if callable(is_available) and not is_available():
                continue

            memory_module = getattr(device_module, "memory", None)
            if memory_module is not None:
                return backend_name, memory_module

        return None, None

    def _safe_get(self, obj, name: str, default=None):
        return getattr(obj, name, default)

    def _event_list_to_rows(self, event_list) -> list[dict]:
        rows = []
        for evt in event_list:
            row = {
                "name": self._safe_get(evt, "key", None) or self._safe_get(evt, "name", None),
                "count": self._safe_get(evt, "count", None),
                "device_type": self._safe_get(evt, "device_type", None),
                "node_id": self._safe_get(evt, "node_id", None),
                "self_cpu_time_total_us": self._safe_get(evt, "self_cpu_time_total", None),
                "cpu_time_total_us": self._safe_get(evt, "cpu_time_total", None),
                "self_cuda_time_total_us": self._safe_get(evt, "self_cuda_time_total", None),
                "cuda_time_total_us": self._safe_get(evt, "cuda_time_total", None),
                "self_xpu_time_total_us": self._safe_get(evt, "self_xpu_time_total", None),
                "xpu_time_total_us": self._safe_get(evt, "xpu_time_total", None),
                "self_cpu_memory_usage_bytes": self._safe_get(evt, "self_cpu_memory_usage", None),
                "cpu_memory_usage_bytes": self._safe_get(evt, "cpu_memory_usage", None),
                "self_cuda_memory_usage_bytes": self._safe_get(evt, "self_cuda_memory_usage", None),
                "cuda_memory_usage_bytes": self._safe_get(evt, "cuda_memory_usage", None),
                "self_xpu_memory_usage_bytes": self._safe_get(evt, "self_xpu_memory_usage", None),
                "xpu_memory_usage_bytes": self._safe_get(evt, "xpu_memory_usage", None),
                "input_shapes": str(self._safe_get(evt, "input_shapes", None)),
                "stack": "\n".join(self._safe_get(evt, "stack", []) or []),
                "overload_name": self._safe_get(evt, "overload_name", None),
                "is_async": self._safe_get(evt, "is_async", None),
                "is_legacy": self._safe_get(evt, "is_legacy", None),
            }
            rows.append(row)
        return rows

    def _write_excel_artifact(self, name: str, sheets: dict[str, list[dict]]) -> str:
        path = self._artifact_path(name, ".xlsx")

        try:
            import pandas as pd
        except Exception as e:
            logger.warning(
                "[Rank %s] pandas not available, skip Excel export: %s",
                self._rank(),
                e,
            )
            return path

        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            for sheet_name, rows in sheets.items():
                df = pd.DataFrame(rows)

                safe_sheet_name = sheet_name if sheet_name else "Sheet1"

                df.to_excel(
                    writer,
                    sheet_name=safe_sheet_name,
                    index=False,
                    freeze_panes=(1, 0),
                )

                ws = writer.sheets[safe_sheet_name]
                ws.auto_filter.ref = ws.dimensions

                for col_cells in ws.columns:
                    max_len = 0
                    col_letter = col_cells[0].column_letter
                    for cell in col_cells[:200]:
                        try:
                            val = "" if cell.value is None else str(cell.value)
                            max_len = max(max_len, len(val))
                        except Exception:
                            pass
                    ws.column_dimensions[col_letter].width = min(max(max_len + 2, 12), 80)

        self._artifact_paths[name] = path
        return path

    @override
    def _start(self) -> None:
        self._ensure_session_dir()
        self._try_enable_memory_history()
        self.profiler.start()

    @override
    def _stop(self) -> None:
        """Stop profiler, export trace via on_trace_ready, and dump table."""
        self.profiler.stop()
        try:
            self._on_stop_hook()
        finally:
            self._try_dump_memory_snapshot()

    def _on_stop_hook(self) -> None:
        """Hook called after profiler.stop().

        Override in subclasses for platform-specific post-stop handling.
        Base implementation handles CUDA time total dump.
        """
        rank = self.local_rank
        sort_key = self._get_time_sort_key()

        excel_sheets: dict[str, list[dict]] = {}

        # 1) Summary op table
        summary_events = self.profiler.key_averages()
        excel_sheets["summary"] = self._event_list_to_rows(summary_events)

        # 2) Shape-grouped op table
        if self.profiler_config.torch_profiler_record_shapes:
            try:
                shape_events = self.profiler.key_averages(
                    group_by_input_shape=True,
                )
                excel_sheets["by_shape"] = self._event_list_to_rows(shape_events)
            except Exception as e:
                logger.warning(
                    "[Rank %s] Failed to export shape-grouped op table: %s",
                    rank,
                    e,
                )

        # 3) Stack-grouped op table
        if self.profiler_config.torch_profiler_with_stack:
            try:
                stack_events = self.profiler.key_averages(
                    group_by_stack_n=8,
                )
                excel_sheets["by_stack"] = self._event_list_to_rows(stack_events)
            except Exception as e:
                logger.warning(
                    "[Rank %s] Failed to export stack-grouped op table: %s",
                    rank,
                    e,
                )

            # 4) Export stack files
            try:
                cpu_stack_file = self._artifact_path("stacks_cpu", ".txt")
                self.profiler.export_stacks(
                    cpu_stack_file,
                    metric="self_cpu_time_total",
                )
                self._artifact_paths["stacks_cpu"] = cpu_stack_file
            except Exception as e:
                logger.warning("[Rank %s] export_stacks(cpu) failed: %s", rank, e)

            if self._has_cuda_like_activity():
                try:
                    cuda_stack_file = self._artifact_path("stacks_cuda", ".txt")
                    self.profiler.export_stacks(
                        cuda_stack_file,
                        metric="self_cuda_time_total",
                    )
                    self._artifact_paths["stacks_cuda"] = cuda_stack_file
                except Exception as e:
                    logger.warning("[Rank %s] export_stacks(cuda) failed: %s", rank, e)

        try:
            self._table_path = self._write_excel_artifact("ops", excel_sheets)
        except Exception as e:
            logger.warning("[Rank %s] Failed to export Excel workbook: %s", rank, e)

        if self.profiler_config.torch_profiler_dump_cuda_time_total:
            profiler_dir = self.profiler_config.torch_profiler_dir
            sort_key = "self_cuda_time_total"
            table = self.profiler.key_averages().table(sort_by=sort_key)

            if not _is_uri_path(profiler_dir):
                table_file = os.path.join(
                    self._ensure_session_dir(),
                    f"profiler_out_{rank}.txt",
                )
                with open(table_file, "w") as f:
                    print(table, file=f)
                self._artifact_paths["profiler_out"] = table_file

            if rank == 0:
                print(table)

        if self.dump_cpu_time_total and rank == 0:
            logger.info(self.profiler.key_averages().table(sort_by="self_cpu_time_total", row_limit=50))

    @override
    def annotate_context_manager(self, name: str):
        return torch.profiler.record_function(name)

    def get_results(self) -> dict:
        """Return collected trace and table paths after stop."""
        return {
            "trace": self._trace_path,
            "table": self._table_path,
            **self._artifact_paths,
        }


def create_omni_profiler(
    profiler_config: ProfilerConfig,
    worker_name: str,
    local_rank: int,
    activities: list[TorchProfilerActivity] | None = None,
) -> OmniTorchProfilerWrapper:
    """Factory function to create platform-specific profiler.

    Uses the current platform's get_profiler_cls() to determine which
    profiler class to instantiate.

    Args:
        profiler_config: Profiler configuration.
        worker_name: Name of the worker.
        local_rank: Local rank of the worker.
        activities: Optional list of profiler activities.

    Returns:
        Platform-specific profiler instance.
    """
    from vllm.utils.import_utils import resolve_obj_by_qualname

    from vllm_omni.platforms import current_omni_platform

    profiler_cls_path = current_omni_platform.get_profiler_cls()
    profiler_cls = resolve_obj_by_qualname(profiler_cls_path)
    return profiler_cls(profiler_config, worker_name, local_rank, activities)
