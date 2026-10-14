# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from typing_extensions import override
from vllm.logger import init_logger

from vllm_omni.profiler.omni_torch_profiler import (
    OmniTorchProfilerWrapper,
    TorchProfilerActivity,
)

logger = init_logger(__name__)


class XPUTorchProfilerWrapper(OmniTorchProfilerWrapper):
    """XPU-specific profiler wrapper.

    Uses torch.profiler with XPU ProfilerActivity to capture Intel GPU events.
    XPU events are exposed via torch.profiler.ProfilerActivity.XPU and work
    with the standard torch.profiler interface.
    """

    @override
    def _get_default_activities(self) -> list[TorchProfilerActivity]:
        """Default to CPU + XPU profiling for Intel GPUs."""
        return ["CPU", "XPU"]

    @override
    def _on_stop_hook(self) -> None:
        """XPU-specific stop hook that handles XPU event aggregation.

        XPU profiler follows CUDA-like patterns but uses xpu_time_total metrics.
        """
        rank = self.local_rank
        sort_key = "self_xpu_time_total"

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

            # XPU stack traces
            try:
                xpu_stack_file = self._artifact_path("stacks_xpu", ".txt")
                self.profiler.export_stacks(
                    xpu_stack_file,
                    metric="self_xpu_time_total",
                )
                self._artifact_paths["stacks_xpu"] = xpu_stack_file
            except Exception as e:
                logger.warning("[Rank %s] export_stacks(xpu) failed: %s", rank, e)

        try:
            self._table_path = self._write_excel_artifact("ops", excel_sheets)
        except Exception as e:
            logger.warning("[Rank %s] Failed to export Excel workbook: %s", rank, e)

        # Print XPU time table for rank 0
        if rank == 0:
            try:
                table = self.profiler.key_averages().table(sort_by=sort_key, row_limit=50)
                logger.info("XPU Profiler Results:\n%s", table)
            except Exception as e:
                logger.warning("[Rank %s] Failed to generate XPU time table: %s", rank, e)

        if self.dump_cpu_time_total and rank == 0:
            logger.info(self.profiler.key_averages().table(sort_by="self_cpu_time_total", row_limit=50))
