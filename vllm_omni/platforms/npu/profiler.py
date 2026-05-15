# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import os

from typing_extensions import override
from vllm.config import ProfilerConfig
from vllm.logger import init_logger

from vllm_omni.profiler.omni_torch_profiler import (
    OmniTorchProfilerWrapper,
    TorchProfilerActivity,
)

logger = init_logger(__name__)


class NPUTorchProfilerWrapper(OmniTorchProfilerWrapper):
    """NPU-specific profiler using torch_npu.profiler.

    Key differences from base profiler:
    - Uses torch_npu.profiler instead of torch.profiler
    - Different experimental_config options (AiCMetrics, profiler_level, etc.)
    - Uses tensorboard_trace_handler for trace output
    - No key_averages() support - requires offline parsing with
      torch_npu.profiler.profiler.analyse()
    """

    @override
    def _get_default_activities(self) -> list[TorchProfilerActivity]:
        return ["CPU", "NPU"]

    @override
    def _create_profiler(
        self,
        profiler_config: ProfilerConfig,
        activities: list[TorchProfilerActivity],
    ):
        import torch_npu

        # Map activity names to torch_npu profiler activities
        npu_activities = []
        for activity in activities:
            if activity == "CPU":
                npu_activities.append(torch_npu.profiler.ProfilerActivity.CPU)
            elif activity == "NPU":
                npu_activities.append(torch_npu.profiler.ProfilerActivity.NPU)

        # NPU-specific experimental config for detailed profiling
        experimental_config = torch_npu.profiler._ExperimentalConfig(
            export_type=torch_npu.profiler.ExportType.Text,
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
            msprof_tx=False,
            aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
            l2_cache=False,
            op_attr=False,
            data_simplification=True,
            record_op_args=False,
            gc_detect_threshold=None,
        )

        # Set up trace directory for NPU - tensorboard_trace_handler creates
        # its own subdirectory structure. Use worker_name which includes stage_id.
        npu_trace_dir = os.path.join(self._trace_dir, self._worker_name)
        os.makedirs(npu_trace_dir, exist_ok=True)
        self._trace_path = npu_trace_dir

        return torch_npu.profiler.profile(
            activities=npu_activities,
            with_stack=False,
            profile_memory=profiler_config.torch_profiler_with_memory,
            # NOTE: torch_npu.profiler.with_modules is equivalent to
            # torch.profiler.with_stack. The with_stack option in
            # torch_npu.profiler introduces significant time overhead.
            with_modules=profiler_config.torch_profiler_with_stack,
            experimental_config=experimental_config,
            # Use tensorboard_trace_handler directly - NPU profiler expects
            # this specific handler format
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(npu_trace_dir),
        )

    @override
    def _on_stop_hook(self) -> None:
        """NPU profiler doesn't support key_averages() - log offline parsing hint."""
        if self.local_rank == 0:
            logger.info(
                "NPU profiler stopped. Use offline parsing to analyze: "
                "from torch_npu.profiler.profiler import analyse; "
                "analyse('%s')",
                self._trace_path or self._trace_dir,
            )
