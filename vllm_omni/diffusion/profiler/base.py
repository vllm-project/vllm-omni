# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod

from vllm.logger import init_logger

logger = init_logger(__name__)


class ProfilerBase(ABC):
    """
    Abstract base class for all diffusion profilers.
    Defines the common interface used by GPUWorker and DiffusionEngine.
    """

    @abstractmethod
    def start(self, trace_path_template: str) -> str:
        """
        Start profiling.

        Args:
            trace_path_template: Base path (without rank or extension).
                                 e.g. "/tmp/profiles/sdxl_run"

        Returns:
            Full path of the trace file this rank will write.
        """
        pass

    @abstractmethod
    def stop(self) -> str | None:
        """
        Stop profiling and finalize/output the trace.

        Returns:
            Path to the saved trace file, or None if not active.
        """
        pass

    @abstractmethod
    def get_step_context(self):
        """
        Returns a context manager that advances one profiling step.
        Should be a no-op (nullcontext) when profiler is not active.
        """
        pass

    @abstractmethod
    def is_active(self) -> bool:
        """Return True if profiling is currently running."""
        pass

    @classmethod
    def _get_rank(cls) -> int:
        import os

        return int(os.getenv("RANK", "0"))
