# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker classes for diffusion models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner
    from vllm_omni.diffusion.worker.diffusion_worker import DiffusionWorker, WorkerProc

__all__ = [
    "DiffusionModelRunner",
    "DiffusionWorker",
    "WorkerProc",
]


def __getattr__(name: str) -> Any:
    if name == "DiffusionModelRunner":
        from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner

        return DiffusionModelRunner
    if name in {"DiffusionWorker", "WorkerProc"}:
        from vllm_omni.diffusion.worker.diffusion_worker import DiffusionWorker, WorkerProc

        return {
            "DiffusionWorker": DiffusionWorker,
            "WorkerProc": WorkerProc,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
