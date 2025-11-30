# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker classes for diffusion models."""

from vllm_omni.diffusion.worker.gpu_worker import GPUWorker, WorkerProc

__all__ = ["GPUWorker", "WorkerProc"]
