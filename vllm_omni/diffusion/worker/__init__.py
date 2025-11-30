# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Diffusion worker components."""

from vllm_omni.diffusion.worker.gpu_worker import GPUWorker, WorkerProc

__all__ = ["GPUWorker", "WorkerProc"]
