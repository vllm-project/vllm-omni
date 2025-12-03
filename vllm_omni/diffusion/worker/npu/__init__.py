# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NPU worker classes for diffusion models."""

from vllm_omni.diffusion.worker.npu.npu_worker import NPUWorker, NPUWorkerProc

__all__ = ["NPUWorker", "NPUWorkerProc"]
