# Copyright (c) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HPU worker classes for diffusion models."""

from vllm_omni.diffusion.worker.hpu.hpu_worker import HPUWorker, HPUWorkerProc

__all__ = ["HPUWorker", "HPUWorkerProc"]
