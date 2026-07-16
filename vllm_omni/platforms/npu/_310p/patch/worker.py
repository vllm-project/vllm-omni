# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Patch Omni NPU worker device initialization for 310P.

This patches:
    vllm_omni.platforms.npu.worker.base.OmniNPUWorkerBase

Omni AR/generation workers share this base initialization path, so the 310P
compile-mode setup is applied here after device initialization.
"""

from __future__ import annotations

from vllm.v1.sample.sampler import Sampler
from vllm_ascend._310p.sample.sampler import AscendSampler310

from vllm_omni.platforms.npu._310p import disable_jit_compile
from vllm_omni.platforms.npu.worker import base as worker_base


class _OmniNPUWorkerBase310P(worker_base.OmniNPUWorkerBase):
    def _init_device(self):
        device = super()._init_device()
        disable_jit_compile()
        return device


def apply_patch() -> None:
    # Triton-Ascend does not target 310P; use vLLM's native penalty path.
    AscendSampler310.apply_penalties = staticmethod(Sampler.apply_penalties)
    worker_base.OmniNPUWorkerBase = _OmniNPUWorkerBase310P
