# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Patch Omni NPU worker device initialization for 310P."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from vllm_omni.platforms.npu._310p import disable_jit_compile

_original_init_device: Callable[..., Any] | None = None


def apply_patch() -> None:
    global _original_init_device

    if _original_init_device is not None:
        return

    from vllm_omni.platforms.npu.worker import base as worker_base

    cls = worker_base.OmniNPUWorkerBase
    _original_init_device = cls._init_device
    cls._init_device = _init_device_310p


def _init_device_310p(self):
    if _original_init_device is None:
        raise RuntimeError("310P worker patch was not initialized.")

    device = _original_init_device(self)
    # Omni AR/generation worker subclasses share this base init path; keep the
    # 310P compile-mode setup here so the backend worker setup is not skipped.
    disable_jit_compile()
    return device
