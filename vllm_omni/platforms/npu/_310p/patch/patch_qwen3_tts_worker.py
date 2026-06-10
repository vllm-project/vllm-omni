# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Patch Qwen3-TTS worker device init for 310P."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from vllm_omni.platforms.npu._310p import disable_jit_compile
from vllm_omni.platforms.npu._310p.qwen3_tts_runtime import use_qwen3_tts_310p_path

_original_init_device: Callable[..., Any] | None = None


def apply_patch() -> None:
    global _original_init_device

    if _original_init_device is not None:
        return

    from vllm_omni.platforms.npu.worker import base as worker_base

    cls = worker_base.OmniNPUWorkerBase
    _original_init_device = cls._init_device
    cls._init_device = _init_device_310p_qwen3_tts


def _init_device_310p_qwen3_tts(self):
    assert _original_init_device is not None

    device = _original_init_device(self)
    if _is_qwen3_tts_worker(self):
        # Qwen3-TTS uses Omni AR/generation workers, which bypass
        # NPUWorker310.init_device() and need this 310P setup after device init.
        disable_jit_compile()
    return device


def _is_qwen3_tts_worker(worker: Any) -> bool:
    model_config = getattr(getattr(worker, "vllm_config", None), "model_config", None)
    return use_qwen3_tts_310p_path(model_config)
