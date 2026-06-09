# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Patch Qwen3-TTS worker device init for 310P."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from vllm_omni.platforms.npu._310p import disable_jit_compile

TARGET_MODULE = "vllm_omni.platforms.npu.worker.base"

_QWEN3_TTS_ARCHS = {
    "Qwen3TTSTalkerForConditionalGeneration",
    "Qwen3TTSCode2Wav",
}
_original_init_device: Callable[..., Any] | None = None


def is_ready(module: Any) -> bool:
    return hasattr(module, "OmniNPUWorkerBase")


def apply(module: Any) -> None:
    global _original_init_device

    if _original_init_device is not None:
        return

    cls = module.OmniNPUWorkerBase
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
    return getattr(model_config, "model_arch", None) in _QWEN3_TTS_ARCHS
