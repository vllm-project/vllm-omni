# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Monkey-patch ``Qwen3TTSCode2Wav.__init__`` for NPU Code2Wav runtime knobs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from vllm.config import VllmConfig
from vllm.logger import init_logger

if TYPE_CHECKING:
    pass

logger = init_logger(__name__)

_PATCHED = False
_original_init = None


def _prepare_npu_code2wav_runtime() -> None:
    from vllm_omni.platforms import current_omni_platform

    if not current_omni_platform.is_npu():
        return
    torch.npu.config.allow_internal_format = False
    torch.npu.set_compile_mode(jit_compile=False)


def _patched_init(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
    _prepare_npu_code2wav_runtime()
    assert _original_init is not None
    _original_init(self, vllm_config=vllm_config, prefix=prefix)


def apply_qwen3_tts_code2wav_patch() -> None:
    global _PATCHED, _original_init
    if _PATCHED:
        return

    from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_code2wav import Qwen3TTSCode2Wav

    _original_init = Qwen3TTSCode2Wav.__init__
    Qwen3TTSCode2Wav.__init__ = _patched_init  # type: ignore[method-assign]
    _PATCHED = True
    logger.debug("Applied NPU patch for Qwen3TTSCode2Wav.__init__")
