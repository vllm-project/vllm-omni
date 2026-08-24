# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Patch Qwen3-TTS Code2Wav NPU runtime setup and weight preparation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch_npu
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm_ascend.utils import maybe_trans_nz

if TYPE_CHECKING:
    pass

logger = init_logger(__name__)

_PATCHED = False
_original_init = None
_original_load_weights = None
_ACL_FORMAT_FRACTAL_Z = 4


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


def _prepare_npu_decoder_weights(decoder: nn.Module) -> None:
    linear_count = 0
    conv_count = 0
    with torch.no_grad():
        for module in decoder.modules():
            if isinstance(module, nn.Linear):
                module.weight.data = maybe_trans_nz(module.weight.data)
                linear_count += 1
            elif isinstance(module, (nn.Conv1d, nn.ConvTranspose1d)) and module.groups == 1:
                module.weight.data = torch_npu.npu_format_cast(module.weight.data.contiguous(), _ACL_FORMAT_FRACTAL_Z)
                conv_count += 1

    logger.info("Prepared NPU Code2Wav weights: linear=%d conv=%d", linear_count, conv_count)


def _patched_load_weights(self, weights):
    assert _original_load_weights is not None
    loaded = _original_load_weights(self, weights)
    device = self.vllm_config.device_config.device
    runtime_dtype = getattr(self, "_npu_decoder_runtime_dtype", lambda _: torch.float32)(device)
    self.decoder.to(device=device, dtype=runtime_dtype)
    _prepare_npu_decoder_weights(self.decoder)
    if runtime_dtype != torch.float32 and hasattr(self.decoder, "precompute_snake_caches"):
        self.decoder.precompute_snake_caches()
    return loaded


def apply_qwen3_tts_code2wav_patch() -> None:
    global _PATCHED, _original_init, _original_load_weights
    if _PATCHED:
        return

    from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_code2wav import Qwen3TTSCode2Wav

    _original_init = Qwen3TTSCode2Wav.__init__
    _original_load_weights = Qwen3TTSCode2Wav.load_weights
    Qwen3TTSCode2Wav.__init__ = _patched_init  # type: ignore[method-assign]
    Qwen3TTSCode2Wav.load_weights = _patched_load_weights  # type: ignore[method-assign]
    _PATCHED = True
    logger.debug("Applied NPU patch for Qwen3TTSCode2Wav")
