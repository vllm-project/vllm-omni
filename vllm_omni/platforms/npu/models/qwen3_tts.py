# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Patch Qwen3-TTS NPU runtime setup and weights.

The code2wav runtime/weight fixes apply on every NPU. On A5 (Ascend 950 PR)
``torch.stft`` is unavailable, so the speaker-encoder mel front-end of the
prompt-embeds builder runs on CPU and the result is moved back to NPU
(same trade-off as the 310P path).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch_npu
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm_ascend.utils import maybe_trans_nz

from vllm_omni.platforms.npu import is_a5

logger = init_logger(__name__)

_PATCHED = False
_A5_PATCHED = False
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
                # A5 (Ascend 950 PR) does not support the FRACTAL_Z conv
                # weight layout, keep the contiguous ND layout there.
                if not is_a5():
                    module.weight.data = torch_npu.npu_format_cast(
                        module.weight.data.contiguous(), _ACL_FORMAT_FRACTAL_Z
                    )
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


def apply_qwen3_tts_patches() -> None:
    """Install all Qwen3-TTS NPU patches.

    The code2wav runtime/weight patch applies on every NPU; the A5
    prompt-embeds builder swap is gated on ``is_a5()`` because it changes the
    speaker-encoder mel front-end (``torch.stft`` is unsupported there).
    """
    apply_qwen3_tts_code2wav_patch()
    if not is_a5():
        return
    _apply_a5_prompt_embeds_builder_patch()


def _apply_a5_prompt_embeds_builder_patch() -> None:
    global _A5_PATCHED
    if _A5_PATCHED:
        return

    # Import and patch lazily so the Qwen3-TTS model modules are only loaded
    # on A5 devices (same gate-then-import convention as the 310P patch).
    from vllm_omni.model_executor.models.qwen3_tts import (
        prompt_embeds_builder,
        qwen3_tts_talker,
    )

    class _Qwen3TTSPromptEmbedsBuilderA5(prompt_embeds_builder.Qwen3TTSPromptEmbedsBuilder):
        """Qwen3-TTS prompt-embeds builder specialized for the A5 NPU path."""

        _mel_spectrogram_on_cpu = True

    # The talker imports the builder by name at module scope, so its reference
    # must be swapped as well (same as the 310P patch).
    prompt_embeds_builder.Qwen3TTSPromptEmbedsBuilder = _Qwen3TTSPromptEmbedsBuilderA5
    qwen3_tts_talker.Qwen3TTSPromptEmbedsBuilder = _Qwen3TTSPromptEmbedsBuilderA5
    _A5_PATCHED = True
    logger.debug("Applied A5 prompt-embeds builder patch for Qwen3-TTS")


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
