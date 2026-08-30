# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NPU patches for Step-Audio2 / MiniCPM Token2Wav.

Ascend-specific workarounds that must not live in the shared GPU model file:

1. HiFT sine-source downsample — replace the failing 480x ``linear1d``
   downsample with its exact midpoint form while keeping HiFT on NPU.
2. CosyVoice2 DiT SDPA — force MATH backend (+ DiT attn mask expand) to
   avoid fused FA rejecting CosyVoice ``(B,1,1,S)`` masks (error 161001).
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from types import MethodType

import numpy as np
import torch
import torch.nn.functional as F
from vllm.logger import init_logger

logger = init_logger(__name__)

_PATCHED = False
_original_ensure_models_loaded = None
_original_forward = None
_original_stream_chunk_for = None


def _linear_downsample_even_scale(x: torch.Tensor, scale: int) -> torch.Tensor:
    """Match ``F.interpolate(..., mode="linear")`` for an even integer scale.

    With ``align_corners=False``, every output location for an even integer
    downsample lies exactly halfway between two source samples. Selecting and
    averaging those samples avoids Ascend/pytorch#150's ``linear1d`` kernel.
    """
    if scale <= 0 or scale % 2:
        raise ValueError(f"scale must be a positive even integer, got {scale}")
    if x.shape[-1] % scale:
        raise ValueError(f"input length {x.shape[-1]} must be divisible by scale {scale}")

    left = scale // 2 - 1
    right = scale // 2
    return (x[..., left::scale] + x[..., right::scale]) * 0.5


def _run_original_f02sine_on_cpu(self, f0_values: torch.Tensor) -> torch.Tensor:
    """Run the unmodified ``_f02sine`` without invoking NPU ``linear1d``."""
    output_device = f0_values.device
    output = self._step_audio2_original_f02sine(f0_values.cpu())
    return output.to(output_device)


def _f02sine_with_npu_safe_downsample(self, f0_values: torch.Tensor) -> torch.Tensor:
    """Use the exact NPU midpoint path, with a narrow CPU fallback."""
    if getattr(self, "flag_for_pulse", False):
        return _run_original_f02sine_on_cpu(self, f0_values)

    upsample_scale = self.upsample_scale
    if upsample_scale <= 0:
        raise ValueError(f"upsample_scale must be positive, got {upsample_scale}")

    scale = int(upsample_scale)
    midpoint_supported = scale == upsample_scale and scale % 2 == 0 and f0_values.shape[1] % scale == 0
    if not midpoint_supported:
        return _run_original_f02sine_on_cpu(self, f0_values)

    rad_values = (f0_values / self.sampling_rate) % 1
    rand_ini = torch.rand(f0_values.shape[0], f0_values.shape[2], device=f0_values.device)
    rand_ini[:, 0] = 0
    rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini

    rad_values = _linear_downsample_even_scale(rad_values.transpose(1, 2), scale).transpose(1, 2)
    phase = torch.cumsum(rad_values, dim=1) * 2 * np.pi
    phase = F.interpolate(
        phase.transpose(1, 2) * self.upsample_scale,
        scale_factor=self.upsample_scale,
        mode="linear",
    ).transpose(1, 2)
    return torch.sin(phase)


def patch_step_audio2_hift_for_npu(hift: torch.nn.Module) -> None:
    """Patch the non-causal Step-Audio2 HiFT implementation for Ascend.

    The ``flashcosyvoice.SineGen2`` instantiated by Step-Audio2 1.0.0 is
    non-causal and reduces a full-rate phase tensor by ``1 / 480`` before
    restoring it to the waveform rate. Ascend's ``upsample_linear1d`` kernel
    can raise an AIVector UB-address exception (ACL 507015) for that reduction.

    The exact midpoint form keeps the common path on NPU. Unsupported or pulse
    configurations delegate only ``_f02sine`` to CPU, preserving upstream
    behavior without restoring the old whole-HiFT CPU offload.
    """
    if getattr(hift, "_step_audio2_npu_downsample_patched", False):
        return

    try:
        sine_gen = hift.m_source.l_sin_gen
        original_f02sine = sine_gen._f02sine
    except AttributeError as exc:
        raise TypeError("expected a Step-Audio2 flashcosyvoice HiFT with m_source.l_sin_gen._f02sine") from exc

    if getattr(sine_gen, "causal", False):
        raise ValueError("the Step-Audio2 NPU HiFT patch only supports non-causal SineGen2")

    sine_gen._step_audio2_original_f02sine = original_f02sine
    sine_gen._f02sine = MethodType(_f02sine_with_npu_safe_downsample, sine_gen)
    hift._step_audio2_npu_downsample_patched = True
    logger.info("Patched Step-Audio2 HiFT linear downsample for Ascend NPU")


@contextmanager
def npu_token2wav_sdpa_context() -> Iterator[None]:
    """Expand CosyVoice masks + force MATH SDPA to avoid FA 161001."""
    try:
        from vllm_omni.platforms.npu.models.cosyvoice2_dit_attn import (
            apply_cosyvoice2_dit_attn_npu_patch,
            npu_math_sdpa_context,
        )

        apply_cosyvoice2_dit_attn_npu_patch()
        with npu_math_sdpa_context():
            yield
    except Exception:
        with nullcontext():
            yield


def _patched_ensure_models_loaded(self) -> None:
    assert _original_ensure_models_loaded is not None
    was_loaded = self._models_loaded
    _original_ensure_models_loaded(self)
    if was_loaded or self.device.type != "npu" or self._hift is None:
        return
    patch_step_audio2_hift_for_npu(self._hift)


def _patched_forward(self, generated_speech_tokens, prompt_wav, return_bytes=True):
    assert _original_forward is not None
    if self.device.type != "npu":
        return _original_forward(self, generated_speech_tokens, prompt_wav, return_bytes)
    with npu_token2wav_sdpa_context():
        return _original_forward(self, generated_speech_tokens, prompt_wav, return_bytes)


def _patched_stream_chunk_for(self, audio_tokens, prompt_wav, last_chunk, state):
    assert _original_stream_chunk_for is not None
    if self.device.type != "npu":
        return _original_stream_chunk_for(self, audio_tokens, prompt_wav, last_chunk, state)
    with npu_token2wav_sdpa_context():
        return _original_stream_chunk_for(self, audio_tokens, prompt_wav, last_chunk, state)


def apply_step_audio2_token2wav_npu_patch() -> None:
    """Monkey-patch StepAudio2Token2WavCore for Ascend NPU.

    Import is deferred and optional: platform bootstrap (e.g. resolving
    ``current_omni_platform`` from rotary embedding) must not require
    Token2Wav optional deps such as ``librosa``.
    """
    global _PATCHED, _original_ensure_models_loaded, _original_forward, _original_stream_chunk_for
    if _PATCHED:
        return

    try:
        from vllm_omni.model_executor.models.step_audio2.step_audio2_token2wav import (
            StepAudio2Token2WavCore,
        )
    except ImportError as e:
        logger.debug("step_audio2 token2wav deps unavailable; skip NPU patch: %s", e)
        return

    _original_ensure_models_loaded = StepAudio2Token2WavCore._ensure_models_loaded
    _original_forward = StepAudio2Token2WavCore.forward
    _original_stream_chunk_for = StepAudio2Token2WavCore.stream_chunk_for

    StepAudio2Token2WavCore._ensure_models_loaded = _patched_ensure_models_loaded  # type: ignore[method-assign]
    StepAudio2Token2WavCore.forward = _patched_forward  # type: ignore[method-assign]
    StepAudio2Token2WavCore.stream_chunk_for = _patched_stream_chunk_for  # type: ignore[method-assign]

    _PATCHED = True
    logger.debug("Applied NPU patch for StepAudio2Token2WavCore")
