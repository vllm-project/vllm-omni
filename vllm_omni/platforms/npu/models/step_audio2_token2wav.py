# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NPU patches for Step-Audio2 / MiniCPM Token2Wav.

Ascend-specific workarounds that must not live in the shared GPU model file:

1. HiFT sine-source downsample — replace the failing 480x ``linear1d``
   downsample with its exact midpoint form while keeping HiFT on NPU.
2. HiFT device constants — keep the harmonic multiplier and the STFT window
   resident on the accelerator instead of copying them up from host memory on
   every vocoder call.
3. CosyVoice2 DiT SDPA — force MATH backend (+ DiT attn mask expand) to
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


def _sinegen2_forward_with_resident_harmonics(
    self,
    f0: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``SineGen2.forward``, reading its harmonic multiplier from the device.

    Identical to upstream in every operation and in the order it draws from
    the RNG; the only difference is that the multiplier is a tensor that
    already lives on ``f0``'s device instead of one built on the host and
    copied up.
    """
    multipliers = self._npu_resident_harmonics
    if multipliers.device != f0.device or multipliers.dtype != torch.float32:
        # Upstream builds a fresh FP32 tensor on the input device every call,
        # so honour that invariant if the module is migrated after loading —
        # without paying for it in steady state.
        multipliers = multipliers.to(device=f0.device, dtype=torch.float32)
        self._npu_resident_harmonics = multipliers
    fn = torch.multiply(f0, multipliers)
    sine_waves = self._f02sine(fn) * self.sine_amp
    uv = self._f02uv(f0)
    noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
    noise = noise_amp * torch.randn_like(sine_waves)
    sine_waves = sine_waves * uv + noise
    return sine_waves, uv, noise


def _make_hift_constants_resident(hift: torch.nn.Module, sine_gen: torch.nn.Module) -> int:
    """Materialize HiFT's two immutable tensors once, on the model's device.

    ``SineGen2.forward`` rebuilds ``torch.FloatTensor([[range(1, n + 2)]])`` on
    the host and copies it to the device on every call, and ``_stft`` /
    ``_istft`` each run ``self.stft_window.to(x.device)`` — the window is a
    plain attribute rather than a registered buffer, so it never moved with
    the module. That is three unpinned host-to-device copies per vocoder call.

    On Ascend an unpinned copy costs far more than its bytes: it drains the
    device queue, so the price is a pipeline stall on a hot streaming path.
    Both tensors are constants, so hoisting them is numerically exact.

    Returns the number of constants made resident, for the log line.
    """
    parameters = getattr(hift, "parameters", None)
    if not callable(parameters):
        return 0
    parameter = next(parameters(), None)
    if parameter is None:
        return 0
    device = parameter.device
    resident = 0

    harmonic_num = getattr(sine_gen, "harmonic_num", None)
    if harmonic_num is not None and all(
        hasattr(sine_gen, name) for name in ("sine_amp", "noise_std", "_f02uv", "forward")
    ):
        # Deliberately a plain attribute, not a registered buffer: a buffer
        # would follow ``Module.half()`` away from the FP32 tensor upstream
        # builds explicitly. The replacement forward repairs device and dtype
        # once if the module is migrated later.
        sine_gen._npu_resident_harmonics = torch.arange(
            1,
            int(harmonic_num) + 2,
            device=device,
            dtype=torch.float32,
        ).view(1, 1, -1)
        sine_gen._step_audio2_original_forward = sine_gen.forward
        sine_gen.forward = MethodType(_sinegen2_forward_with_resident_harmonics, sine_gen)
        resident += 1

    window = getattr(hift, "stft_window", None)
    if isinstance(window, torch.Tensor):
        # ``_patched_ensure_models_loaded`` runs this after final placement, so
        # the window lands on the device the convolutions are already on.
        hift.stft_window = window.to(device=device)
        resident += 1
    return resident


def patch_step_audio2_hift_for_npu(hift: torch.nn.Module) -> None:
    """Patch the non-causal Step-Audio2 HiFT implementation for Ascend.

    The ``flashcosyvoice.SineGen2`` instantiated by Step-Audio2 1.0.0 is
    non-causal and reduces a full-rate phase tensor by ``1 / 480`` before
    restoring it to the waveform rate. Ascend's ``upsample_linear1d`` kernel
    can raise an AIVector UB-address exception (ACL 507015) for that reduction.

    The exact midpoint form keeps the common path on NPU. Unsupported or pulse
    configurations delegate only ``_f02sine`` to CPU, preserving upstream
    behavior without restoring the old whole-HiFT CPU offload.

    It also makes HiFT's two host-built constants resident on the device — see
    ``_make_hift_constants_resident``. Both are numerically exact.
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
    resident = _make_hift_constants_resident(hift, sine_gen)
    hift._step_audio2_npu_downsample_patched = True
    logger.info(
        "Patched Step-Audio2 HiFT for Ascend NPU: linear downsample, %d resident constants",
        resident,
    )


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
