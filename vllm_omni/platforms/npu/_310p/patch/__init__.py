# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Apply 310P patches."""

from __future__ import annotations

_WORKER_PATCHED = False
_QWEN3_TTS_TALKER_ARCH = "Qwen3TTSTalkerForConditionalGeneration"
_QWEN3_TTS_CODE2WAV_ARCH = "Qwen3TTSCode2Wav"


def apply_patches() -> None:
    global _WORKER_PATCHED

    if _WORKER_PATCHED:
        return

    from vllm_omni.platforms.npu._310p.patch.worker import apply_patch

    apply_patch()
    _WORKER_PATCHED = True


def apply_fft_patches() -> None:
    """Install NPU-native FFT/STFT replacements on 310P.

    310P ships no FFT-family aclnn kernels (``torch.fft.*``, ``stft``,
    ``istft``, ``hann_window`` all fail at runtime), so Qwen3-TTS and other
    speech models currently pin their mel front-ends to CPU
    (``_mel_spectrogram_on_cpu = True``). The rebuilt operators are
    torch-API-compatible drop-ins verified against CPU references at
    rel ~1e-7 on 310P3 (CANN 9.2, torch_npu 2.10.post7); see
    ``vllm_omni/platforms/npu/_310p/fft_ops.py`` for the design notes.
    """
    import torch

    import vllm_omni.platforms.npu._310p.fft_ops as fft_ops

    if getattr(torch, "_npu_fft_310p_patched", False):
        return
    torch.stft = fft_ops.stft
    torch.istft = fft_ops.istft
    torch.hann_window = fft_ops.hann_window
    torch.hamming_window = fft_ops.hamming_window
    torch.kaiser_window = fft_ops.kaiser_window
    import torch.fft as tfft

    tfft.fft = fft_ops.fft
    tfft.ifft = fft_ops.ifft
    tfft.rfft = fft_ops.rfft
    tfft.irfft = fft_ops.irfft
    torch._npu_fft_310p_patched = True


def apply_model_patches(model_config) -> None:
    model_arch = getattr(model_config, "model_arch", None)
    if model_arch == _QWEN3_TTS_TALKER_ARCH:
        from vllm_omni.platforms.npu._310p.patch.qwen3_tts import apply_talker_patches

        apply_talker_patches()
        return
    elif model_arch == _QWEN3_TTS_CODE2WAV_ARCH:
        from vllm_omni.platforms.npu._310p.patch.qwen3_tts import apply_code2wav_patches

        apply_code2wav_patches()
