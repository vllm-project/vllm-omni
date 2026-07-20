# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NPU patches for Step-Audio2 / MiniCPM Token2Wav.

Ascend-specific workarounds that must not live in the shared GPU model file:

1. HiFT vocoder CPU offload — STFT/ISTFT and sine-source generation are
   unstable on Ascend (aicore 507015). HiFT is tiny, so running it on CPU
   is acceptable; inputs/outputs are moved transparently.
2. CosyVoice2 DiT SDPA — force MATH backend (+ DiT attn mask expand) to
   avoid fused FA rejecting CosyVoice ``(B,1,1,S)`` masks (error 161001).
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from types import MethodType

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)

_PATCHED = False
_original_ensure_models_loaded = None
_original_forward = None
_original_stream_chunk_for = None


def patch_hift_module_for_npu(hift: torch.nn.Module) -> None:
    """Run the entire HiFT vocoder on CPU when on Ascend NPU.

    HiFT is tiny, so keeping it on NPU yields negligible speedup while
    several of its operators are unstable on Ascend and trigger aicore
    exceptions (runtime error 507015): the STFT/ISTFT as well as the
    sine-source generation (``f0_upsamp`` + ``m_source`` cumsum/sin over
    the full waveform length). Running the whole module on CPU sidesteps
    all of them. Inputs are copied to CPU and outputs are moved back to
    the original device transparently, so callers stay unchanged.
    """
    if getattr(hift, "_npu_cpu_offload_patched", False):
        return

    hift.to("cpu")
    original_forward = hift.forward

    def _forward_on_cpu(_module, *args, **kwargs):
        output_device: torch.device | None = None

        def to_cpu(value):
            nonlocal output_device
            if isinstance(value, torch.Tensor):
                if output_device is None and value.device.type != "cpu":
                    output_device = value.device
                return value.cpu()
            return value

        cpu_args = tuple(to_cpu(a) for a in args)
        cpu_kwargs = {k: to_cpu(v) for k, v in kwargs.items()}
        output = original_forward(*cpu_args, **cpu_kwargs)

        if output_device is None:
            return output

        def to_device(value):
            if isinstance(value, torch.Tensor):
                return value.to(output_device)
            return value

        if isinstance(output, tuple):
            return tuple(to_device(v) for v in output)
        return to_device(output)

    hift.forward = MethodType(_forward_on_cpu, hift)
    hift._npu_cpu_offload_patched = True


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
    patch_hift_module_for_npu(self._hift)


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
