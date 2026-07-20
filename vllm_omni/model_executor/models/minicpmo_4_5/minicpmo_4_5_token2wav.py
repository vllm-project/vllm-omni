# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniCPM-o 4.5 Token2wav adapter over in-tree ``StepAudio2Token2WavCore``.

``minicpmo_4_5_omni_tts`` historically depended on the external
``from stepaudio2 import Token2wav`` entry point (``stepaudio2-minicpmo``).
That package hard-codes ``.cuda()`` and duplicates the flow/HiFT stack that
vLLM-Omni already vendors for Step-Audio2 (which also carries the Ascend/NPU
fixes: CPU HiFT, DiT mask expand, MATH SDPA, compile disable, PE buffer
extension).

This module exposes the same call surface MiniCPM expects
(``__call__`` / ``set_stream_cache`` / ``stream``) while delegating the
actual vocoder work to
``vllm_omni.model_executor.models.step_audio2.step_audio2_token2wav.StepAudio2Token2WavCore``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from vllm_omni.model_executor.models.step_audio2.step_audio2_token2wav import (
    StepAudio2Token2WavCore,
    _StreamState,
)


def _resolve_device(device: str | torch.device | None) -> str:
    if device is not None:
        return str(device)
    try:
        from vllm_omni.platforms import current_omni_platform

        plat_dev = getattr(current_omni_platform, "device_type", None)
        if plat_dev:
            return str(plat_dev)
    except Exception:
        pass
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "npu") and torch.npu.is_available():
        return "npu"
    return "cpu"


class MiniCPMO45Token2wav:
    """Token2wav-compatible facade around ``StepAudio2Token2WavCore``.

    Matches the MiniCPM-o / ``stepaudio2-minicpmo`` vocoder API used by
    ``MiniCPMO45OmniTTSForConditionalGeneration.generate_speech``:

    * ``Token2wav(model_path, float16=False, n_timesteps=10)``
    * ``__(tokens, prompt_wav) -> wav_bytes``
    * ``set_stream_cache(prompt_wav) -> (stream_cache, hift_cache_dict)``
    * ``stream(tokens, prompt_wav, last_chunk=..., return_waveform=...)``
    """

    def __init__(
        self,
        model_path: str,
        float16: bool = False,
        n_timesteps: int = 10,
        device: str | torch.device | None = None,
    ):
        self.float16 = float16
        self.n_timesteps = n_timesteps
        self.device = _resolve_device(device)
        self._core = StepAudio2Token2WavCore(
            model_path=model_path,
            float16=float16,
            device=self.device,
            n_timesteps=n_timesteps,
        )
        # Eager-load so construction failures surface at init time (same as
        # the external Token2wav package), not on the first request.
        self._core._ensure_models_loaded()

        # Mutable streaming fields expected by MiniCPM's long-form path.
        self.stream_cache: Any | None = None
        self.hift_cache_dict: dict[str, torch.Tensor] = {}

    def __call__(self, generated_speech_tokens, prompt_wav) -> bytes:
        """One-shot tokens → 24 kHz WAV bytes."""
        return self._core.forward(
            generated_speech_tokens,
            prompt_wav,
            return_bytes=True,
        )

    def set_stream_cache(self, prompt_wav: str):
        """Initialise flow + HiFT caches for chunked vocoding.

        Returns ``(stream_cache, hift_cache_dict)`` so callers can assign them
        back onto this object the way ``stepaudio2.Token2wav`` does.
        """
        state = _StreamState()
        self._core.setup_stream_for(prompt_wav, state)
        self.stream_cache = state.stream_cache
        self.hift_cache_dict = state.hift_cache_dict
        return self.stream_cache, self.hift_cache_dict

    def stream(
        self,
        generated_speech_tokens,
        prompt_wav: str,
        last_chunk: bool = False,
        return_waveform: bool = False,
    ):
        """Process one streaming chunk; updates ``stream_cache`` / ``hift_cache_dict``."""
        if self.stream_cache is None:
            raise ValueError("stream_cache is not set")

        state = _StreamState()
        state.setup_done = True
        state.stream_cache = self.stream_cache
        state.hift_cache_dict = self.hift_cache_dict

        speech = self._core.stream_chunk_for(
            list(generated_speech_tokens),
            prompt_wav,
            last_chunk,
            state,
        )
        self.stream_cache = state.stream_cache
        self.hift_cache_dict = state.hift_cache_dict

        wav_np = speech.detach().float().cpu().numpy()
        if return_waveform:
            return wav_np

        wav_np = np.clip(wav_np, -1.0, 1.0)
        wav_int16 = (wav_np * 32767.0).astype("<i2")
        return wav_int16.tobytes()
