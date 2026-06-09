# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Patch Qwen3-TTS prompt embedding helpers for 310P."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from vllm_omni.platforms.npu._310p.qwen3_tts_runtime import (
    audio_frontend_runtime,
    patch_module_bfloat16,
    runtime_dtype,
)

TARGET_MODULE = "vllm_omni.model_executor.models.qwen3_tts.prompt_embeds_builder"

_prompt_builder_module: Any | None = None


def is_ready(module: Any) -> bool:
    return hasattr(module, "Qwen3TTSPromptEmbedsBuilder")


def apply(module: Any) -> None:
    global _prompt_builder_module

    if _prompt_builder_module is not None:
        return

    _prompt_builder_module = module
    patch_module_bfloat16(module)
    module.Qwen3TTSPromptEmbedsBuilder.extract_speaker_embedding = _extract_speaker_embedding_310p


def _extract_speaker_embedding_310p(self, wav: np.ndarray, sr: int) -> torch.Tensor:
    assert _prompt_builder_module is not None

    dev = self._device()
    runtime_dev, runtime_dt = audio_frontend_runtime(dev)
    try:
        spk_param = next(self._speaker_encoder.parameters())
        if spk_param.device != runtime_dev or spk_param.dtype != runtime_dt:
            self._speaker_encoder.to(device=runtime_dev, dtype=runtime_dt)
    except StopIteration:
        pass

    target_sr = int(getattr(self._config.speaker_encoder_config, "sample_rate", 24000))
    if sr != target_sr:
        resampler = self._get_resampler(int(sr), target_sr)
        wav = resampler.resample(wav.astype(np.float32), orig_sr=int(sr))

    wav_tensor = torch.from_numpy(wav).to(device=runtime_dev, dtype=torch.float32).unsqueeze(0)
    mels = _prompt_builder_module.mel_spectrogram(
        wav_tensor,
        n_fft=1024,
        num_mels=128,
        sampling_rate=24000,
        hop_size=256,
        win_size=1024,
        fmin=0,
        fmax=12000,
    ).transpose(1, 2)
    spk = self._speaker_encoder(mels.to(dtype=runtime_dt))[0]
    return spk.to(device=dev, dtype=runtime_dtype(dev))
