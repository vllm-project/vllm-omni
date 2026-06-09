# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Patch Qwen3-TTS talker runtime helpers for 310P."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from vllm.multimodal.audio import AudioResampler

from vllm_omni.platforms.npu._310p.qwen3_tts_runtime import (
    audio_frontend_runtime,
    patch_module_bfloat16,
)

TARGET_MODULE = "vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_talker"

_original_load_weights: Callable[..., Any] | None = None


def is_ready(module: Any) -> bool:
    return hasattr(module, "Qwen3TTSTalkerForConditionalGeneration")


def apply(module: Any) -> None:
    global _original_load_weights

    if _original_load_weights is not None:
        return

    patch_module_bfloat16(module)
    cls = module.Qwen3TTSTalkerForConditionalGeneration
    _original_load_weights = cls.load_weights
    cls._encode_ref_audio_batch = _encode_ref_audio_batch_310p
    cls.load_weights = _load_weights_310p


def _encode_ref_audio_batch_310p(
    self,
    wavs: list[np.ndarray],
    sr: int,
    *,
    device: torch.device,
) -> list[torch.Tensor]:
    fe = self._encoder_feature_extractor
    target_sr = int(fe.sampling_rate)
    if int(sr) != target_sr:
        resampler = AudioResampler(target_sr=target_sr)
        wavs = [resampler.resample(w.astype(np.float32), orig_sr=int(sr)) for w in wavs]

    inputs = fe(
        raw_audio=wavs,
        sampling_rate=target_sr,
        return_tensors="pt",
    )
    encoder_device, encoder_dtype = audio_frontend_runtime(device)
    inputs = inputs.to(encoder_device).to(encoder_dtype)

    input_values = inputs["input_values"].squeeze(1)
    padding_mask = inputs["padding_mask"].squeeze(1)

    with torch.inference_mode():
        encoded = self.encoder.encode(
            input_values=input_values.unsqueeze(1),
            return_dict=True,
        )

    audio_codes = encoded.audio_codes[:, : self._encoder_valid_num_quantizers]
    downsample = self._encoder_downsample_rate
    return [
        code[..., : -(-mask.sum() // downsample)].transpose(0, 1).to(device=device, dtype=torch.long)
        for code, mask in zip(audio_codes, padding_mask, strict=True)
    ]


def _load_weights_310p(self, weights):
    assert _original_load_weights is not None

    loaded = _original_load_weights(self, weights)
    encoder_device, encoder_dtype = audio_frontend_runtime(self.vllm_config.device_config.device)
    self.encoder.to(device=encoder_device, dtype=encoder_dtype)
    return loaded
