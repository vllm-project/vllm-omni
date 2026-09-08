# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Breeze's instruction and reference-audio templates for both public APIs."""

import math
from typing import Any

import numpy as np
import torch
from transformers import PreTrainedTokenizerBase
from vllm.inputs import TokensPrompt, tokens_input
from vllm.multimodal.audio import AudioResampler

DEFAULT_INSTRUCTION = "Speak clearly and naturally."
SAMPLE_RATE = 24000
ENCODE_DOWNSAMPLE_RATE = 1920
CFG_UNCOND_SUFFIX = "__cfg_uncond"


def build_breeze_prompt(
    tokenizer: PreTrainedTokenizerBase,
    text: str,
    instructions: str = DEFAULT_INSTRUCTION,
    *,
    ref_audio: tuple[np.ndarray, int] | None = None,
    ref_text: str | None = None,
    guidance_scale: float = 1.0,
    temperature: float = 0.9,
    top_k: int = 50,
    top_p: float = 1.0,
    repetition_penalty: float = 1.1,
) -> TokensPrompt:
    if not text.strip():
        raise ValueError("Breeze input text cannot be empty")
    if not math.isfinite(guidance_scale) or guidance_scale <= 0:
        raise ValueError("Breeze guidance_scale must be finite and positive")
    if top_k == -1:
        top_k = 0
    if (
        not all(math.isfinite(value) for value in (temperature, top_p, repetition_penalty))
        or temperature < 0
        or top_k < 0
        or not 0 < top_p <= 1
        or repetition_penalty <= 0
    ):
        raise ValueError("Invalid Breeze sampling parameters")
    if (ref_audio is None) != (ref_text is None):
        raise ValueError("Breeze voice cloning requires both ref_audio and ref_text")
    target = tokenizer.encode(f"[S0]<ins_bos>{instructions}<ins_eos>{text}", add_special_tokens=True)
    conditioning: dict[str, Any] = {"target_ids": target, "guidance_scale": guidance_scale, "role": "cond"}
    prefix_length = 0
    if ref_audio is not None:
        if not ref_text or not ref_text.strip():
            raise ValueError("Breeze reference transcript cannot be empty")
        waveform, sample_rate = ref_audio
        waveform = np.asarray(waveform, dtype=np.float32)
        if waveform.ndim == 2:
            # Public waveform input follows soundfile: (samples, channels).
            waveform = waveform.mean(axis=1)
        if waveform.ndim != 1 or not waveform.size or not np.isfinite(waveform).all() or sample_rate <= 0:
            raise ValueError("Breeze reference audio must contain finite samples at a positive sample rate")
        if sample_rate != SAMPLE_RATE:
            # Match the released tokenizer's librosa/soxr_hq resampling,
            # including ceil-length handling for non-integral rate ratios.
            length = math.ceil(waveform.size * SAMPLE_RATE / sample_rate)
            waveform = AudioResampler(target_sr=SAMPLE_RATE, method="soxr").resample(waveform, orig_sr=sample_rate)
            waveform = np.pad(waveform, (0, max(0, length - waveform.size)))[:length]
        reference_ids = tokenizer.encode(f"[S0]{ref_text}", add_special_tokens=True)
        frames = math.ceil(waveform.size / ENCODE_DOWNSAMPLE_RATE)
        conditioning.update(
            reference_ids=reference_ids,
            reference_frames=frames,
        )
        prefix_length = len(reference_ids) + frames + 1
    if guidance_scale != 1.0:
        conditioning["negative_ids"] = tokenizer.encode(f"[S0]{text}", add_special_tokens=True)
    prompt = tokens_input(prompt_token_ids=[0] * (prefix_length + len(target)))
    prompt["additional_information"] = {
        "breeze_prompt": conditioning,
        "breeze_sampling": {
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
        },
    }
    if ref_audio is not None:
        # Tensor payloads belong at the transport root (or in a declared
        # tensor namespace), rather than inside JSON-valued model metadata.
        prompt["additional_information"]["reference_waveform"] = torch.from_numpy(np.ascontiguousarray(waveform))
    if guidance_scale != 1.0:
        prompt["additional_information"]["cfg_group"] = {"role": "cond", "uncond_suffix": CFG_UNCOND_SUFFIX}
    return prompt
