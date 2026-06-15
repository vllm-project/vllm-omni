# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Input transform utilities for the Audio-Omni diffusion pipeline.

Prompt templates, voice-reference loading, and optional TTS post-processing. Follows the
upstream gradio path (tail-crop ref to 6 s, pad to 10 s @ 44.1 kHz for the mel conditioner).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import torchaudio

SAMPLE_RATE = 44100
VOICE_PROMPT_SECONDS = 10
VOICE_PROMPT_MAX_SEC = 6.0

SYSTEM_TTS = "Generate speech from the input text."


def build_tts_prompt() -> str:
    return f"<|im_start|>system\n{SYSTEM_TTS}<|im_end|>\n<|im_start|>user\n<|im_end|>\n<|im_start|>assistant\n"


def build_speech_prompt(transcript: str, voice_ref_text: str | None) -> str:
    ref_text = (voice_ref_text or "").strip()
    if not ref_text:
        return transcript
    if not ref_text.endswith(". ") and not ref_text.endswith("。"):
        ref_text = ref_text.rstrip(".") + ". "
    return ref_text + transcript


def load_voice_prompt(path: str) -> tuple[torch.Tensor, float]:
    import soundfile

    data, sr = soundfile.read(path, dtype="float32", always_2d=True)  # [T, C]
    wav = torch.from_numpy(data.T)  # [C, T]
    if wav.dim() == 2 and wav.shape[0] == 2:
        wav = torch.mean(wav, dim=0, keepdim=True)

    duration = wav.shape[-1] / sr
    if duration > VOICE_PROMPT_MAX_SEC:
        wav = wav[..., -int(VOICE_PROMPT_MAX_SEC * sr) :]
        duration = VOICE_PROMPT_MAX_SEC

    if sr != SAMPLE_RATE:
        wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)(wav)
    if wav.dim() > 1:
        wav = wav.squeeze(0)
    wav = wav.to(torch.float32)

    sample_size = SAMPLE_RATE * VOICE_PROMPT_SECONDS
    if wav.shape[0] > sample_size:
        wav = wav[:sample_size]
    elif wav.shape[0] < sample_size:
        wav = F.pad(wav, (0, sample_size - wav.shape[0]), "constant", 0.0)
    return wav, float(duration)


def trim_silence(
    audio: torch.Tensor,
    sample_rate: int = SAMPLE_RATE,
    top_db: float = 30.0,
    frame_length: int = 2048,
    hop_length: int = 512,
    min_length_ms: int = 200,
) -> torch.Tensor:
    import numpy as np

    if audio.numel() == 0:
        return audio
    mono = (audio.float().mean(dim=0) if audio.dim() == 2 else audio.float()).cpu().numpy()
    n = len(mono)
    if n < frame_length:
        return audio
    num_frames = 1 + (n - frame_length) // hop_length
    energy = np.array([np.mean(mono[i * hop_length : i * hop_length + frame_length] ** 2) for i in range(num_frames)])
    energy_db = 10 * np.log10(np.maximum(energy, 1e-10))
    active = np.where(energy_db >= energy_db.max() - top_db)[0]
    if len(active) == 0:
        return audio

    start = int(active[0] * hop_length)
    end = int(min(active[-1] * hop_length + frame_length, n))
    min_samples = int(sample_rate * min_length_ms / 1000)
    if end - start < min_samples:
        center = (start + end) // 2
        start = max(0, center - min_samples // 2)
        end = min(n, start + min_samples)
    return audio[..., start:end]


def postprocess_tts_output(
    audio: torch.Tensor,
    sample_rate: int = SAMPLE_RATE,
    voice_ref_duration: float = 0.0,
) -> torch.Tensor:
    if voice_ref_duration > 0:
        ref_samples = int(voice_ref_duration * sample_rate)
        if ref_samples < audio.shape[-1]:
            audio = audio[..., ref_samples:]
    return trim_silence(audio, sample_rate)
