# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import torch

from vllm_omni.model_executor.models.indextts2.preprocess_utils import (
    _load_audio_1d,
    _resample,
    compute_fbank,
    load_campplus,
    load_wav2vec2,
    resolve_model_file,
)
from vllm_omni.model_executor.models.indextts2.s2mel.modules.audio import (
    mel_spectrogram,
)

from .configuration_confucius4_tts import (
    DEFAULT_MAX_REFERENCE_AUDIO_SECONDS,
    OUTPUT_SAMPLE_RATE,
)


def _stat_vector(value: object, name: str) -> torch.Tensor:
    vector = torch.as_tensor(value).squeeze()
    if vector.ndim != 1:
        raise ValueError(f"Confucius4-TTS Wav2Vec2-BERT {name} must be a vector")
    if not torch.isfinite(vector).all():
        raise ValueError(f"Confucius4-TTS Wav2Vec2-BERT {name} contains non-finite values")
    return vector


def normalize_wav2vec_stats(raw: object) -> torch.Tensor:
    if isinstance(raw, dict):
        if "mean" not in raw or "var" not in raw:
            raise ValueError("Confucius4-TTS Wav2Vec2-BERT stats dict must contain 'mean' and 'var'")
        mean = _stat_vector(raw["mean"], "mean")
        var = _stat_vector(raw["var"], "var")
    else:
        stats = torch.as_tensor(raw)
        if stats.ndim < 2 or stats.shape[0] != 2:
            raise ValueError("Confucius4-TTS Wav2Vec2-BERT stats must have shape [2, hidden_size]")
        mean = _stat_vector(stats[0], "mean")
        var = _stat_vector(stats[1], "var")
    if mean.shape != var.shape:
        raise ValueError(
            f"Confucius4-TTS Wav2Vec2-BERT mean/var shape mismatch: mean={tuple(mean.shape)} var={tuple(var.shape)}"
        )
    if (var < 0).any():
        raise ValueError("Confucius4-TTS Wav2Vec2-BERT variance must be non-negative")
    return torch.stack((mean, var)).float().contiguous()


def load_wav2vec_stats(model_path: str, filename: str) -> torch.Tensor:
    path = resolve_model_file(model_path, filename)
    if path is None:
        raise FileNotFoundError(f"Confucius4-TTS Wav2Vec2-BERT stats {filename!r} not found")
    raw = torch.load(path, map_location="cpu", weights_only=True)
    return normalize_wav2vec_stats(raw)


@torch.no_grad()
def _wav2vec_features(
    wav_16k: torch.Tensor,
    model: object,
    processor: object,
    device: torch.device,
    w2v_stats: torch.Tensor,
) -> torch.Tensor:
    """Layer-17 Wav2Vec2-BERT features, normalised with the checkpoint stats.

    Runs in float32 like the canonical implementation. The shared IndexTTS2
    helper uses bf16 autocast on CUDA, which shifts these features by about
    1.5% relative error and feeds straight into the speaker encoder.
    """
    inputs = processor(
        wav_16k.reshape(-1).float().cpu().numpy(),
        sampling_rate=16000,
        return_tensors="pt",
    )
    input_features = inputs["input_features"].to(device=device, dtype=torch.float32)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device=device)
    outputs = model(  # type: ignore[operator]
        input_features=input_features,
        attention_mask=attention_mask,
        output_hidden_states=True,
    )
    stats = w2v_stats.to(device=device, dtype=torch.float32)
    return (outputs.hidden_states[17].float() - stats[0:1]) / torch.sqrt(stats[1:2])


@torch.no_grad()
def extract_reference_features(
    audio: str | tuple | list,
    *,
    model_path: str,
    device: torch.device,
    w2v_stats: torch.Tensor,
    max_audio_length_seconds: float = DEFAULT_MAX_REFERENCE_AUDIO_SECONDS,
) -> dict[str, torch.Tensor]:
    # The canonical front-end resamples the 16 kHz and 22.05 kHz signals
    # independently from the source. ``load_reference_audio`` chains them
    # (source -> 22.05k -> 16k), which adds a second resampling error.
    wav, source_rate = _load_audio_1d(audio)
    if device.type == "cuda":
        wav = wav.to(device)
    max_samples = int(max_audio_length_seconds * source_rate) if max_audio_length_seconds else 0
    if max_samples > 0 and wav.shape[-1] > max_samples:
        wav = wav[..., :max_samples]
    wav_16k = _resample(wav, source_rate, 16000)
    wav_22k = _resample(wav, source_rate, OUTPUT_SAMPLE_RATE)

    w2v_model, processor = load_wav2vec2(model_path, device)
    condition_vector = _wav2vec_features(wav_16k, w2v_model, processor, device, w2v_stats)
    campplus = load_campplus(
        model_path,
        device,
        allow_remote_fallback=True,
    )
    style = campplus(compute_fbank(wav_16k, device))
    ref_mel = (
        mel_spectrogram(
            wav_22k.float().unsqueeze(0),
            1024,
            80,
            OUTPUT_SAMPLE_RATE,
            256,
            1024,
            0,
            None,
        )
        .transpose(1, 2)
        .contiguous()
    )
    return {
        "condition_vector": condition_vector,
        "style": style,
        "ref_mel": ref_mel,
    }
