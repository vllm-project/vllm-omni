# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-TTS voice cloning frontend: speech tokenizer, speaker embedding, mel features.

Lazy-loads external models (WhisperVQ, CampPlus ONNX) and provides extraction
helpers consumed by the GLM-TTS multimodal processor.
"""

from __future__ import annotations

import os

import torch
import torchaudio.compliance.kaldi as kaldi
import torchaudio.transforms as T
from vllm.logger import init_logger

logger = init_logger(__name__)

_MEL_BASIS_CACHE: dict[tuple[int, int, int, int, int, str], torch.Tensor] = {}
_HANN_WINDOW_CACHE: dict[tuple[int, str], torch.Tensor] = {}


def _spectral_normalize_torch(magnitudes: torch.Tensor) -> torch.Tensor:
    return torch.log(torch.clamp(magnitudes, min=1e-5))


def _get_glm_tts_mel_basis(
    *,
    device: torch.device,
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    f_min: int,
    f_max: int,
) -> torch.Tensor:
    """Compute or retrieve cached GLM-TTS mel filter bank.

    Uses torchaudio melscale_fbanks with Slaney parameters (norm="slaney",
    mel_scale="slaney") to match the official GLM-TTS ``utils/audio.py``
    librosa.filters.mel output.  The previous HTK / norm=None config
    produced a numerically different filter bank that shifted the
    prompt_feat distribution and degraded voice-clone conditioning.
    """
    cache_key = (sample_rate, n_fft, n_mels, f_min, f_max, str(device))
    mel_basis = _MEL_BASIS_CACHE.get(cache_key)
    if mel_basis is not None:
        return mel_basis

    import torchaudio.functional as AF

    mel_basis = (
        AF.melscale_fbanks(
            n_freqs=(n_fft // 2) + 1,
            f_min=float(f_min),
            f_max=float(f_max),
            n_mels=n_mels,
            sample_rate=sample_rate,
            norm="slaney",
            mel_scale="slaney",
        )
        .transpose(0, 1)
        .contiguous()
        .float()
        .to(device)
    )

    _MEL_BASIS_CACHE[cache_key] = mel_basis
    return mel_basis


def _extract_glm_tts_mel_feature(
    audio: torch.Tensor,
    *,
    sample_rate: int,
    n_fft: int,
    win_length: int,
    hop_length: int,
    n_mels: int,
    f_min: int,
    f_max: int,
) -> torch.Tensor:
    """Extract time-major log-mel spectrogram features from audio waveform."""
    device = audio.device
    mel_basis = _get_glm_tts_mel_basis(
        device=device,
        sample_rate=sample_rate,
        n_fft=n_fft,
        n_mels=n_mels,
        f_min=f_min,
        f_max=f_max,
    )

    hann_key = (win_length, str(device))
    window = _HANN_WINDOW_CACHE.get(hann_key)
    if window is None:
        window = torch.hann_window(win_length, device=device)
        _HANN_WINDOW_CACHE[hann_key] = window

    if audio.ndim == 1:
        audio = audio.unsqueeze(0)

    pad = int((n_fft - hop_length) / 2)
    audio = torch.nn.functional.pad(
        audio.unsqueeze(1),
        (pad, pad),
        mode="reflect",
    ).squeeze(1)

    spec = torch.stft(
        audio,
        n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=False,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    spec = torch.sqrt(spec.real.pow(2) + spec.imag.pow(2) + 1e-9)
    spec = torch.matmul(mel_basis, spec)
    spec = _spectral_normalize_torch(spec)
    return spec.squeeze(0).transpose(0, 1).contiguous()


def load_voice_clone_frontend(
    model_root: str,
    model_device: torch.device,
    *,
    speech_tokenizer_cache: tuple | None,
    campplus_cache: object | None,
    campplus_path: str | None = None,
) -> tuple[tuple | None, object | None]:
    """Lazy-load voice cloning frontend models.

    Args:
        campplus_path: Pre-resolved path to ``campplus.onnx``.  When provided
            this takes priority over the default ``model_root/frontend/``
            lookup.  The caller (``GLMTTSForConditionalGeneration``) resolves
            the path via ``transformers.utils.hub.cached_file`` during
            ``__init__`` so the file is already downloaded.

    Returns:
        ``(speech_tokenizer, campplus_session)`` — guaranteed non-None
        on success.  Raises ``RuntimeError`` when a required component
        cannot be loaded.
    """
    speech_tokenizer = speech_tokenizer_cache
    campplus_session = campplus_cache

    if speech_tokenizer is None:
        speech_tokenizer_paths = (
            os.path.join(model_root, "speech_tokenizer"),
            os.path.join(model_root, "ckpt", "speech_tokenizer"),
        )
        speech_tokenizer_path = next((path for path in speech_tokenizer_paths if os.path.isdir(path)), None)
        if speech_tokenizer_path is None:
            raise RuntimeError(
                f"GLM-TTS speech tokenizer directory not found at {speech_tokenizer_paths[0]} "
                f"or {speech_tokenizer_paths[1]}.  Voice cloning requires the WhisperVQ "
                f"speech tokenizer from the model snapshot."
            )
        try:
            from safetensors.torch import load_file
            from transformers import AutoFeatureExtractor, WhisperConfig

            from vllm_omni.model_executor.models.common.whisper_vq import (
                WhisperVQEncoder,
            )

            _config = WhisperConfig.from_pretrained(speech_tokenizer_path)
            _config.quantize_encoder_only = True
            _model = WhisperVQEncoder(_config)
            _sd = load_file(os.path.join(speech_tokenizer_path, "model.safetensors"))
            _prefix, _cleaned = "model.encoder.", {}
            for k, v in _sd.items():
                _cleaned[k[len(_prefix) :] if k.startswith(_prefix) else k] = v
            _model.load_state_dict(_cleaned, strict=False)
            _model = _model.to(model_device).eval()
            _fe = AutoFeatureExtractor.from_pretrained(speech_tokenizer_path)
            speech_tokenizer = (_model, _fe)
            logger.info("Loaded GLM-TTS speech tokenizer from %s", speech_tokenizer_path)
        except Exception as exc:
            raise RuntimeError(f"Failed to load GLM-TTS speech tokenizer from {speech_tokenizer_path}") from exc

    if campplus_session is None:
        cp = campplus_path or os.path.join(model_root, "frontend", "campplus.onnx")
        if not os.path.isfile(cp):
            raise RuntimeError(
                f"GLM-TTS campplus.onnx not found at {cp}.  "
                f"Voice cloning requires the CampPlus speaker embedding model."
            )
        try:
            import onnxruntime

            cuda_avail = "CUDAExecutionProvider" in onnxruntime.get_available_providers()
            cuda_ok = model_device.type == "cuda" and cuda_avail
            providers = (["CUDAExecutionProvider"] if cuda_ok else []) + ["CPUExecutionProvider"]
            campplus_session = onnxruntime.InferenceSession(cp, providers=providers)
            logger.info("Loaded GLM-TTS campplus ONNX from %s", cp)
        except Exception as exc:
            raise RuntimeError(f"Failed to load GLM-TTS campplus ONNX from {cp}") from exc

    return speech_tokenizer, campplus_session


def extract_prompt_speech_token(
    ref_audio_wav: torch.Tensor,
    ref_audio_sr: int,
    speech_tokenizer: tuple,
) -> list[int] | None:
    """Extract prompt speech tokens from reference audio using WhisperVQ."""

    model, feature_extractor = speech_tokenizer
    device = model.device

    audio = ref_audio_wav.float().to(device)
    if ref_audio_sr != 16000:
        resampler = T.Resample(orig_freq=ref_audio_sr, new_freq=16000).to(device)
        audio = resampler(audio)

    if audio.ndim > 1:
        audio = audio[0]

    audio_np = audio.cpu().numpy()
    pooling_kernel_size = getattr(model.config, "pooling_kernel_size", 1)
    stride = model.conv1.stride[0] * model.conv2.stride[0] * pooling_kernel_size * feature_extractor.hop_length

    all_tokens: list[int] = []
    time_step = 0
    while time_step * 16000 < audio_np.shape[0]:
        segment = audio_np[time_step * 16000 : (time_step + 30) * 16000]
        features = feature_extractor(
            [segment],
            sampling_rate=16000,
            return_attention_mask=True,
            return_tensors="pt",
            padding="longest",
            pad_to_multiple_of=stride,
        ).to(device)
        with torch.no_grad():
            outputs = model(**features)
        tokens = outputs.quantized_token_ids
        attn = features.attention_mask[:, :: model.conv1.stride[0] * model.conv2.stride[0]][:, ::pooling_kernel_size]
        all_tokens.extend(tokens[0][attn[0].bool()].tolist())
        time_step += 30

    return all_tokens if all_tokens else None


def extract_spk_embedding(
    ref_audio_wav: torch.Tensor,
    ref_audio_sr: int,
    campplus_session: object,
) -> list[float] | None:
    """Extract speaker embedding from reference audio using CampPlus ONNX."""

    audio = ref_audio_wav.float().cpu()
    if ref_audio_sr != 16000:
        resampler = T.Resample(orig_freq=ref_audio_sr, new_freq=16000)
        audio = resampler(audio)
    if audio.ndim > 1:
        audio = audio[0]

    feat = kaldi.fbank(audio.unsqueeze(0), num_mel_bins=80, dither=0, sample_frequency=16000)
    feat = feat - feat.mean(dim=0, keepdim=True)

    input_name = campplus_session.get_inputs()[0].name
    embedding = campplus_session.run(None, {input_name: feat.unsqueeze(dim=0).cpu().numpy()})[0].flatten().tolist()
    return embedding


def extract_prompt_feat(
    ref_audio_wav: torch.Tensor,
    ref_audio_sr: int,
    model_device: torch.device,
) -> torch.Tensor | None:
    """Extract mel features from reference audio for DiT conditioning."""
    audio = ref_audio_wav.float().to(model_device)
    if ref_audio_sr != 24000:
        resampler = T.Resample(orig_freq=ref_audio_sr, new_freq=24000).to(model_device)
        audio = resampler(audio)
    if audio.ndim > 1:
        audio = audio[0]

    with torch.no_grad():
        feat = _extract_glm_tts_mel_feature(
            audio,
            sample_rate=24000,
            n_fft=1920,
            win_length=1920,
            hop_length=480,
            n_mels=80,
            f_min=0,
            f_max=8000,
        )
    return feat  # [T_mel, mel_dim]
