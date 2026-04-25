# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Audio utilities for F5-TTS inference.

Ported from F5-TTS:
  - MelSpec extraction (vocos / bigvgan backends)
  - Audio preprocessing (load, resample, RMS normalization)
  - Vocoder loading (vocos / bigvgan)
"""

from __future__ import annotations

import base64
import io
import logging
import urllib.request
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import librosa
import numpy as np
import soundfile as sf
import torch
import torchaudio
from pydub import AudioSegment, silence

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class MelSpecConfig:
    """Configuration for mel spectrogram extraction.

    Mirrors ``f5tts.config.MelSpecConfig``.
    """

    sample_rate: int = 24000
    hop_length: int = 256
    win_length: int = 1024
    n_mel_channels: int = 100
    n_fft: int = 1024
    mel_spec_type: str = "vocos"  # "vocos" | "bigvgan"
    wav_norm: bool = False
    mel_stats_path: str | None = None
    mel_norm_mode: str = "mean_std"  # "mean_std" | "mean"


# ---------------------------------------------------------------------------
# Mel spectrogram extraction
# ---------------------------------------------------------------------------


def get_vocos_mel_spectrogram(
    waveform: torch.Tensor,
    *,
    n_fft: int,
    n_mel_channels: int,
    sample_rate: int,
    hop_length: int,
    win_length: int,
) -> torch.Tensor:
    """Extract mel spectrogram using the Vocos recipe (torchaudio).

    Ported from ``f5tts.model.mel_spec.get_vocos_mel_spectrogram``.
    """
    mel_stft = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        n_mels=n_mel_channels,
        power=1,
        center=True,
        normalized=False,
        norm=None,
    ).to(waveform.device)

    if waveform.ndim == 3:
        waveform = waveform.squeeze(1)  # (B, 1, N) -> (B, N)
    assert waveform.ndim == 2

    mel = mel_stft(waveform)
    mel = mel.clamp(min=1e-5).log()
    return mel


def get_bigvgan_mel_spectrogram(
    waveform: torch.Tensor,
    *,
    n_fft: int,
    n_mel_channels: int,
    sample_rate: int,
    hop_length: int,
    win_length: int,
) -> torch.Tensor:
    """Extract mel spectrogram using the BigVGAN recipe.

    Ported from ``f5tts.model.mel_spec.get_bigvgan_mel_spectrogram``.
    Requires ``bigvgan`` to be installed.
    """
    try:
        from bigvgan.meldataset import mel_spectrogram as _bigvgan_mel  # type: ignore[import-untyped]
    except ImportError:
        raise ImportError(
            "bigvgan package is required for mel_spec_type='bigvgan'. Install it with: pip install bigvgan"
        )

    assert waveform.ndim == 2 and waveform.shape[0] == 1
    return _bigvgan_mel(
        waveform,
        n_fft=n_fft,
        num_mels=n_mel_channels,
        sampling_rate=sample_rate,
        hop_size=hop_length,
        win_size=win_length,
        fmin=0,
        fmax=None,
    )


class MelSpec:
    """Mel spectrogram extractor supporting vocos and bigvgan backends.

    Ported from ``f5tts.model.mel_spec.MelSpec``.

    Usage::

        mel_spec = MelSpec(MelSpecConfig())
        mel = mel_spec(waveform)  # waveform: [C, N] -> mel: [T, D]
    """

    def __init__(self, config: MelSpecConfig) -> None:
        if config.mel_spec_type not in ("vocos", "bigvgan"):
            raise ValueError(f"mel_spec_type must be 'vocos' or 'bigvgan', got {config.mel_spec_type!r}")
        self.config = config
        self._extractor = get_vocos_mel_spectrogram if config.mel_spec_type == "vocos" else get_bigvgan_mel_spectrogram

    @property
    def sample_rate(self) -> int:
        return self.config.sample_rate

    def __call__(self, wav: torch.Tensor) -> torch.Tensor:
        """Convert waveform to mel spectrogram.

        Args:
            wav: Audio tensor ``[C, N]`` (any number of channels, mono-mixed).

        Returns:
            Mel spectrogram ``[T, D]``  (time-first, float32).
        """
        # Mono-mix
        wav = wav.mean(0, keepdim=True)
        assert wav.ndim == 2 and wav.shape[0] == 1

        # Optional waveform normalization
        if self.config.wav_norm:
            wav_max = wav.abs().max()
            if wav_max > 0:
                wav = wav / wav_max * 0.6

        mel = self._extractor(
            waveform=wav,
            n_fft=self.config.n_fft,
            n_mel_channels=self.config.n_mel_channels,
            sample_rate=self.config.sample_rate,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length,
        )
        # (1, D, T) -> (T, D)
        mel = mel.squeeze(0).transpose(1, 0)
        assert mel.shape[-1] == self.config.n_mel_channels
        return mel


# ---------------------------------------------------------------------------
# Audio preprocessing (full pipeline)
# ---------------------------------------------------------------------------


def _is_probably_base64(s: str) -> bool:
    if s.startswith("data:audio"):
        return True
    if ("/" not in s and "\\" not in s) and len(s) > 256:
        return True
    return False


def _is_url(s: str) -> bool:
    try:
        u = urlparse(s)
        return u.scheme in ("http", "https") and bool(u.netloc)
    except Exception:
        return False


def _decode_base64_to_wav_bytes(b64: str) -> bytes:
    if "," in b64 and b64.strip().startswith("data:"):
        b64 = b64.split(",", 1)[1]
    return base64.b64decode(b64)


def _load_audio_to_np(x: str) -> tuple[np.ndarray, int]:
    if _is_url(x):
        with urllib.request.urlopen(x) as resp:
            audio_bytes = resp.read()
        with io.BytesIO(audio_bytes) as f:
            audio, sr = sf.read(f, dtype="float32", always_2d=False)
    elif _is_probably_base64(x):
        wav_bytes = _decode_base64_to_wav_bytes(x)
        with io.BytesIO(wav_bytes) as f:
            audio, sr = sf.read(f, dtype="float32", always_2d=False)
    else:
        audio, sr = librosa.load(x, sr=None, mono=True)

    if audio.ndim > 1:
        audio = np.mean(audio, axis=-1)

    return audio.astype(np.float32), int(sr)


def _audio_segment_from_np(audio: np.ndarray, sample_rate: int) -> AudioSegment:
    clipped = np.clip(audio, -1.0, 1.0)
    pcm = (clipped * np.iinfo(np.int16).max).astype(np.int16)
    return AudioSegment(
        data=pcm.tobytes(),
        sample_width=pcm.dtype.itemsize,
        frame_rate=sample_rate,
        channels=1,
    )


def _audio_segment_to_np(audio: AudioSegment) -> tuple[np.ndarray, int]:
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
    if audio.channels > 1:
        samples = samples.reshape(-1, audio.channels).mean(axis=1)
    scale = float(1 << (8 * audio.sample_width - 1))
    return (samples / scale).astype(np.float32), int(audio.frame_rate)


def _remove_silence_edges(audio: AudioSegment, silence_threshold_db: float = -42.0) -> AudioSegment:
    non_silent_start_idx = silence.detect_leading_silence(
        audio,
        silence_threshold=silence_threshold_db,
    )
    audio = audio[non_silent_start_idx:]

    reversed_audio = audio.reverse()
    non_silent_end_idx = silence.detect_leading_silence(
        reversed_audio,
        silence_threshold=silence_threshold_db,
    )
    if non_silent_end_idx > 0:
        return audio[: len(audio) - non_silent_end_idx]
    return audio


def _preprocess_reference_audio(
    audio: np.ndarray,
    sample_rate: int,
) -> tuple[np.ndarray, int]:
    """Apply the F5 reference clipping pass before RMS normalization."""
    aseg = _audio_segment_from_np(audio, sample_rate)

    non_silent_segs = silence.split_on_silence(
        aseg,
        min_silence_len=1000,
        silence_thresh=-50,
        keep_silence=1000,
        seek_step=10,
    )
    non_silent_wave = AudioSegment.silent(duration=0)
    for seg in non_silent_segs:
        if len(non_silent_wave) > 6000 and len(non_silent_wave + seg) > 12000:
            break
        non_silent_wave += seg

    if len(non_silent_wave) > 12000:
        non_silent_segs = silence.split_on_silence(
            aseg,
            min_silence_len=100,
            silence_thresh=-40,
            keep_silence=1000,
            seek_step=10,
        )
        non_silent_wave = AudioSegment.silent(duration=0)
        for seg in non_silent_segs:
            if len(non_silent_wave) > 6000 and len(non_silent_wave + seg) > 12000:
                break
            non_silent_wave += seg

    aseg = non_silent_wave if len(non_silent_wave) > 0 else aseg
    if len(aseg) > 12000:
        aseg = aseg[:12000]

    aseg = _remove_silence_edges(aseg) + AudioSegment.silent(duration=50)
    return _audio_segment_to_np(aseg)


def process_audio(
    audio_path: str,
    *,
    mel_spec: MelSpec,
    target_rms: float | None = 0.1,
) -> tuple[torch.Tensor, float]:
    """Full audio preprocessing: load → RMS norm → resample → mel.

    Ported from ``f5tts.infer.model.process_audio``.

    Args:
        audio_path: Reference audio path or URI.
        mel_spec: MelSpec extractor instance.
        target_rms: Target RMS level for normalization.  ``None`` disables.

    Returns:
        ``(mel, audio_rms)`` where ``mel`` is ``[T, D]`` float32 and
        ``audio_rms`` is the original RMS of the audio before normalization.
    """
    # # Load audio via soundfile (avoids torchcodec dependency)
    # with io.BytesIO(audio_bytes) as f:
    #     data, audio_sr = sf.read(f, dtype="float32")
    #     # data, audio_sr = librosa.load(audio_bytes, sr=None, mono=True)
    # # soundfile returns (samples,) for mono or (samples, channels) for multi-channel

    data, audio_sr = _load_audio_to_np(audio_path)
    data, audio_sr = _preprocess_reference_audio(data, audio_sr)

    audio = torch.from_numpy(data)

    if audio.ndim == 1:
        audio = audio.unsqueeze(0)  # [1, T]
    else:
        audio = audio.T  # [channels, T]

    # RMS normalization
    audio_rms = torch.sqrt(torch.mean(torch.square(audio))).item()
    if target_rms is not None and audio_rms < target_rms:
        audio = audio * target_rms / audio_rms

    # Check minimum duration
    duration = audio.shape[-1] / audio_sr
    if duration < 0.3:
        logger.warning("conditioning audio is very short (%.2fs)", duration)

    # Resample to target sample rate
    if audio_sr != mel_spec.config.sample_rate:
        resampler = torchaudio.transforms.Resample(audio_sr, mel_spec.config.sample_rate)
        audio = resampler(audio)

    # Convert to mel spectrogram
    mel = mel_spec(audio)  # [T, D]
    assert mel.dtype == torch.float32 or mel.dtype == torch.float64
    mel = mel.float()

    return mel, audio_rms


# ---------------------------------------------------------------------------
# Vocoder loading
# ---------------------------------------------------------------------------


def load_vocoder(
    vocoder_name: str,
    *,
    sample_rate: int = 24000,
    device: str | torch.device = "cpu",
    hf_cache_dir: str | None = None,
) -> Any:
    """Load a neural vocoder for mel-to-waveform conversion.

    Ported from ``f5tts.infer.utils_infer.load_vocoder``.

    Args:
        vocoder_name: ``"vocos"`` or ``"bigvgan"``.
        sample_rate: Expected sample rate (must be 24000 for both vocoders).
        device: Torch device to place the vocoder on.
        hf_cache_dir: Optional HuggingFace cache directory.

    Returns:
        The loaded vocoder model (ready for inference).
    """
    if vocoder_name == "vocos":
        return _load_vocos_vocoder(sample_rate=sample_rate, device=str(device))
    if vocoder_name == "bigvgan":
        return _load_bigvgan_vocoder(
            sample_rate=sample_rate,
            device=str(device),
            hf_cache_dir=hf_cache_dir,
        )
    raise ValueError(f"Unknown vocoder_name={vocoder_name!r}, expected 'vocos' or 'bigvgan'")


def _load_vocos_vocoder(*, sample_rate: int, device: str) -> Any:
    """Load Vocos vocoder from HuggingFace.

    Ported from ``f5tts.infer.vocoders.load_vocos_vocoder``.
    """
    if sample_rate != 24000:
        raise ValueError(f"vocos vocoder expects sample_rate=24000, got {sample_rate}")

    try:
        from huggingface_hub import hf_hub_download
        from vocos import Vocos  # type: ignore[import-untyped]
        from vocos.feature_extractors import EncodecFeatures  # type: ignore[import-untyped]
    except ImportError:
        raise ImportError("vocos is required for vocoder_name='vocos'. Install with: pip install vocos huggingface-hub")

    repo_id = "charactr/vocos-mel-24khz"
    config_path = hf_hub_download(repo_id=repo_id, filename="config.yaml")
    model_path = hf_hub_download(repo_id=repo_id, filename="pytorch_model.bin")

    vocoder = Vocos.from_hparams(config_path)
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)

    if isinstance(vocoder.feature_extractor, EncodecFeatures):
        encodec_parameters = {
            "feature_extractor.encodec." + key: value
            for key, value in vocoder.feature_extractor.encodec.state_dict().items()
        }
        state_dict.update(encodec_parameters)

    vocoder.load_state_dict(state_dict)
    return vocoder.eval().to(device)


def _load_bigvgan_vocoder(
    *,
    sample_rate: int,
    device: str,
    hf_cache_dir: str | None = None,
) -> Any:
    """Load BigVGAN vocoder from HuggingFace.

    Ported from ``f5tts.infer.vocoders.load_bigvgan_vocoder``.
    """
    if sample_rate != 24000:
        raise ValueError(f"bigvgan vocoder expects sample_rate=24000, got {sample_rate}")

    try:
        from bigvgan import BigVGAN  # type: ignore[import-untyped]
    except ImportError:
        raise ImportError("bigvgan is required for vocoder_name='bigvgan'. Install with: pip install bigvgan")

    vocoder = BigVGAN.from_pretrained(
        "nvidia/bigvgan_v2_24khz_100band_256x",
        use_cuda_kernel=False,
        cache_dir=hf_cache_dir,
    )
    vocoder.remove_weight_norm()
    return vocoder.eval().to(device)


def run_vocoder(
    mel: torch.Tensor,
    vocoder: Any,
    vocoder_name: str,
) -> torch.Tensor:
    """Run vocoder to convert mel spectrogram to waveform.

    Args:
        mel: Mel spectrogram ``[B, D, T]`` (channel-first, as vocoders expect).
        vocoder: Loaded vocoder model.
        vocoder_name: ``"vocos"`` or ``"bigvgan"``.

    Returns:
        Waveform tensor ``[B, N]``.
    """
    with torch.inference_mode():
        # Vocos/BigVGAN vocoders work best in float32; cast mel and
        # ensure the vocoder is also float32 to avoid dtype mismatches.
        mel = mel.float()
        if vocoder_name == "vocos":
            vocoder.float()
            wav = vocoder.decode(mel)
        elif vocoder_name == "bigvgan":
            vocoder.float()
            wav = vocoder(mel)
            if wav.ndim == 3:
                wav = wav.squeeze(1)
        else:
            raise ValueError(f"Unknown vocoder_name={vocoder_name!r}")
    return wav
