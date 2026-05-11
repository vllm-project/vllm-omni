# SPDX-License-Identifier: Apache-2.0
"""Speaker encoder modules for Raon."""

from __future__ import annotations

import base64
import dataclasses
import io
import math
import os
import re
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.compliance.kaldi
import torchaudio.functional
from torch import nn

try:
    import soundfile
except ImportError:
    soundfile = None

from vllm.logger import init_logger

from vllm_omni.model_executor.models.raon.raon_utils import (
    module_device,
    module_dtype,
    unwrap_singleton_list,
)
from vllm_omni.transformers_utils.configs.raon import (
    AUDIO_SAMPLE_RATE,
    TARGET_ENCODER_SAMPLE_RATE,
    SpeakerEncoderConfig,
)

logger = init_logger(__name__)


# ECAPA-TDNN architecture constants — never overridden at runtime.
_ECAPA_MEL_DIM = 80
_ECAPA_ENC_DIM = 192
_ECAPA_CHANNELS = [1024, 1024, 1024, 1024, 3072]
_ECAPA_KERNEL_SIZES = [5, 3, 3, 3, 1]
_ECAPA_DILATIONS = [1, 2, 3, 4, 1]
_ECAPA_RES2NET_SCALE = 8
_ECAPA_SE_CHANNELS = 128
_ECAPA_ATTENTION_CHANNELS = 128


@dataclasses.dataclass
class _EcapaConfig:
    mel_dim: int = _ECAPA_MEL_DIM
    enc_dim: int = _ECAPA_ENC_DIM
    enc_channels: list[int] = dataclasses.field(default_factory=lambda: list(_ECAPA_CHANNELS))
    enc_kernel_sizes: list[int] = dataclasses.field(default_factory=lambda: list(_ECAPA_KERNEL_SIZES))
    enc_dilations: list[int] = dataclasses.field(default_factory=lambda: list(_ECAPA_DILATIONS))
    enc_res2net_scale: int = _ECAPA_RES2NET_SCALE
    enc_se_channels: int = _ECAPA_SE_CHANNELS
    enc_attention_channels: int = _ECAPA_ATTENTION_CHANNELS


class TimeDelayNetBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding="same",
            padding_mode="reflect",
        )
        self.norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.norm(self.activation(self.conv(hidden_states)))


class Res2NetBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale: int = 8,
        kernel_size: int = 3,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        in_channel = in_channels // scale
        hidden_channel = out_channels // scale
        self.blocks = nn.ModuleList(
            [
                TimeDelayNetBlock(
                    in_channel,
                    hidden_channel,
                    kernel_size=kernel_size,
                    dilation=dilation,
                )
                for _ in range(scale - 1)
            ]
        )
        self.scale = scale

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        outputs = []
        for i, hidden_part in enumerate(torch.chunk(hidden_states, self.scale, dim=1)):
            if i == 0:
                output_part = hidden_part
            elif i == 1:
                output_part = self.blocks[i - 1](hidden_part)
            else:
                output_part = self.blocks[i - 1](hidden_part + output_part)
            outputs.append(output_part)
        return torch.cat(outputs, dim=1)


class SqueezeExcitationBlock(nn.Module):
    def __init__(self, in_channels: int, se_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels,
            se_channels,
            kernel_size=1,
            padding="same",
            padding_mode="reflect",
        )
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(
            se_channels,
            out_channels,
            kernel_size=1,
            padding="same",
            padding_mode="reflect",
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states_mean = hidden_states.mean(dim=2, keepdim=True)
        hidden_states_mean = self.relu(self.conv1(hidden_states_mean))
        hidden_states_mean = self.sigmoid(self.conv2(hidden_states_mean))
        return hidden_states * hidden_states_mean


class SqueezeExcitationRes2NetBlock(nn.Module):
    """TDNN-Res2Net-TDNN-SE building block used in ECAPA-TDNN."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        res2net_scale: int = 8,
        se_channels: int = 128,
        kernel_size: int = 1,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.tdnn1 = TimeDelayNetBlock(in_channels, out_channels, kernel_size=1, dilation=1)
        self.res2net_block = Res2NetBlock(
            out_channels,
            out_channels,
            res2net_scale,
            kernel_size,
            dilation,
        )
        self.tdnn2 = TimeDelayNetBlock(out_channels, out_channels, kernel_size=1, dilation=1)
        self.se_block = SqueezeExcitationBlock(out_channels, se_channels, out_channels)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        residual = hidden_state
        hidden_state = self.tdnn1(hidden_state)
        hidden_state = self.res2net_block(hidden_state)
        hidden_state = self.tdnn2(hidden_state)
        hidden_state = self.se_block(hidden_state)
        return hidden_state + residual


class AttentiveStatisticsPooling(nn.Module):
    """Attentive statistic pooling layer: returns concatenated mean and std."""

    def __init__(self, channels: int, attention_channels: int = 128) -> None:
        super().__init__()
        self.tdnn = TimeDelayNetBlock(channels * 3, attention_channels, 1, 1)
        self.tanh = nn.Tanh()
        self.conv = nn.Conv1d(
            attention_channels,
            channels,
            kernel_size=1,
            padding="same",
            padding_mode="reflect",
        )

    @staticmethod
    def _length_to_mask(
        length: torch.Tensor,
        max_len: int | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        if max_len is None:
            max_len = length.max().long().item()
        mask = torch.arange(max_len, device=length.device, dtype=length.dtype).expand(
            len(length), max_len
        ) < length.unsqueeze(1)
        return torch.as_tensor(mask, dtype=dtype, device=device)

    @staticmethod
    def _compute_statistics(
        x: torch.Tensor,
        m: torch.Tensor,
        dim: int = 2,
        eps: float = 1e-12,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean = (m * x).sum(dim)
        std = torch.sqrt((m * (x - mean.unsqueeze(dim)).pow(2)).sum(dim).clamp(eps))
        return mean, std

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        seq_length = hidden_states.shape[-1]
        lengths = torch.ones(hidden_states.shape[0], device=hidden_states.device)
        mask = self._length_to_mask(
            lengths * seq_length, max_len=seq_length, dtype=hidden_states.dtype, device=hidden_states.device
        )
        mask = mask.unsqueeze(1)
        total = mask.sum(dim=2, keepdim=True)
        mean, std = self._compute_statistics(hidden_states, mask / total)
        mean = mean.unsqueeze(2).repeat(1, 1, seq_length)
        std = std.unsqueeze(2).repeat(1, 1, seq_length)
        attention = torch.cat([hidden_states, mean, std], dim=1)
        attention = self.conv(self.tanh(self.tdnn(attention)))
        attention = attention.masked_fill(mask == 0, float("-inf"))
        attention = F.softmax(attention, dim=2)
        mean, std = self._compute_statistics(hidden_states, attention)
        pooled_stats = torch.cat((mean, std), dim=1)
        return pooled_stats.unsqueeze(2)


class EcapaTdnnEncoder(nn.Module):
    """ECAPA-TDNN speaker encoder.

    Reference: "ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in
    TDNN Based Speaker Verification" (https://huggingface.co/papers/2005.07143).
    """

    def __init__(self, config: _EcapaConfig):
        super().__init__()
        if len(config.enc_channels) != len(config.enc_kernel_sizes) or len(config.enc_channels) != len(
            config.enc_dilations
        ):
            raise ValueError("enc_channels, enc_kernel_sizes and enc_dilations should have same length")
        self.channels = config.enc_channels
        self.blocks = nn.ModuleList()
        self.blocks.append(
            TimeDelayNetBlock(
                config.mel_dim,
                config.enc_channels[0],
                config.enc_kernel_sizes[0],
                config.enc_dilations[0],
            )
        )
        for i in range(1, len(config.enc_channels) - 1):
            self.blocks.append(
                SqueezeExcitationRes2NetBlock(
                    config.enc_channels[i - 1],
                    config.enc_channels[i],
                    res2net_scale=config.enc_res2net_scale,
                    se_channels=config.enc_se_channels,
                    kernel_size=config.enc_kernel_sizes[i],
                    dilation=config.enc_dilations[i],
                )
            )
        self.mfa = TimeDelayNetBlock(
            config.enc_channels[-1],
            config.enc_channels[-1],
            config.enc_kernel_sizes[-1],
            config.enc_dilations[-1],
        )
        self.asp = AttentiveStatisticsPooling(
            config.enc_channels[-1],
            attention_channels=config.enc_attention_channels,
        )
        self.asp_bn = nn.BatchNorm1d(config.enc_channels[-1] * 2)
        self.fc = nn.Conv1d(
            config.enc_channels[-1] * 2,
            config.enc_dim,
            kernel_size=1,
            padding="same",
            padding_mode="reflect",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states_list = []
        for layer in self.blocks:
            hidden_states = layer(hidden_states)
            hidden_states_list.append(hidden_states)
        hidden_states = torch.cat(hidden_states_list[1:], dim=1)
        hidden_states = self.mfa(hidden_states)
        hidden_states = self.asp(hidden_states)
        hidden_states = self.asp_bn(hidden_states)
        hidden_states = self.fc(hidden_states)
        return hidden_states.squeeze(-1)


_ECAPA_CONFIG = _EcapaConfig()


# Pretrained weights loaded from the SpeechBrain ECAPA-TDNN checkpoint
# (Apache-2.0, https://github.com/speechbrain/speechbrain).
# The encoder architecture above is a clean-room reimplementation from:
# Desplanques et al., "ECAPA-TDNN" (https://arxiv.org/abs/2005.07143).


def _map_speechbrain_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Map speechbrain ``embedding_model.ckpt`` keys to ``EcapaTdnnEncoder`` keys.

    The main difference is that speechbrain wraps Conv1d in an extra container,
    producing keys like ``blocks.0.conv.conv.weight`` where our module expects
    ``blocks.0.conv.weight``.  We collapse the extra wrapper level in three
    patterns:

    1. ``.conv.conv.`` → ``.conv.`` (double-wrapped Conv1d)
    2. ``fc.conv.`` → ``fc.`` (final fully-connected conv)
    3. ``.convN.conv.`` → ``.convN.`` (single-wrapped, e.g. SE block conv1/conv2)

    BatchNorm parameters (``.norm.norm.*``) are mapped to ``.norm.*`` to match
    the ``nn.BatchNorm1d`` layers in ``TimeDelayNetBlock``.
    """
    mapped: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = key
        new_key = new_key.replace(".conv.conv.", ".conv.")
        new_key = new_key.replace("fc.conv.", "fc.")
        # SE block: se_block.conv1.conv.{w,b} → se_block.conv1.{w,b}
        new_key = re.sub(r"(\.conv\d+)\.conv\.", r"\1.", new_key)
        # BatchNorm: .norm.norm.{weight,bias,...} → .norm.{weight,bias,...}
        new_key = new_key.replace(".norm.norm.", ".norm.")
        # ASP BatchNorm: asp_bn.norm.{weight,...} → asp_bn.{weight,...}
        new_key = new_key.replace("asp_bn.norm.", "asp_bn.")
        mapped[new_key] = value
    return mapped


def _load_ecapa_from_hf(model_id: str) -> EcapaTdnnEncoder:
    from huggingface_hub import hf_hub_download

    model = EcapaTdnnEncoder(_ECAPA_CONFIG)
    ckpt_path = hf_hub_download(repo_id=model_id, filename="embedding_model.ckpt")
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    mapped = _map_speechbrain_keys(state_dict)

    missing, unexpected = model.load_state_dict(mapped, strict=False)
    if missing:
        logger.debug("ECAPA-TDNN load: missing keys (expected): %s", missing)
    if unexpected:
        logger.debug("ECAPA-TDNN load: unexpected keys (batch-norm, expected): %s", unexpected)

    model.eval()
    return model


class EcapaSpeakerBackend(nn.Module):
    """Drop-in replacement for the speechbrain ``EncoderClassifier`` backend.

    Provides the same ``encode_batch(audio_16k, wav_lens)`` interface used by
    ``PretrainedSpeakerEncoder._extract_embedding``.
    """

    def __init__(self, ecapa: EcapaTdnnEncoder) -> None:
        super().__init__()
        self.ecapa = ecapa

    @staticmethod
    def _compute_fbank(waveform: torch.Tensor, sample_rate: int = 16000) -> torch.Tensor:
        """Compute 80-mel Fbank features matching the speechbrain pipeline.

        Reproduces ``speechbrain.lobes.features.Fbank`` with default ECAPA-TDNN
        settings: STFT(win_length=25ms, hop_length=10ms, n_fft=400,
        hamming_window, center=True) -> power spectrogram -> triangular mel
        filterbank (80 mels, 0-8000Hz) -> 10*log10 dB.
        """
        n_fft = 400
        win_length = int(round(sample_rate / 1000.0 * 25))  # 400 samples at 16kHz
        hop_length = int(round(sample_rate / 1000.0 * 10))  # 160 samples at 16kHz
        n_mels = 80
        f_min = 0
        f_max = sample_rate // 2

        window = torch.hamming_window(
            win_length,
            device=waveform.device,
            dtype=waveform.dtype,
        )

        feats_list = []
        for i in range(waveform.shape[0]):
            # STFT (center=True, pad_mode="constant")
            stft = torch.stft(
                waveform[i],
                n_fft,
                hop_length,
                win_length,
                window,
                center=True,
                pad_mode="constant",
                normalized=False,
                onesided=True,
                return_complex=True,
            )
            # Power spectrogram: |STFT|^2
            power_spec = stft.real.pow(2) + stft.imag.pow(2)  # [n_fft//2+1, T]
            power_spec = power_spec.transpose(0, 1)  # [T, n_fft//2+1]

            # Triangular mel filterbank
            n_stft = n_fft // 2 + 1
            mel_low = 2595.0 * math.log10(1.0 + f_min / 700.0)
            mel_high = 2595.0 * math.log10(1.0 + f_max / 700.0)
            mel_points = torch.linspace(
                mel_low,
                mel_high,
                n_mels + 2,
                device=waveform.device,
            )
            hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)
            all_freqs = torch.linspace(
                0,
                sample_rate // 2,
                n_stft,
                device=waveform.device,
            )

            # Build triangular filters
            fb = torch.zeros(
                n_stft,
                n_mels,
                device=waveform.device,
                dtype=waveform.dtype,
            )
            for m in range(n_mels):
                f_c = hz_points[m + 1]
                band_low = hz_points[m + 1] - hz_points[m]
                band_high = hz_points[m + 2] - hz_points[m + 1]
                # Rising slope
                mask_low = (all_freqs >= hz_points[m]) & (all_freqs <= f_c)
                fb[mask_low, m] = (all_freqs[mask_low] - hz_points[m]) / band_low.clamp(min=1e-10)
                # Falling slope
                mask_high = (all_freqs > f_c) & (all_freqs <= hz_points[m + 2])
                fb[mask_high, m] = (hz_points[m + 2] - all_freqs[mask_high]) / band_high.clamp(min=1e-10)

            # Apply filterbank + log10 dB scale (matching _amplitude_to_DB)
            mel_spec = torch.matmul(power_spec, fb)  # [T, n_mels]
            amin = 1e-10
            mel_spec = mel_spec.clamp(min=amin)
            mel_db = 10.0 * torch.log10(mel_spec)
            mel_db = mel_db.clamp(min=mel_db.max().item() - 80.0)

            feats_list.append(mel_db)

        max_len = max(f.shape[0] for f in feats_list)
        batch = torch.zeros(
            len(feats_list),
            max_len,
            n_mels,
            device=waveform.device,
            dtype=waveform.dtype,
        )
        for i, feat in enumerate(feats_list):
            batch[i, : feat.shape[0]] = feat
        return batch

    @staticmethod
    def _normalize(feats: torch.Tensor) -> torch.Tensor:
        """Per-utterance mean-only normalization (matches speechbrain std_norm=False)."""
        return feats - feats.mean(dim=1, keepdim=True)

    def encode_batch(self, audio_16k: torch.Tensor, wav_lens: torch.Tensor | None = None) -> torch.Tensor:
        """Produce speaker embeddings matching speechbrain's output shape ``[B, 1, 192]``."""
        feats = self._compute_fbank(audio_16k)
        feats = self._normalize(feats)
        feats = feats.to(device=next(self.ecapa.parameters()).device)
        embeddings = self.ecapa(feats)
        return embeddings.unsqueeze(1)


def _soundfile_to_tensor(source: str | io.BytesIO) -> tuple[torch.Tensor, int]:
    if soundfile is None:
        raise RuntimeError("soundfile is unavailable.")
    audio_np, sr = soundfile.read(source, dtype="float32", always_2d=True)
    audio = (
        torch.from_numpy(
            np.asarray(audio_np, dtype=np.float32),
        )
        .transpose(0, 1)
        .contiguous()
    )
    return audio, int(sr)


def _parse_data_url(data: str) -> io.BytesIO:
    """Decode a ``data:<mime>;base64,<payload>`` URL into a BytesIO buffer."""
    _, payload = data.split(",", 1)
    return io.BytesIO(base64.b64decode(payload))


class PretrainedSpeakerEncoder(nn.Module):
    """Speaker encoder using ECAPA-TDNN with a frozen backend and trainable projection."""

    def __init__(self, config: SpeakerEncoderConfig, dtype: torch.dtype | None = None) -> None:
        super().__init__()
        if config.pretrained_dim is None:
            raise ValueError(f"pretrained_dim must be set for encoder_type={config.encoder_type}")

        self.pretrained_model_id = config.pretrained_model_id
        self.pretrained_dim = int(config.pretrained_dim)
        self.output_size = int(config.output_size)
        self.source_sample_rate = AUDIO_SAMPLE_RATE
        self.target_sample_rate = TARGET_ENCODER_SAMPLE_RATE

        self.projection = nn.Linear(self.pretrained_dim, self.output_size, bias=False, dtype=dtype)

        self._backend: Any = None
        self._backend_device = torch.device("cpu")

    def _load_backend(self) -> None:
        if self._backend is not None:
            return

        model_id = self.pretrained_model_id or "speechbrain/spkrec-ecapa-voxceleb"
        ecapa = _load_ecapa_from_hf(model_id)
        backend = EcapaSpeakerBackend(ecapa)

        # Use object.__setattr__ to bypass any nn.Module __setattr__ interception
        # that would register _backend as a submodule (it must stay frozen/unregistered).
        object.__setattr__(self, "_backend", backend)
        for param in self._backend.parameters():
            param.requires_grad = False
        self._backend_device = torch.device("cpu")

    def warm_backend_artifacts(self) -> None:
        """Populate local caches for the pretrained backend without binding it."""
        model_id = self.pretrained_model_id or "speechbrain/spkrec-ecapa-voxceleb"
        ecapa = _load_ecapa_from_hf(model_id)
        del ecapa

    def _ensure_backend_device(self) -> None:
        if self._backend is None:
            self._load_backend()

        target_device = self.projection.weight.device
        if self._backend_device == target_device:
            return

        self._backend.to(target_device)
        self._backend_device = target_device
        logger.info("Speaker backend ready on %s", target_device)

    def _extract_embedding(
        self,
        audio_16k: torch.Tensor,
        lengths_16k: torch.Tensor,
    ) -> torch.Tensor:
        if self._backend is None:
            raise RuntimeError("Pretrained speaker backend is not initialized.")

        min_samples_16k = 1600
        batch = int(audio_16k.shape[0])
        if int(audio_16k.shape[1]) < min_samples_16k:
            return torch.zeros(
                batch,
                self.pretrained_dim,
                device=audio_16k.device,
                dtype=audio_16k.dtype,
            )
        lengths_16k = lengths_16k.clamp(min=min_samples_16k)
        wav_lens = lengths_16k.float() / lengths_16k.max().float()
        with torch.no_grad():
            embeddings = self._backend.encode_batch(audio_16k, wav_lens)
        return embeddings.squeeze(1)

    def forward(
        self,
        audio: torch.Tensor,
        audio_lengths: torch.Tensor,
    ) -> torch.Tensor:
        self._ensure_backend_device()
        if self._backend is None:
            raise RuntimeError("Pretrained speaker backend is unavailable.")

        if audio.ndim == 3:
            if int(audio.shape[1]) != 1:
                raise ValueError(
                    "Pretrained speaker encoder expects mono audio when 3D input is provided, "
                    f"got shape={tuple(audio.shape)}."
                )
            audio = audio[:, 0]
        elif audio.ndim != 2:
            raise ValueError(f"Pretrained speaker encoder expects audio as [B, T], got shape={tuple(audio.shape)}.")

        audio_lengths = audio_lengths.to(
            device=audio.device,
            dtype=torch.long,
        ).reshape(-1)
        if int(audio_lengths.shape[0]) != int(audio.shape[0]):
            raise ValueError(
                "audio_lengths batch mismatch for speaker encoder: "
                f"got {int(audio_lengths.shape[0])}, expected {int(audio.shape[0])}."
            )

        audio_16k = torchaudio.functional.resample(
            audio.float(),
            orig_freq=self.source_sample_rate,
            new_freq=self.target_sample_rate,
        )
        ratio = self.target_sample_rate / self.source_sample_rate
        lengths_16k = (audio_lengths.float() * ratio).long()
        lengths_16k = lengths_16k.clamp(min=1, max=int(audio_16k.shape[1]))

        raw_embedding = self._extract_embedding(audio_16k, lengths_16k)
        raw_embedding = raw_embedding.to(dtype=self.projection.weight.dtype)
        projected = self.projection(raw_embedding)
        return projected.unsqueeze(1)


def build_speaker_encoder(
    config: SpeakerEncoderConfig,
    dtype: torch.dtype | None = None,
) -> PretrainedSpeakerEncoder:
    return PretrainedSpeakerEncoder(config, dtype=dtype)


def normalize_speaker_ref_audio(value: Any) -> str | None:
    value = unwrap_singleton_list(value)

    if isinstance(value, dict):
        for key in ("speaker_ref_audio", "ref_audio", "audio_path", "path", "value"):
            if key in value:
                return normalize_speaker_ref_audio(value.get(key))
        return None

    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")

    if hasattr(value, "__fspath__"):
        value = os.fspath(value)  # type: ignore[arg-type]

    if not isinstance(value, str):
        return None

    normalized = value.strip()
    return normalized if normalized else None


def load_speaker_ref_audio(path: str) -> tuple[torch.Tensor, int] | None:
    normalized_path = path.strip()
    if not normalized_path:
        return None

    if normalized_path.startswith("data:"):
        try:
            buffer = _parse_data_url(normalized_path)
            if soundfile is not None:
                audio, sr = _soundfile_to_tensor(buffer)
            else:
                audio, sr = torchaudio.load(buffer)
                sr = int(sr)
            logger.info("Loaded ref from data URL: shape=%s sr=%s", tuple(audio.shape), sr)
            return audio, sr
        except Exception as exc:
            logger.warning("Failed to decode data URL ref audio: %s", exc)
            return None

    if not os.path.isfile(normalized_path):
        logger.warning("Ref audio path does not exist: %s", normalized_path[:200])
        return None

    # Try soundfile first for WAV (faster), fall back to torchaudio, then soundfile again.
    try:
        if normalized_path.lower().endswith(".wav") and soundfile is not None:
            audio, sr = _soundfile_to_tensor(normalized_path)
        else:
            audio, sr = torchaudio.load(normalized_path)
            sr = int(sr)
        logger.info(
            "Loaded ref audio: path=%s shape=%s sr=%s",
            normalized_path[:200],
            tuple(audio.shape),
            sr,
        )
        return audio, sr
    except Exception as exc:
        first_exc = exc

    # Secondary fallback: try the other backend.
    try:
        if soundfile is not None:
            audio, sr = _soundfile_to_tensor(normalized_path)
        else:
            audio, sr = torchaudio.load(normalized_path)
            sr = int(sr)
        return audio, sr
    except Exception as second_exc:
        logger.warning(
            "Failed to load ref audio %s (primary error=%s, fallback error=%s).",
            normalized_path,
            first_exc,
            second_exc,
        )
        return None


@torch.inference_mode()
def compute_speaker_embeds(
    speaker_encoder: PretrainedSpeakerEncoder,
    *,
    audio: torch.Tensor,
    audio_lengths: torch.Tensor | None = None,
    sampling_rate: int | None = None,
    model_sampling_rate: int,
) -> torch.Tensor:
    if speaker_encoder is None:
        raise AssertionError("speaker_encoder is not initialized on this checkpoint.")

    if audio.ndim == 1:
        audio = audio[None, None]
    elif audio.ndim == 2:
        audio = audio[:, None]
    elif audio.ndim != 3:
        raise ValueError(f"speaker audio must be 1D/2D/3D tensor, got shape={tuple(audio.shape)}.")

    target_device = module_device(speaker_encoder)
    target_dtype = module_dtype(speaker_encoder)
    audio = audio.to(device=target_device, dtype=target_dtype)

    if audio_lengths is None:
        audio_lengths = torch.full(
            (int(audio.shape[0]),),
            int(audio.shape[-1]),
            dtype=torch.long,
            device=audio.device,
        )
    else:
        audio_lengths = audio_lengths.to(
            device=audio.device,
            dtype=torch.long,
        ).reshape(-1)
        if int(audio_lengths.shape[0]) != int(audio.shape[0]):
            raise ValueError(
                "audio_lengths batch mismatch for speaker embeddings: "
                f"got {int(audio_lengths.shape[0])}, expected {int(audio.shape[0])}."
            )

    if sampling_rate is not None and sampling_rate != model_sampling_rate:
        audio = torchaudio.functional.resample(
            audio,
            orig_freq=int(sampling_rate),
            new_freq=int(model_sampling_rate),
        )
        sr_ratio = float(model_sampling_rate) / float(sampling_rate)
        audio_lengths = (audio_lengths.float() * sr_ratio).long()

    audio_lengths = audio_lengths.clamp(min=1, max=int(audio.shape[-1]))

    embeds = speaker_encoder(audio[:, 0], audio_lengths)
    return embeds
