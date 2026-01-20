import base64
import io
import math
import os
from collections.abc import Iterable
from typing import Any

import librosa
import numpy as np
import pydub
import scipy.signal
import torch
import torch.nn as nn
from librosa.filters import mel as librosa_mel_fn
from pydub import AudioSegment
from vllm.logger import init_logger
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.request import OmniDiffusionRequest

from .constants import AUDIO_FORMAT_MAP, DEFAULT_FORMAT, FORMAT_MIME_MAP, SPEAKERS_LIST, VOLUME_LEVEL
from .hyperclovax_audio_decoder import HyperCLOVAXAudioDecoderModel

logger = init_logger(__name__)

# Global caches for mel filter banks and Hann windows.
mel_basis = {}
hann_window = {}


def get_hyperclovax_audio_post_process_func(od_config: OmniDiffusionConfig):
    """
    Get post-processing function for HyperCLOVAX Audio pipeline.

    Returns a function that converts model output tensors to audio file.
    """

    def post_process_func(output: list[tuple[torch.Tensor, str]]) -> list[bytes]:
        response = []
        for wav_tensor, fmt in output:
            wav = wav_tensor.squeeze().cpu().numpy()
            pcm = (wav * 32767.0).astype(np.int16)

            if fmt == "pcm":
                response.append(pcm.tobytes())
                continue

            segment = AudioSegment(pcm.tobytes(), frame_rate=24000, sample_width=pcm.dtype.itemsize, channels=1)

            buf = io.BytesIO()
            export_kwargs = {"format": fmt}
            segment.export(buf, **export_kwargs)
            response.append(buf.getvalue())

    return post_process_func


class HyperCLOVAXAudioPipeline(nn.Module):
    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        prefix: str = "",
    ):
        super().__init__()
        self.od_config = od_config
        self.device = get_local_device()
        self._dtype = od_config.dtype

        self.model = self.od_config.model
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=od_config.model, subfolder="bigvgan", revision=None, prefix=None, fall_back_to_pt=True
            )
        ]

        self.bigvgan = HyperCLOVAXAudioDecoderModel(od_config=od_config).to(self.device)

        self.spk_emb = self.bigvgan.spk_emb.to(self.device)
        self._vocab = int(getattr(self.bigvgan.h, "num_units", 0))

        speakers = SPEAKERS_LIST
        self.speaker_map = {spk: i for i, spk in enumerate(speakers)}

    def _prepare_batch(
        self, audio_tokens: list[list[int]], speakers: list[str], formats: list[str], ref_audio_tokens: list[str]
    ) -> list[tuple[torch.Tensor, torch.Tensor, str]]:
        """
        Construct batch to forward through the model.

        Args:
            - audio_tokens: List[List[int]]: discrete audio tokens to decode.
            - speakers: List[str]: speaker IDs for output audio.
            - formats: List[str]: output audio formats.
            - ref_audio_tokens: List[str]:
                List of base64 encoded reference audio.
                If provided, speaker and format will be ignored.

        Returns:
            batch: List of tuples of (audio_tokens, speaker_id or ref_mel, format)
        """
        batch = []
        for units, speaker, fmt, ref_audio in zip(audio_tokens, speakers, formats, ref_audio_tokens):
            units = torch.tensor(units, dtype=torch.long, device=self.device)

            if self._vocab > 0:
                mask = (units < 0) | (units >= self._vocab)
                if mask.any():
                    bad_idxs = units[mask].tolist()
                    raise ValueError(f"Unit indices out of range [0-{self._vocab - 1}]: {bad_idxs}")

            if ref_audio is not None:
                ref_audio_bytes = base64.b64decode(ref_audio.encode("ascii"), validate=True)
                ref_mel = (
                    self._get_reference_mel_spectrogram(ref_audio_bytes, self.bigvgan.h).to(self.device).to(self._dtype)
                )
                batch.append((units, ref_mel, None))
            else:
                speaker = "fkms" if speaker is None else speaker
                fmt = DEFAULT_FORMAT.lower() if fmt is None else fmt.lower()
                if fmt not in FORMAT_MIME_MAP:
                    raise ValueError(f"Unsupported format '{fmt}'. Choose from {list(FORMAT_MIME_MAP)}")
                speaker_id = torch.tensor([self.speaker_map[speaker]], dtype=torch.long)
                speaker_id = speaker_id.unsqueeze(0).to(self.device)

                batch.append((units, speaker_id, fmt))

        return batch

    def forward(self, req: OmniDiffusionRequest) -> DiffusionOutput:
        """
        Generate audio from audio tokens.

        Args:
            req: OmniDiffusionRequest must containing:
                - extra["audio_tokens"]: List[List[int]]: [B, L] or [L, ] audio token ids.
                - extra["speakers"]: List[str]: speaker for each audio sample.
                - extra["formats"]: List[str]: output audio format for each audio sample.
                - extra["ref_audio_tokens"]: List[str]: base64 encoded reference audio for each audio sample.

        Returns:
            OmniDiffusionResponse: The diffusion response.
        """

        # 1. Validate inputs exist in request
        audio_tokens = req.extra.get("audio_tokens")
        if audio_tokens is None:
            return DiffusionOutput(output=None, error="audio_tokens required in req.extra")

        speakers = req.extra.get("speakers")
        if speakers is None:
            return DiffusionOutput(output=None, error="speakers required in req.extra")

        if len(audio_tokens) != len(speakers):
            return DiffusionOutput(output=None, error="length of speakers and audio_tokens must be the same")

        # Optional: audio format. If not provided, use wav format as default.
        formats = req.extra.get("formats", [DEFAULT_FORMAT.lower()] * len(audio_tokens))
        if len(audio_tokens) != len(formats):
            return DiffusionOutput(output=None, error="length of formats and audio_tokens must be the same")

        ref_audio_tokens = req.extra.get("ref_audio_tokens")
        if len(audio_tokens) != len(ref_audio_tokens):
            return DiffusionOutput(output=None, error="length of ref_audio_tokens and audio_tokens must be the same")

        # 2. Construct batch from given request inputs
        batch = self._prepare_batch(audio_tokens, speakers, formats, ref_audio_tokens)
        results: list[tuple[torch.Tensor, str]] = []

        for units, speaker, fmt in batch:
            # 3. Convert to tensor if needed
            if isinstance(units, list):
                units = torch.tensor(units, dtype=torch.long)
            elif isinstance(units, np.ndarray):
                units = torch.from_numpy(units).long()

            if len(units.size()) == 2 and units.size(0) == 1:
                return DiffusionOutput(output=None, error="the underlying decoder does not support batch inference yet")

            units = units.unsqueeze(0)
            units = units.to(self.device)
            padded_unit, original_portion = self.pad(units)

            # 4. Generate speaker embedding
            spk_emb = self.spk_emb(speaker)

            # 5. Decode audio
            padded_out, hidden = self.bigvgan(padded_unit, spk_emb=spk_emb)
            del hidden
            out = self.unpad(padded_out, original_portion)

            # 6. Append decoded audio to result
            results.append((out.to(torch.float32), fmt))

        return DiffusionOutput(
            output=results, post_process_func=get_hyperclovax_audio_post_process_func(self.od_config)
        )

    def pad(self, unit: torch.Tensor) -> tuple[torch.Tensor, float]:
        """
        Pad the `unit` tensor to AUDIOLLM_PAD_MULTIPLE environment variable.

        Args:
            unit: int tensor of shape [1, L]
        """

        pad_multiple = self._get_pad_multiple()
        if not pad_multiple:
            return unit, 1.0

        pad_token_id = self._get_pad_token_id()
        if pad_token_id is None:
            return unit, 1.0

        overflow = unit.shape[1] % pad_multiple
        pad_amount = pad_multiple - overflow
        padded = torch.nn.functional.pad(unit, (0, pad_amount), mode="constant", value=pad_token_id)
        return padded, unit.shape[-1] / padded.shape[-1]

    def unpad(self, x: torch.Tensor, original_portion: float) -> torch.Tensor:
        """
        Unpad the `x` tensor by retaining only the `original_portion`.

        Args:
            x: tensor of shape [..., T]
            original_portion: ratio of original unit length over padded unit length
        """
        return x[..., : math.ceil(x.shape[-1] * original_portion)]

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """
        Load model weights using AutoWeightsLoader.
        """
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)

    def _get_pad_multiple(self) -> int | None:
        pad_multiple_str = os.getenv("AUDIOLLM_PAD_MULTIPLE", 100)
        if not pad_multiple_str:
            return None

        try:
            pad_multiple = int(pad_multiple_str)
        except ValueError:
            logger.warning("AUDIOLLM_PAD_MULTIPLE environment variable is not a valid int. Skipping padding...")
            return None

        if pad_multiple <= 0:
            logger.warning("AUDIOLLM_PAD_MULTIPLE environment variable is not a positive int. Skipping padding...")
            return None

        return pad_multiple

    def _get_pad_token_id(self) -> int | None:
        pad_token_id_str = os.getenv("AUDIOLLM_PAD_TOKEN_ID", 3894)
        if not pad_token_id_str:
            logger.warning("AUDIOLLM_PAD_TOKEN_ID environment variable is not set. Skipping padding...")
            return None

        try:
            pad_token_id = int(pad_token_id_str)
        except ValueError:
            logger.warning("AUDIOLLM_PAD_TOKEN_ID environment variable is not a valid int. Skipping padding...")
            return None

        if pad_token_id < 0:
            logger.warning("AUDIOLLM_PAD_TOKEN_ID environment variable is a negative int. Skipping padding...")
            return None

        return pad_token_id

    def _get_down_sample_rate(self) -> float | None:
        down_sample_rate_str = os.getenv("AUDIOLLM_DOWN_SAMPLE_RATE")
        if not down_sample_rate_str:
            return None

        try:
            down_sample_rate = float(down_sample_rate_str)
        except ValueError:
            logger.warning(
                "AUDIOLLM_DOWN_SAMPLE_RATE environment variable is not a valid float. Skipping down-sampling..."
            )
            return None

        if down_sample_rate <= 0:
            logger.warning(
                "AUDIOLLM_DOWN_SAMPLE_RATE environment variable is not a positive float. Skipping down-sampling..."
            )
            return None

        return down_sample_rate

    def _detect_audio_format(self, header_bytes: bytes) -> str | None:
        """
        Detect audio format from header bytes of audio file.

        Args:
            header_bytes: first 4 bytes of audio file.
        """
        for prefix_bytes, fmt in AUDIO_FORMAT_MAP:
            if header_bytes.startswith(prefix_bytes):
                return fmt
        return None

    def _hpf_normalize(self, pcm: np.ndarray, sr: int | float, volume_level: float) -> np.ndarray:
        assert (pcm**2).mean() > 0, "Error in the wav file"
        assert np.issubdtype(pcm.dtype, np.floating)

        # highpass filter
        filter_ = scipy.signal.butter(2, 70, "highpass", fs=sr, output="sos")
        pcm = scipy.signal.sosfilt(filter_, pcm)
        pcm = pcm.astype(np.float32)

        # volume normalize
        gain = min(volume_level / (pcm**2).mean() ** 0.5, 1 / np.max(np.abs(pcm)))
        pcm *= gain
        return pcm

    def _load_reference_audio(self, audio: bytes, sample_rate: float) -> np.ndarray:
        audio = io.BytesIO(audio)
        fmt = self._detect_audio_format(audio[:4])

        if fmt:
            segment = pydub.AudioSegment.from_file(audio, format=fmt)
        else:
            segment = pydub.AudioSegment.from_file(audio)

        wav_file = io.BytesIO()
        segment.export(wav_file, format="wav")
        wav_file.seek(0)

        # Down-sample to reduce noise in final result.
        load_sr = self._get_down_sample_rate()
        if load_sr is None:
            load_sr = sample_rate
        pcm, sr = librosa.load(wav_file, sr=load_sr, mono=True)
        pcm = librosa.resample(pcm, orig_sr=sr, target_sr=sample_rate)

        pcm = self._hpf_normalize(pcm, sample_rate, VOLUME_LEVEL)
        return pcm

    def _compute_mel_spectrogram(self, y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
        global mel_basis, hann_window
        # Create a unique key based on fmax and device
        key = f"{fmax}_{y.device}"
        if key not in mel_basis:
            mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
            mel_basis[key] = torch.from_numpy(mel).float().to(y.device)
            hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

        # Pad the signal for STFT
        pad_amount = int((n_fft - hop_size) / 2)
        y = torch.nn.functional.pad(y.unsqueeze(1), (pad_amount, pad_amount), mode="reflect").squeeze(1)

        # Compute the Short-Time Fourier Transform (STFT)
        spec = torch.stft(
            y,
            n_fft,
            hop_length=hop_size,
            win_length=win_size,
            window=hann_window[str(y.device)],
            center=center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )

        # Compute the magnitude spectrogram with a small epsilon to avoid log(0)
        spec = torch.sqrt(torch.real(spec * spec.conj() + 1e-9))

        # Map the linear-frequency spectrogram to the mel scale
        spec = torch.matmul(mel_basis[key], spec)

        # Apply spectral normalization (dynamic range compression)
        spec = torch.log(torch.clamp(spec, min=1e-5))

        return spec

    def _get_reference_mel_spectrogram(self, ref_audio: bytes, h: dict[str, Any]) -> torch.Tensor:
        pcm = self._load_reference_audio(ref_audio, h.sampling_rate)
        pcm = torch.from_numpy(pcm).unsqueeze(0)

        mel = self._compute_mel_spectrogram(
            pcm,
            h.n_fft,
            h.num_mels,
            h.sampling_rate,
            h.hop_size,
            h.win_size,
            h.fmin,
            h.fmax,
        )
        return mel
