import os
import io
import pydub
import json
import base64
import binascii
import math
import librosa
import scipy.signal
import torch
import torch.nn as nn
import numpy as np

from typing import Any, Dict, List, Iterable, Tuple, Optional
from librosa.filters import mel as librosa_mel_fn

from vllm.model_executor.models.utils import AutoWeightsLoader
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.request import OmniDiffusionRequest

from .hyperclovax_audio_decoder import HyperCLOVAXAudioTransformer

FORMAT_MIME_MAP = {
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
    "flac": "audio/flac",
    "ogg": "audio/ogg",
    "aac": "audio/aac",
    "pcm": "audio/pcm",
}

DEFAULT_FORMAT = "wav"

AUDIO_FORMAT_MAP = [
    (b"RIFF", "wav"),                      # WAV (RIFF container)
    (b"\x1a\x45\xdf\xa3", "webm"),         # WebM / MKV (EBML header)
    (b"OggS", "ogg"),                      # OGG
    (b"fLaC", "flac"),                     # FLAC
    (b"ID3", "mp3"),                       # MP3 with ID3 tag
    (b"\xff\xfb", "mp3"),                  # MP3 without ID3
    (b"\x00\x00\x00\x1c", "mp4"),          # MP4 / M4A
    (b"\x00\x00\x00\x20", "mp4"),          # MP4 / M4A
]

VOLUME_LEVEL_DB = -26
VOLUME_LEVEL = 10 ** (VOLUME_LEVEL_DB / 20)

# Global caches for mel filter banks and Hann windows.
mel_basis = {}
hann_window = {}

def get_hyperclovax_audio_post_process_func(od_config: OmniDiffusionConfig):
    """
    Get post-processing function for HyperCLOVAX Audio pipeline.

    Returns a function that converts model output tensors to audio file.
    """
    pass


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

        model = self.od_config.model
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=od_config.model,
                subfolder="transformer",
                revision=None,
                prefix="transformer.",
                fall_back_to_pt=True
            )
        ]
        self.transformer = HyperCLOVAXAudioTransformer(
            od_config=od_config
        )
        self.spk_emb = self.transformer.spk_emb
        self._vocab = int(getattr(self.transformer.model.h, "num_units", 0))

    def _extract(self, record: Dict[str, Any]) -> Tuple[Dict[str, Any], List[int], str]:
        """
        Decode and parse the incoming request record to a JSON payload.
        """
        data = record.get("data") or record.get("body")
        if isinstance(data, (bytes, bytearray)):
            data = data.decode("utf-8")
        elif isinstance(data, str):
            data = json.loads(data)

        units_list = data.get("unit")
        
        if not isinstance(units_list, list) or not all(
            isinstance(unit, int) for unit in units_list
        ):
            raise ValueError("Missing or invalid 'unit' field; must be a list of ints.")

        ref_audio = data.get("ref_audio", None)
        if ref_audio is None:
            return data, units_list, None

        if not isinstance(ref_audio, str):
            raise ValueError("Missing 'ref_audio' field; must be a base64 string.")

        try:
            ref_audio_bytes = base64.b64decode(ref_audio.encode("ascii"), validate=True)
        except binascii.Error:
            raise ValueError("Invalid 'ref_audio' fields; must be a base64 string.")

        return data, units_list, ref_audio_bytes


    def _prepare_batch(self, audio_tokens: List[Dict[str, Any]]) -> List[Tuple[torch.Tensor, torch.Tensor, str]]:
        batch = []
        for record in audio_tokens:
            payload, units, ref_audio = self._extract(record)
            units = torch.tensor(units, dtype=torch.long, device=self.device)

            if self._vocab > 0:
                mask = (units < 0) | (units >= self._vocab)
                if mask.any():
                    bad_idxs = units[mask].tolist()
                    raise ValueError(
                        f"Unit indices out of range [0-{self._vocab - 1}]: {bad_idxs}"
                    )
            
            if ref_audio is not None:
                ref_mel = (
                    self._get_reference_mel_spectrogram(ref_audio, self.model.model.h)
                    .to(self.device)
                    .to(self._dtype)
                )
                batch.append((units, ref_mel, None))
            else:
                speaker = str(payload.get("speaker", "fkms"))
                fmt = str(payload.get("format", DEFAULT_FORMAT)).lower()
                if fmt not in FORMAT_MIME_MAP:
                    raise ValueError(
                        f"Unsupported format '{fmt}'. Choose from {list(FORMAT_MIME_MAP)}"
                    )
                batch.append((units, speaker, fmt))

        return batch

    def forward(self, req: OmniDiffusionRequest) -> DiffusionOutput:
        """
        Generate audio from audio tokens.

        Args:
            req: OmniDiffusionRequest must containing:
                - extra["audio_tokens"]: List[Dict[str, Any]]

        Returns:
            OmniDiffusionResponse: The diffusion response.
        """
        audio_tokens = req.extra.get("audio_tokens")
        if audio_tokens is None:
            return DiffusionOutput(output=None, error="audio_tokens required in req.extra")
        
        results: List[Tuple[torch.Tensor, str]] = []
        batch = self._prepare_batch(audio_tokens)
        
        for units, speaker, fmt in batch:
            # Convert to tensor if needed
            if isinstance(units, list):
                units = torch.tensor(units, dtype=torch.long)
            elif isinstance(units, np.ndarray):
                units = torch.from_numpy(units).long()

            if len(units.size()) == 2 and units.size(0) == 1:
                return DiffusionOutput(output=None, error="the underlying decoder does not support batch inference yet")
            
            units = units.unsqueeze(0)
            units = units.to(self.device)
            padded_unit, original_portion = self.pad(units)

            if speaker is None:
                return DiffusionOutput(output=None, error="speaker required in req.extra.audio_tokens")
            else:
                if isinstance(speaker, list):
                    speaker = torch.tensor(speaker, dtype=torch.long)
                elif isinstance(speaker, np.ndarray):
                    speaker = torch.from_numpy(speaker).to(self.device)

            spk_emb = self.spk_emb(speaker)
            padded_out, hidden = self.transformer(padded_unit, spk_emb=spk_emb)
            del hidden
            
            out = self.unpad(padded_out, original_portion)
            results.append((out.to(torch.float32), fmt))

        return DiffusionOutput(output=results)

    def pad(self, unit: torch.Tensor) -> Tuple[torch.Tensor, float]:
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
        padded = torch.nn.functional.pad(
            unit, (0, pad_amount), mode="constant", value=pad_token_id
        )
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

    def _get_pad_multiple(self) -> Optional[int]:
        pad_multiple_str = os.getenv("AUDIOLLM_PAD_MULTIPLE", 100)
        if not pad_multiple_str:
            return None

        try:
            pad_multiple = int(pad_multiple_str)
        except ValueError:
            #logger.warning(
            #    "AUDIOLLM_PAD_MULTIPLE environment variable is not a valid int. Skipping padding..."
            #)
            return None

        if pad_multiple <= 0:
            #logger.warning(
            #    "AUDIOLLM_PAD_MULTIPLE environment variable is not a positive int. Skipping padding..."
            #)
            return None

        return pad_multiple

    def _get_pad_token_id(self) -> Optional[int]:
        pad_token_id_str = os.getenv("AUDIOLLM_PAD_TOKEN_ID", 3894)
        if not pad_token_id_str:
            #logger.warning(
            #    "AUDIOLLM_PAD_TOKEN_ID environment variable is not set. Skipping padding..."
            #)
            return None

        try:
            pad_token_id = int(pad_token_id_str)
        except ValueError:
            #logger.warning(
            #    "AUDIOLLM_PAD_TOKEN_ID environment variable is not a valid int. Skipping padding..."
            #)
            return None

        if pad_token_id < 0:
            #logger.warning(
            #    "AUDIOLLM_PAD_TOKEN_ID environment variable is a negative int. Skipping padding..."
            #)
            return None

        return pad_token_id
    
    def _get_down_sample_rate(self) -> Optional[float]:
        down_sample_rate_str = os.getenv("AUDIOLLM_DOWN_SAMPLE_RATE")
        if not down_sample_rate_str:
            return None

        try:
            down_sample_rate = float(down_sample_rate_str)
        except ValueError:
            #logger.warning(
            #    "AUDIOLLM_DOWN_SAMPLE_RATE environment variable is not a valid float. Skipping down-sampling..."
            #)
            return None

        if down_sample_rate <= 0:
            #logger.warning(
            #    "AUDIOLLM_DOWN_SAMPLE_RATE environment variable is not a positive float. Skipping down-sampling..."
            #)
            return None

        return down_sample_rate

    def _detect_audio_format(self, header_bytes: bytes) -> Optional[str]:
        for magic, fmt in AUDIO_FORMAT_MAP:
            if header_bytes.startswith(magic):
                return fmt
        return None

    def _hpf_normalize(
        self, pcm: np.ndarray, sr: Union[int, float], volume_level: float
    ) -> np.ndarray:
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
    
    def _compute_mel_spectrogram(self, y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, C=1, center=False):
        global mel_basis, hann_window
        # Create a unique key based on fmax and device
        key = f"{fmax}_{y.device}"
        if key not in mel_basis:
            mel = librosa_mel_fn(
                sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax
            )
            mel_basis[key] = torch.from_numpy(mel).float().to(y.device)
            hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

        # Pad the signal for STFT
        pad_amount = int((n_fft - hop_size) / 2)
        y = torch.nn.functional.pad(
            y.unsqueeze(1), (pad_amount, pad_amount), mode="reflect"
        ).squeeze(1)

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
        spec = torch.log(torch.clamp(x, min=1e-5))

        return spec
    
    def _get_reference_mel_spectrogram(self, ref_audio: bytes, h: Dict[str, Any]) -> torch.Tensor:
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