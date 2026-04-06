import base64
import io
import json
import math
import os
import tempfile
import zipfile
from collections.abc import Iterable
from pathlib import Path
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
from vllm_omni.diffusion.model_loader.diffusers_loader import (
    DiffusersPipelineLoader,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest

from .constants import (
    AUDIO_FORMAT_MAP,
    DEFAULT_FORMAT,
    FORMAT_MIME_MAP,
    SPEAKERS_LIST,
    VOLUME_LEVEL,
)
from .hyperclovax_audio_decoder import HyperCLOVAXAudioDecoderModel

logger = init_logger(__name__)

# Global caches for mel filter banks and Hann windows.
mel_basis = {}
hann_window = {}


def get_hyperclovax_audio_post_process_func(od_config: OmniDiffusionConfig):
    """Get post-processing function for HyperCLOVAX Audio pipeline."""

    def post_process_func(
        output: list[tuple[torch.Tensor, str]],
    ) -> list[bytes]:
        response = []
        for wav_tensor, fmt in output:
            wav = wav_tensor.squeeze().cpu().numpy()
            pcm = (wav * 32767.0).astype(np.int16)

            if fmt == "pcm":
                response.append(pcm.tobytes())
                continue

            segment = AudioSegment(pcm.tobytes(), frame_rate=24000, sample_width=pcm.dtype.itemsize, channels=1)
            buf = io.BytesIO()
            segment.export(buf, format=fmt if fmt is not None else "wav")
            response.append(buf.getvalue())

        # BUG FIX #1: Original PR #869 was missing this return statement
        return response

    return post_process_func


class HyperCLOVAXAudioPipeline(nn.Module):
    support_audio_output: bool = True

    @staticmethod
    def get_dummy_extra() -> dict:
        """Return dummy extra dict for warmup dummy run."""
        # Minimal dummy: one empty token sequence, default speaker
        return {"audio_tokens": [[0] * 10]}

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__()
        self.od_config = od_config
        self.device = get_local_device()
        self._dtype = od_config.dtype

        self.model = self.od_config.model
        self._using_mar_checkpoint = False
        self._mar_extract_dir: str | None = None

        # Default path: diffusers-style weights in bigvgan/ subfolder.
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=od_config.model,
                subfolder="bigvgan",
                revision=None,
                prefix="bigvgan.",
                fall_back_to_pt=True,
            )
        ]

        mar_path = self._resolve_mar_path(self.model)
        if mar_path is not None:
            self._using_mar_checkpoint = True
            self.weights_sources = []
            ckpt_path, config_path = self._extract_mar_checkpoint(mar_path)
            self.bigvgan = HyperCLOVAXAudioDecoderModel.from_pretrained(
                ckpt_path=ckpt_path,
                config_path=config_path,
                map_location="cpu",
            ).to(self.device)
        else:
            self.bigvgan = HyperCLOVAXAudioDecoderModel(od_config=od_config).to(self.device)

        self.spk_emb = self.bigvgan.spk_emb.to(self.device)
        self._vocab = int(getattr(self.bigvgan.h, "num_units", 0))

        speakers = SPEAKERS_LIST
        self.speaker_map = {spk: i for i, spk in enumerate(speakers)}

    def _resolve_mar_path(self, model: str | None) -> Path | None:
        if model is None:
            return None

        model_path = Path(model)
        if model_path.is_file() and model_path.suffix == ".mar":
            return model_path

        if not model_path.is_dir():
            return None

        candidates = [
            model_path / "NCCosybigvganDecoder.mar",
            model_path / "NCZSCosybigvganDecoder.mar",
            model_path / "decoder" / "audio" / "NCCosybigvganDecoder.mar",
            model_path / "decoder" / "audio" / "NCZSCosybigvganDecoder.mar",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def _extract_mar_checkpoint(self, mar_path: Path) -> tuple[str, str]:
        extract_dir = Path(tempfile.mkdtemp(prefix="hcx_audio_decoder_"))
        self._mar_extract_dir = str(extract_dir)

        with zipfile.ZipFile(mar_path) as zf:
            manifest = json.loads(zf.read("MAR-INF/MANIFEST.json"))
            serialized_file = manifest.get("model", {}).get("serializedFile")
            if not serialized_file:
                raise ValueError(f"serializedFile not found in {mar_path}")

            zf.extract(serialized_file, path=extract_dir)
            zf.extract("config.json", path=extract_dir)

        return str(extract_dir / serialized_file), str(extract_dir / "config.json")

    def _prepare_batch(
        self,
        audio_tokens: list[list[int]],
        speakers: list[str],
        formats: list[str],
        ref_audio_tokens: list[str | None],
    ) -> list[tuple[torch.Tensor, torch.Tensor, str | None]]:
        batch = []
        for units, speaker, fmt, ref_audio in zip(audio_tokens, speakers, formats, ref_audio_tokens):
            units = torch.tensor(units, dtype=torch.long, device=self.device)

            if self._vocab > 0:
                mask = (units < 0) | (units >= self._vocab)
                if mask.any():
                    bad_idxs = units[mask].tolist()
                    raise ValueError(f"Unit indices out of range [0-{self._vocab - 1}]: {bad_idxs}")

            if ref_audio is not None and not self.bigvgan.finetune:
                ref_audio_bytes = base64.b64decode(ref_audio.encode("ascii"), validate=True)
                ref_mel = (
                    self._get_reference_mel_spectrogram(ref_audio_bytes, self.bigvgan.h).to(self.device).to(self._dtype)
                )
                batch.append((units, ref_mel, None))
            elif ref_audio is None and not self.bigvgan.finetune:
                # Zero-shot decoder (ECAPA-TDNN) with no reference: use zero
                # mel as fallback so text-only requests don't crash.
                n_mels = int(getattr(self.bigvgan.h, "num_mels", 100))
                ref_mel = torch.zeros(1, n_mels, 64, device=self.device, dtype=self._dtype)
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
        audio_tokens = req.extra.get("audio_tokens")
        if audio_tokens is None:
            return DiffusionOutput(output=None, error="audio_tokens required in req.extra")

        # Default speakers to "fkms" for each sample when not provided
        # (e.g., when called from the pipeline stage processor).
        speakers = req.extra.get("speakers", ["fkms"] * len(audio_tokens))

        if len(audio_tokens) != len(speakers):
            return DiffusionOutput(output=None, error="length of speakers and audio_tokens must be the same")

        formats = req.extra.get("formats", [DEFAULT_FORMAT.lower()] * len(audio_tokens))
        if len(audio_tokens) != len(formats):
            return DiffusionOutput(output=None, error="length of formats and audio_tokens must be the same")

        # BUG FIX #2: Original PR #869 didn't handle None ref_audio_tokens,
        # causing len(None) TypeError
        ref_audio_tokens = req.extra.get("ref_audio_tokens", [None] * len(audio_tokens))
        if len(audio_tokens) != len(ref_audio_tokens):
            return DiffusionOutput(output=None, error="length of ref_audio_tokens and audio_tokens must be the same")

        batch = self._prepare_batch(audio_tokens, speakers, formats, ref_audio_tokens)
        results: list[tuple[torch.Tensor, str]] = []

        for units, speaker_or_mel, fmt in batch:
            if isinstance(units, list):
                units = torch.tensor(units, dtype=torch.long)
            elif isinstance(units, np.ndarray):
                units = torch.from_numpy(units).long()

            if len(units.size()) == 2 and units.size(0) == 1:
                return DiffusionOutput(output=None, error="the underlying decoder does not support batch inference yet")

            units = units.unsqueeze(0).to(self.device)
            padded_unit, original_portion = self.pad(units)

            if fmt is None:
                # ref_audio path: speaker_or_mel is a mel spectrogram
                # (float tensor). Only works with ECAPA_TDNN (finetune=False).
                if self.bigvgan.finetune:
                    return DiffusionOutput(
                        output=None, error="Reference audio requires finetune=False (ECAPA_TDNN speaker encoder)"
                    )
                spk_emb = self.spk_emb(speaker_or_mel)
            else:
                # speaker_id path: speaker_or_mel is a LongTensor
                spk_emb = self.spk_emb(speaker_or_mel)

            padded_out, hidden = self.bigvgan(padded_unit, spk_emb=spk_emb)
            del hidden
            out = self.unpad(padded_out, original_portion)

            results.append((out.to(torch.float32), fmt))

        return DiffusionOutput(output=results)

    def pad(self, unit: torch.Tensor) -> tuple[torch.Tensor, float]:
        pad_multiple = self._get_pad_multiple()
        if not pad_multiple:
            return unit, 1.0

        pad_token_id = self._get_pad_token_id()
        if pad_token_id is None:
            return unit, 1.0

        # BUG FIX #4: Original PR #869 always padded, even when already aligned.
        # When overflow==0, pad_amount was pad_multiple instead of 0.
        overflow = unit.shape[1] % pad_multiple
        if overflow == 0:
            return unit, 1.0
        pad_amount = pad_multiple - overflow
        padded = torch.nn.functional.pad(unit, (0, pad_amount), mode="constant", value=pad_token_id)
        return padded, unit.shape[-1] / padded.shape[-1]

    def unpad(self, x: torch.Tensor, original_portion: float) -> torch.Tensor:
        return x[..., : math.ceil(x.shape[-1] * original_portion)]

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # MAR checkpoint path already loads bigvgan weights eagerly.
        if self._using_mar_checkpoint:
            # Weights already loaded in __init__ via MAR extraction.
            # Return all parameter names to pass the strict loading check.
            return {name for name, _ in self.named_parameters()}

        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)

    def _get_pad_multiple(self) -> int | None:
        pad_multiple_str = os.getenv("AUDIOLLM_PAD_MULTIPLE", "100")
        if not pad_multiple_str:
            return None
        try:
            pad_multiple = int(pad_multiple_str)
        except ValueError:
            logger.warning("AUDIOLLM_PAD_MULTIPLE is not a valid int.")
            return None
        if pad_multiple <= 0:
            return None
        return pad_multiple

    def _get_pad_token_id(self) -> int | None:
        pad_token_id_str = os.getenv("AUDIOLLM_PAD_TOKEN_ID", "3894")
        if not pad_token_id_str:
            return None
        try:
            pad_token_id = int(pad_token_id_str)
        except ValueError:
            logger.warning("AUDIOLLM_PAD_TOKEN_ID is not a valid int.")
            return None
        if pad_token_id < 0:
            return None
        return pad_token_id

    def _get_down_sample_rate(self) -> float | None:
        down_sample_rate_str = os.getenv("AUDIOLLM_DOWN_SAMPLE_RATE")
        if not down_sample_rate_str:
            return None
        try:
            down_sample_rate = float(down_sample_rate_str)
        except ValueError:
            return None
        if down_sample_rate <= 0:
            return None
        return down_sample_rate

    def _detect_audio_format(self, header_bytes: bytes) -> str | None:
        for prefix_bytes, fmt in AUDIO_FORMAT_MAP:
            if header_bytes.startswith(prefix_bytes):
                return fmt
        return None

    def _hpf_normalize(self, pcm: np.ndarray, sr: int | float, volume_level: float) -> np.ndarray:
        assert (pcm**2).mean() > 0, "Error in the wav file"
        assert np.issubdtype(pcm.dtype, np.floating)
        filter_ = scipy.signal.butter(2, 70, "highpass", fs=sr, output="sos")
        pcm = scipy.signal.sosfilt(filter_, pcm)
        pcm = pcm.astype(np.float32)
        gain = min(volume_level / (pcm**2).mean() ** 0.5, 1 / np.max(np.abs(pcm)))
        pcm *= gain
        return pcm

    def _load_reference_audio(self, audio: bytes, sample_rate: float) -> np.ndarray:
        # BUG FIX #3: Original PR #869 tried audio[:4] on BytesIO object.
        # Must read header bytes BEFORE wrapping in BytesIO.
        header = audio[:4]
        audio_io = io.BytesIO(audio)
        fmt = self._detect_audio_format(header)

        if fmt:
            segment = pydub.AudioSegment.from_file(audio_io, format=fmt)
        else:
            segment = pydub.AudioSegment.from_file(audio_io)

        wav_file = io.BytesIO()
        segment.export(wav_file, format="wav")
        wav_file.seek(0)

        load_sr = self._get_down_sample_rate()
        if load_sr is None:
            load_sr = sample_rate
        pcm, sr = librosa.load(wav_file, sr=load_sr, mono=True)
        pcm = librosa.resample(pcm, orig_sr=sr, target_sr=sample_rate)
        pcm = self._hpf_normalize(pcm, sample_rate, VOLUME_LEVEL)
        return pcm

    def _compute_mel_spectrogram(
        self,
        y,
        n_fft,
        num_mels,
        sampling_rate,
        hop_size,
        win_size,
        fmin,
        fmax,
        center=False,
    ):
        global mel_basis, hann_window
        key = f"{fmax}_{y.device}"
        if key not in mel_basis:
            mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
            mel_basis[key] = torch.from_numpy(mel).float().to(y.device)
            hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

        pad_amount = int((n_fft - hop_size) / 2)
        y = torch.nn.functional.pad(y.unsqueeze(1), (pad_amount, pad_amount), mode="reflect").squeeze(1)

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
        spec = torch.sqrt(torch.real(spec * spec.conj() + 1e-9))
        spec = torch.matmul(mel_basis[key], spec)
        spec = torch.log(torch.clamp(spec, min=1e-5))
        return spec

    def _get_reference_mel_spectrogram(self, ref_audio: bytes, h: dict[str, Any]) -> torch.Tensor:
        pcm = self._load_reference_audio(ref_audio, h.sampling_rate)
        pcm = torch.from_numpy(pcm).unsqueeze(0)
        mel = self._compute_mel_spectrogram(
            pcm, h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax
        )
        return mel
