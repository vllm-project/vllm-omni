# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Lazy, local-only Qwen3 2048D speaker conditioning on the serving CPU."""

from __future__ import annotations

import os
import threading
from pathlib import Path

import torch
import torch.nn.functional as F


class Zonos2SpeakerEncoder:
    model_id = "marksverdhei/Qwen3-Voice-Embedding-12Hz-1.7B"

    def __init__(self):
        self._model = None
        self._mel = None
        self._lock = threading.Lock()

    def _load(self):
        if self._model is not None:
            return
        try:
            import torchaudio
            from transformers import AutoModel
        except (ImportError, OSError) as exc:
            raise ImportError("ZONOS2 reference audio requires torch-matched torchaudio and transformers") from exc
        path = os.environ.get("VLLM_ZONOS2_SPEAKER_PATH", self.model_id)
        if path != self.model_id and not Path(path).is_dir():
            raise FileNotFoundError(f"ZONOS2 speaker encoder path does not exist: {path}")
        try:
            model = AutoModel.from_pretrained(path, trust_remote_code=True, local_files_only=True)
        except Exception as exc:
            raise RuntimeError(
                "ZONOS2 requires locally cached Qwen3 voice embedding weights and Python code; "
                "set VLLM_ZONOS2_SPEAKER_PATH to the complete snapshot (no automatic download)"
            ) from exc
        self._model = model.float().to("cpu").eval().requires_grad_(False)
        self._mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=24000,
            n_fft=1024,
            win_length=1024,
            hop_length=256,
            f_min=0.0,
            f_max=12000.0,
            n_mels=128,
            power=1.0,
            center=False,
            norm="slaney",
            mel_scale="slaney",
        )

    @torch.inference_mode()
    def encode(self, samples, sample_rate: int) -> torch.Tensor:
        wav = torch.as_tensor(samples, dtype=torch.float32, device="cpu")
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        elif wav.ndim == 2:
            wav = wav.mean(0, keepdim=True)
        else:
            raise ValueError("ZONOS2 reference audio must be mono or channels-first stereo")
        if sample_rate <= 0 or wav.numel() == 0 or not torch.isfinite(wav).all():
            raise ValueError("ZONOS2 reference audio must be nonempty and finite with a positive sample rate")
        if sample_rate != 24000:
            try:
                import torchaudio
            except (ImportError, OSError) as exc:
                raise ImportError("ZONOS2 reference audio requires torch-matched torchaudio") from exc
            wav = torchaudio.transforms.Resample(sample_rate, 24000)(wav)
        if wav.shape[-1] <= 384:
            raise ValueError("ZONOS2 reference audio must exceed 384 samples at 24kHz for reflection padding")
        with self._lock:
            self._load()
            assert self._model is not None and self._mel is not None
            mel = self._mel(F.pad(wav.unsqueeze(1), (384, 384), mode="reflect").squeeze(1))
            mel = torch.log(mel.clamp_min(1e-5)).transpose(1, 2)
            embedding = self._model(input_values=mel).last_hidden_state.float().reshape(-1)
        if embedding.shape != (2048,) or not torch.isfinite(embedding).all():
            raise RuntimeError("ZONOS2 speaker encoder must return a finite 2048D embedding")
        return embedding.cpu()
