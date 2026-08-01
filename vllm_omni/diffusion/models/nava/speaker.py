# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn.functional as F
import torchaudio
from torch import nn

from vllm_omni.diffusion.models.nava.config import NAVAConfig


class NAVASpeakerEncoder(nn.Module):
    def __init__(self, model_root: str, config: NAVAConfig) -> None:
        super().__init__()
        self.embed_dim = config.speaker_embed_dim
        self.speaker_dir = os.path.join(model_root, config.speaker_dir)
        self.sample_rate = config.audio_sample_rate
        self._model: nn.Module | None = None

    def encode(self, wavs: list[Any], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if not wavs:
            return torch.empty(0, self.embed_dim, device=device, dtype=dtype)
        model = self._load_model(device)
        embeddings = []
        for wav in wavs:
            waveform = self._load_waveform(wav, device)
            with torch.inference_mode():
                embedding = model(waveform)
            embeddings.append(embedding.reshape(1, -1).to(device=device, dtype=dtype))
        return torch.cat(embeddings, dim=0)

    def _load_model(self, device: torch.device) -> nn.Module:
        if self._model is not None:
            return self._model

        kwargs = {
            "model_name": "M",
            "train_type": "ft_mix",
            "dataset": "vb2+vox2+cnc",
        }
        speaker_dir = self._resolve_speaker_dir()
        if speaker_dir is None:
            raise FileNotFoundError(
                "NAVA speaker timbre conditioning requires a local ReDimNet speaker encoder under "
                f"{self.speaker_dir} or TORCH_HOME/hub. Prepare the speaker assets before using spk_wavs."
            )
        model = torch.hub.load(speaker_dir, "ReDimNet", source="local", **kwargs)
        self._model = model.eval().to(device)
        return self._model

    def _resolve_speaker_dir(self) -> str | None:
        if os.path.exists(os.path.join(self.speaker_dir, "hubconf.py")):
            return self.speaker_dir
        torch_home = os.environ.get("TORCH_HOME")
        hub_root = os.path.join(torch_home, "hub") if torch_home else torch.hub.get_dir()
        for dirname in ("IDRnD_ReDimNet_main", "IDRnD_ReDimNet_master"):
            candidate = os.path.join(hub_root, dirname)
            if os.path.exists(os.path.join(candidate, "hubconf.py")):
                return candidate
        return None

    def _load_waveform(self, wav: Any, device: torch.device) -> torch.Tensor:
        if isinstance(wav, (str, os.PathLike)):
            waveform, sample_rate = self._load_waveform_file(os.fspath(wav))
        elif isinstance(wav, tuple) and len(wav) == 2:
            sample_rate, waveform = wav
            waveform = torch.as_tensor(waveform)
        else:
            waveform = torch.as_tensor(wav)
            sample_rate = self.sample_rate

        waveform = waveform.to(torch.float32)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.ndim != 2:
            raise ValueError(f"NAVA speaker reference must be waveform [C,T] or [T], got {tuple(waveform.shape)}.")
        if int(sample_rate) != self.sample_rate:
            waveform = self._resample(waveform, orig_freq=int(sample_rate), new_freq=self.sample_rate)
        waveform = waveform.mean(dim=0, keepdim=True)
        max_samples = int(30.0 * self.sample_rate)
        if waveform.shape[-1] > max_samples:
            waveform = waveform[..., :max_samples]
        if waveform.shape[-1] == 0:
            raise ValueError("NAVA speaker reference waveform is empty.")
        if waveform.shape[-1] < self.sample_rate:
            waveform = F.pad(waveform, (0, self.sample_rate - waveform.shape[-1]))
        return waveform.to(device)

    def _load_waveform_file(self, path: str) -> tuple[torch.Tensor, int]:
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        if os.path.isdir(path):
            raise IsADirectoryError(path)
        if not os.access(path, os.R_OK):
            raise PermissionError(path)

        try:
            import soundfile as sf
        except ImportError:
            return torchaudio.load(path)

        try:
            waveform, sample_rate = sf.read(path, dtype="float32", always_2d=True)
        except (FileNotFoundError, PermissionError, OSError):
            raise
        except sf.LibsndfileError:
            return torchaudio.load(path)
        return torch.from_numpy(waveform.T.copy()), int(sample_rate)

    @staticmethod
    def _resample(waveform: torch.Tensor, *, orig_freq: int, new_freq: int) -> torch.Tensor:
        return torchaudio.functional.resample(waveform, orig_freq=orig_freq, new_freq=new_freq)
