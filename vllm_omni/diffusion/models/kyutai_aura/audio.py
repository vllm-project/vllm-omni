# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Speech features that condition AURA, sampled at the video frame rate.

AURA-1.3B uses the dequantized first (semantic) codebook of Kyutai's Mimi codec: 512-d at
12.5 Hz, linearly resampled to 25 Hz over the whole utterance.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile
import soxr
import torch
import torch.nn.functional as F

FEATURE_RATE = 25  # Hz, the rate of the features before they are sampled at the video frame rate
MIMI_REPO = "kyutai/mimi"
MIMI_SAMPLE_RATE = 24000


def load_speech(audio: str | Path | np.ndarray | tuple[np.ndarray, int], sample_rate: int) -> np.ndarray:
    """Mono float32 speech at ``sample_rate``. Accepts a file path, (samples, rate), or samples
    already at ``sample_rate``."""
    if isinstance(audio, (str, Path)):
        samples, rate = soundfile.read(str(audio), dtype="float32", always_2d=True)
        samples = samples.mean(axis=1)
    elif isinstance(audio, tuple):
        samples, rate = np.asarray(audio[0], dtype=np.float32), int(audio[1])
    else:
        samples, rate = np.asarray(audio, dtype=np.float32), sample_rate
    if samples.ndim > 1:
        samples = samples.reshape(-1)
    if rate != sample_rate:
        samples = soxr.resample(samples, rate, sample_rate, quality="HQ").astype(np.float32)
    return samples


class MimiSemanticFeatures:
    """Dequantized first Mimi codebook of a whole utterance: [1, T @ 25 Hz, 512] in float32."""

    def __init__(self, model_dir: str, device: torch.device) -> None:
        from transformers import MimiModel

        self.model = MimiModel.from_pretrained(model_dir, torch_dtype=torch.float32).to(device).eval()
        self.device = device
        self.sample_rate = int(self.model.config.sampling_rate)
        self.frame_rate = float(self.model.config.frame_rate)

    @torch.no_grad()
    def __call__(self, audio: str | Path | np.ndarray | tuple[np.ndarray, int]) -> torch.Tensor:
        """Speech (file path, (samples, rate) or samples at the Mimi rate) -> [1, T @ 25 Hz, 512]."""
        samples = load_speech(audio, self.sample_rate)
        wav = torch.from_numpy(samples).to(self.device)[None, None]
        codes = self.model.encode(wav, num_quantizers=1).audio_codes  # [1, 1, T @ 12.5 Hz]
        latent = self.model.quantizer.decode(codes)  # [1, 512, T]
        n = int(latent.shape[-1] / self.frame_rate * FEATURE_RATE)
        latent = F.interpolate(latent, size=n, mode="linear", align_corners=True)
        return latent.transpose(1, 2).float()


def to_video_rate(features: torch.Tensor, fps: int, video_frames_per_chunk: int) -> tuple[torch.Tensor, int]:
    """Sample [num_layers, T @ 25 Hz, dim] features at the video frame rate, zero-padded to a
    whole number of chunks. Returns ([1, num_layers, dim, num_chunks * frames], num_chunks)."""
    num_layers, t, dim = features.shape
    num_chunks = int(t / (video_frames_per_chunk * FEATURE_RATE / fps)) + 1
    n = num_chunks * video_frames_per_chunk
    index = np.round(np.linspace(0, n / fps, n, endpoint=False) * FEATURE_RATE).astype(np.int64)
    out = features.new_zeros(num_layers, dim, n)
    valid = index < t
    mask = torch.from_numpy(valid).to(features.device)
    picked = torch.from_numpy(index[valid]).to(features.device)
    out[:, :, mask] = features[:, picked].transpose(1, 2)
    return out.unsqueeze(0), num_chunks
