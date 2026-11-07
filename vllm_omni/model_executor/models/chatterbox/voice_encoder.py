# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Adapted from https://github.com/CorentinJ/Real-Time-Voice-Cloning
# MIT License
"""Chatterbox's speaker encoder: a three-layer LSTM over 40-band mel windows.

Vendored from ``chatterbox.models.voice_encoder`` (0.1.7) with what the
reference conditioning reaches: ``embeds_from_mels`` with Resemble's partial
window rate, every partial window of a batch embedded at once.
"""

import numpy as np
import torch
from torch import nn

# Leftover frames past the last whole window form one more window when they
# cover at least this fraction of it; upstream's default.
MIN_COVERAGE = 0.8


class VoiceEncConfig:
    """The checkpoint's mel and network sizes (upstream's ``VoiceEncConfig``, as shipped)."""

    num_mels = 40
    sample_rate = 16000
    speaker_embed_size = 256
    ve_hidden_size = 256
    n_fft = 400
    hop_size = 160
    win_size = 400
    fmax = 8000
    fmin = 0
    mel_power = 2.0
    ve_partial_frames = 160


def get_num_wins(n_frames: int, step: int, hp: VoiceEncConfig) -> tuple[int, int]:
    """Partial windows over one mel, and the frames they span.

    Args:
        n_frames: The mel's frames.
        step: Frames between the starts of two windows.
        hp: The encoder's config.

    Returns:
        The window count and the frames the windows span, which may run past
        ``n_frames``.
    """
    win_size = hp.ve_partial_frames
    n_wins, remainder = divmod(max(n_frames - win_size + step, 0), step)
    if n_wins == 0 or (remainder + (win_size - step)) / win_size >= MIN_COVERAGE:
        n_wins += 1
    return n_wins, win_size + step * (n_wins - 1)


class VoiceEncoder(nn.Module):
    """The LSTM speaker encoder with upstream's parameter names.

    Args:
        hp: The encoder's config.
    """

    def __init__(self, hp: VoiceEncConfig) -> None:
        super().__init__()
        self.hp = hp
        self.lstm = nn.LSTM(self.hp.num_mels, self.hp.ve_hidden_size, num_layers=3, batch_first=True)
        self.proj = nn.Linear(self.hp.ve_hidden_size, self.hp.speaker_embed_size)
        # Training-only cosine-similarity scale; the checkpoint carries both.
        self.similarity_weight = nn.Parameter(torch.tensor([10.0]))
        self.similarity_bias = nn.Parameter(torch.tensor([-5.0]))

    def forward(self, mels: torch.Tensor) -> torch.Tensor:
        """Embed a batch of partial windows.

        Args:
            mels: Shape (B, ``ve_partial_frames``, ``num_mels``), unscaled mels.

        Returns:
            Shape (B, ``speaker_embed_size``), L2-normalized, nonnegative.
        """
        _, (hidden, _) = self.lstm(mels)
        raw_embeds = torch.relu(self.proj(hidden[-1]))
        return raw_embeds / torch.linalg.norm(raw_embeds, dim=1, keepdim=True)

    def inference(self, mels: torch.Tensor, mel_lens: list[int], rate: float) -> torch.Tensor:
        """Embed whole clips as the mean of their partial windows' embeddings.

        Args:
            mels: Shape (B, T, ``num_mels``), unscaled mels zero-padded to T.
            mel_lens: Each clip's own frame count.
            rate: Partial windows per second of audio.

        Returns:
            Shape (B, ``speaker_embed_size``), L2-normalized, on the CPU.
        """
        frame_step = int(np.round((self.hp.sample_rate / rate) / self.hp.ve_partial_frames))
        n_partials, target_lens = zip(*(get_num_wins(mel_len, frame_step, self.hp) for mel_len in mel_lens))

        # Windows that run past the longest clip read zeros.
        len_diff = max(target_lens) - mels.size(1)
        if len_diff > 0:
            pad = torch.full((mels.size(0), len_diff, self.hp.num_mels), 0, dtype=torch.float32)
            mels = torch.cat((mels, pad.to(mels.device)), dim=1)

        partials = torch.stack(
            [
                mel[i * frame_step : i * frame_step + self.hp.ve_partial_frames]
                for mel, n_partial in zip(mels, n_partials)
                for i in range(n_partial)
            ]
        )
        partial_embeds = self(partials).cpu()

        slices = np.concatenate(([0], np.cumsum(n_partials)))
        raw_embeds = torch.stack(
            [torch.mean(partial_embeds[start:end], dim=0) for start, end in zip(slices[:-1], slices[1:])]
        )
        return raw_embeds / torch.linalg.norm(raw_embeds, dim=1, keepdim=True)
