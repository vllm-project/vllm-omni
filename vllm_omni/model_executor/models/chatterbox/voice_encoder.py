# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Adapted from https://github.com/CorentinJ/Real-Time-Voice-Cloning
# MIT License

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class VoiceEncConfig:
    num_mels = 40
    sample_rate = 16000
    speaker_embed_size = 256
    ve_hidden_size = 256
    n_fft = 400
    hop_size = 160
    win_size = 400
    fmax = 8000
    fmin = 0
    preemphasis = 0.0
    mel_power = 2.0
    mel_type = "amp"
    ve_partial_frames = 160
    ve_final_relu = True
    stft_magnitude_min = 1e-4


def get_num_wins(
    n_frames: int,
    step: int,
    min_coverage: float,
    hp: VoiceEncConfig,
):
    assert n_frames > 0
    win_size = hp.ve_partial_frames
    n_wins, remainder = divmod(max(n_frames - win_size + step, 0), step)
    if n_wins == 0 or (remainder + (win_size - step)) / win_size >= min_coverage:
        n_wins += 1
    target_n = win_size + step * (n_wins - 1)
    return n_wins, target_n


def get_frame_step(
    overlap: float,
    rate: float | None,
    hp: VoiceEncConfig,
):
    # Compute how many frames separate two partial utterances
    assert 0 <= overlap < 1
    if rate is None:
        frame_step = int(np.round(hp.ve_partial_frames * (1 - overlap)))
    else:
        frame_step = int(np.round((hp.sample_rate / rate) / hp.ve_partial_frames))
    assert 0 < frame_step <= hp.ve_partial_frames
    return frame_step


class VoiceEncoder(nn.Module):
    def __init__(self, hp=VoiceEncConfig()):
        super().__init__()

        self.hp = hp

        # Network definition
        self.lstm = nn.LSTM(self.hp.num_mels, self.hp.ve_hidden_size, num_layers=3, batch_first=True)
        self.proj = nn.Linear(self.hp.ve_hidden_size, self.hp.speaker_embed_size)

        # Cosine similarity scaling (fixed initial parameter values)
        self.similarity_weight = nn.Parameter(torch.tensor([10.0]), requires_grad=True)
        self.similarity_bias = nn.Parameter(torch.tensor([-5.0]), requires_grad=True)

    def forward(self, mels: torch.FloatTensor):
        """
        Computes the embeddings of a batch of partial utterances.

        Args:
            mels: A batch of unscaled mel spectrograms of same duration as a float32 tensor
                of shape (B, T, M) where T is hp.ve_partial_frames.

        Returns:
            embeds (torch.Tensor): The embeddings as a float32 tensor of shape (B, E) where E is
                hp.speaker_embed_size. Embeddings are L2-normed and thus lay in the range [-1, 1].
        """
        # Pass the input through the LSTM layers
        _, (hidden, _) = self.lstm(mels)

        # Project the final hidden state
        raw_embeds = self.proj(hidden[-1])
        if self.hp.ve_final_relu:
            raw_embeds = F.relu(raw_embeds)

        # L2 normalize the embeddings.
        return raw_embeds / torch.linalg.norm(raw_embeds, dim=1, keepdim=True)

    def inference(
        self, mels: torch.Tensor, mel_lens, overlap=0.5, rate: float = None, min_coverage=0.8, batch_size=None
    ):
        """
        Computes the embeddings of a batch of full utterances with gradients.

        Args:
            mels: (B, T, M) unscaled mels.
            mel_lens (torch.Tensor | list[int]): Length of each mel in `mels`, as a tensor or a list.
            overlap (float): Fraction of a partial utterance shared with the next one.
            rate: If given, the step between partial utterances is derived from it and the sample rate
                instead of from `overlap`.
            min_coverage (float): Minimum fraction of one more partial utterance that the leftover frames must cover
                for it to be kept.
            batch_size (int | None): Number of partial utterances embedded per forward pass, or None for all at once.

        Returns:
            embeds (torch.Tensor): (B, E) embeddings on CPU.
        """
        mel_lens = mel_lens.tolist() if torch.is_tensor(mel_lens) else mel_lens

        # Compute where to split the utterances into partials
        frame_step = get_frame_step(overlap, rate, self.hp)
        n_partials, target_lens = zip(
            *(get_num_wins(mel_len, frame_step, min_coverage, self.hp) for mel_len in mel_lens)
        )

        # Possibly pad the mels to reach the target lengths
        len_diff = max(target_lens) - mels.size(1)
        if len_diff > 0:
            pad = torch.full((mels.size(0), len_diff, self.hp.num_mels), 0, dtype=torch.float32)
            mels = torch.cat((mels, pad.to(mels.device)), dim=1)

        # Group all partials together so that we can batch them easily
        partials = [
            mel[i * frame_step : i * frame_step + self.hp.ve_partial_frames]
            for mel, n_partial in zip(mels, n_partials)
            for i in range(n_partial)
        ]
        assert all(partials[0].shape == partial.shape for partial in partials)
        partials = torch.stack(partials)

        # Forward the partials
        n_chunks = int(np.ceil(len(partials) / (batch_size or len(partials))))
        partial_embeds = torch.cat([self(batch) for batch in partials.chunk(n_chunks)], dim=0).cpu()

        # Reduce the partial embeds into full embeds and L2-normalize them
        slices = np.concatenate(([0], np.cumsum(n_partials)))
        raw_embeds = [torch.mean(partial_embeds[start:end], dim=0) for start, end in zip(slices[:-1], slices[1:])]
        raw_embeds = torch.stack(raw_embeds)
        embeds = raw_embeds / torch.linalg.norm(raw_embeds, dim=1, keepdim=True)

        return embeds
