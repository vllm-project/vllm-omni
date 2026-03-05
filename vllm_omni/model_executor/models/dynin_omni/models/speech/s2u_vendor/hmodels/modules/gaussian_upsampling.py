import torch
from torch import nn
from torch.nn import functional as F
import numpy as np


class GaussianUpsampling(nn.Module):
    """
        Non-attention Tacotron:
            - https://arxiv.org/abs/2010.04301
        this source code is implemenation of the ExpressiveTacotron from BridgetteSong
            - https://github.com/BridgetteSong/ExpressiveTacotron/blob/master/model_duration.py
    """
    def __init__(self, variance=1.0):
        super().__init__()
        self.mask_score = -1e15
        self.var = torch.tensor(variance)
        self.HH_skip_init = True

    def forward(self, inputs, inputs_len, durations, output_max_len):
        """ Gaussian upsampling
        ------
        inputs: [B, N, H]
        inputs_len : [B]
        durations: phoneme durations  [B, N]
        vars : phoneme attended ranges [B, N]
        RETURNS
        -------
        upsampling_outputs: upsampled output  [B, T, H]
        """
        # output_len_max = int(torch.sum(durations, dim=1).max().item())
        w_t = get_upsampling_weights(durations, output_max_len, self.var, inputs_len)

        upsampling_outputs = torch.bmm(w_t.transpose(1, 2), inputs)  # [B, T, encoder_hidden_size]

        return upsampling_outputs


def get_upsampling_weights(durations, output_max_len, variance, input_lens, mask_score=-1e15):
    B, N = durations.shape
    c = torch.cumsum(durations, dim=1).float() - 0.5 * durations
    c = c.unsqueeze(2)  # [B, N, 1]
    t = torch.arange(output_max_len, device=durations.device).expand(B, N, output_max_len).float()  # [B, N, T]
    # Gaussian distribution density in log domain
    w_t = -0.5 * (np.log(2.0 * np.pi) + torch.log(variance) + torch.pow(t - c, 2) / variance)  # [B, N, T]
    if input_lens is not None:
        input_masks = ~get_mask_from_lengths(input_lens, N)  # [B, N]
        # input_masks = torch.tensor(input_masks, dtype=torch.bool, device=w_t.device)
        masks = input_masks.unsqueeze(2)
        w_t.data.masked_fill_(masks, mask_score)
    w_t = F.softmax(w_t, dim=1)
    return w_t


def get_mask_from_lengths(lengths, max_len=None):
    if max_len is None:
        max_len = max(lengths)
    ids = torch.arange(max_len, device=lengths.device)
    mask = (ids < lengths.reshape(-1, 1))
    return mask
