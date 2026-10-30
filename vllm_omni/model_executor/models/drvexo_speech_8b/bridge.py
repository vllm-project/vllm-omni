# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Trainable language-to-acoustic cross attention for DRVEXO."""

import torch
from torch import nn
from torch.nn import functional as F


class DRVEXOSpeechBridge(nn.Module):
    def __init__(self, language_dim=4096, acoustic_dim=2048, width=128, heads=4):
        super().__init__()
        if width < 1 or width % heads:
            raise ValueError("Bridge width must be divisible by its head count")
        self.width, self.heads = width, heads
        self.language_norm = nn.LayerNorm(language_dim)
        self.acoustic_norm = nn.LayerNorm(acoustic_dim)
        self.q = nn.Linear(acoustic_dim, width, bias=False)
        self.k = nn.Linear(language_dim, width, bias=False)
        self.v = nn.Linear(language_dim, width, bias=False)
        self.output = nn.Linear(width, acoustic_dim, bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(self, acoustic, language, valid):
        if acoustic.ndim != 3 or language.ndim != 3 or acoustic.shape[0] != language.shape[0]:
            raise ValueError("Bridge inputs must be rank-3 with equal batch sizes")
        if valid.dtype != torch.bool or valid.shape != language.shape[:2]:
            raise ValueError("Bridge language mask must be boolean and match memory")
        if not valid.any(dim=-1).all():
            raise ValueError("Bridge memory cannot be empty")
        dtype = self.output.weight.dtype
        q = self.q(self.acoustic_norm(acoustic.to(dtype)))
        k = self.k(self.language_norm(language.to(dtype)))
        v = self.v(self.language_norm(language.to(dtype)))

        def split(x):
            return x.reshape(x.shape[0], x.shape[1], self.heads, -1).transpose(1, 2)

        attended = (
            F.scaled_dot_product_attention(
                split(q),
                split(k),
                split(v),
                attn_mask=valid[:, None, None, :],
                dropout_p=0.0,
            )
            .transpose(1, 2)
            .reshape(q.shape[0], q.shape[1], self.width)
        )
        return acoustic + self.output(attended).to(acoustic.dtype)


def condition_spans(bridge, hidden, memories, spans):
    """Condition each packed request only on its own language memory."""
    if len(memories) != len(spans):
        raise ValueError("Bridge request count does not match token spans")
    output = hidden.clone()
    previous_end = 0
    for memory, (start, end) in zip(memories, spans):
        if not 0 <= previous_end <= start <= end <= len(hidden):
            raise ValueError("Bridge spans overlap or exceed scheduled tokens")
        previous_end = end
        if memory is None or start == end:
            continue
        if memory.ndim != 2 or memory.shape[-1] != bridge.language_norm.normalized_shape[0]:
            raise ValueError("Bridge memory has invalid dimensions")
        memory = memory.to(device=hidden.device, dtype=bridge.output.weight.dtype)
        valid = torch.ones((1, len(memory)), dtype=torch.bool, device=hidden.device)
        output[start:end] = bridge(hidden[start:end][None], memory[None], valid)[0]
    return output
