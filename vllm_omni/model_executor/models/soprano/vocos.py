# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Adapted from ekwek1/soprano, commit 12fac06eb8fa53bad8b3941d3cb11e9c869477c4,
# soprano/vocos/{decoder,models,modules,heads,spectral_ops}.py, Apache-2.0.
# ConvNeXt block originally adapted upstream from facebookresearch/ConvNeXt.
"""Only the checkpoint's kernel-3 Vocos backbone and center ISTFT head."""

import torch
from torch import nn
from torch.nn import functional as F


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim: int = 768, intermediate_dim: int = 2304):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, 3, padding=1, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
        self.gamma = nn.Parameter(torch.ones(dim) / 8**0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(self.dwconv(x).transpose(1, 2))
        x = self.pwconv2(self.act(self.pwconv1(x))) * self.gamma
        return residual + x.transpose(1, 2)


class VocosBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Conv1d(512, 768, 1)
        self.norm = nn.LayerNorm(768, eps=1e-6)
        self.convnext = nn.ModuleList([ConvNeXtBlock() for _ in range(8)])
        self.final_layer_norm = nn.LayerNorm(768, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(self.embed(x).transpose(1, 2)).transpose(1, 2)
        for block in self.convnext:
            x = block(x)
        return self.final_layer_norm(x.transpose(1, 2)).transpose(1, 2)


class CenterISTFT(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("window", torch.hann_window(2048))

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        # Preserve the reference's explicit removal of DC and Nyquist bins.
        spec[:, 0] = 0
        spec[:, -1] = 0
        return torch.istft(spec, 2048, 512, 2048, self.window, center=True)


class ISTFTHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.out = nn.Linear(768, 2050)
        self.istft = CenterISTFT()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mag, phase = self.out(x.transpose(1, 2)).transpose(1, 2).chunk(2, dim=1)
        mag = torch.clip(torch.exp(mag), max=1e2)
        return self.istft(mag * (torch.cos(phase) + 1j * torch.sin(phase)))


class SopranoDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = VocosBackbone()
        self.head = ISTFTHead()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.float32 or x.ndim != 3 or x.shape[1] != 512 or x.shape[2] < 2:
            raise ValueError("Decoder requires float32 [B,512,T] with T >= 2")
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = F.interpolate(x, size=4 * (x.shape[2] - 1) + 1, mode="linear", align_corners=True)
            return self.head(self.decoder(x))
