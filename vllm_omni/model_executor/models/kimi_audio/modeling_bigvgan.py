# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-Audio BigVGAN-v2 vocoder. Mel (B, n_mels, T) -> wav (B, 1, T*hop).

Reuses the alias-free SnakeBeta primitives that ship with Qwen2.5-Omni's
token2wav rather than vendoring a second copy. Weights are loaded from a
weight_norm-folded checkpoint hosted on HF; the original NVIDIA-format
checkpoint at ``<model_path>/vocoder/model.pt`` is no longer used.
"""

import json
import logging

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torch import Tensor, nn

from vllm_omni.model_executor.models.qwen2_5_omni.qwen2_5_omni_token2wav import (
    AMPBlock,
    SnakeBeta,
    TorchActivation1d,
)

logger = logging.getLogger(__name__)

# weight_norm-folded checkpoint: weight_g/weight_v collapsed into weight at
# rest, math-preserving (audio matches original to float32 epsilon).
DEFAULT_HF_REPO = "zhangj1an/kimi-audio-bigvgan-hf"


class KimiBigVGAN(nn.Module):
    """conv_pre → 7 upsample stages × (ConvTranspose1d + 4 AMPBlock)
    → SnakeBeta → conv_post (bias=False) → clamp(-1, 1)."""

    def __init__(
        self,
        mel_dim: int,
        upsample_initial_channel: int,
        upsample_rates: list[int],
        upsample_kernel_sizes: list[int],
        resblock_kernel_sizes: list[int],
        resblock_dilation_sizes: list[list[int]],
    ):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)

        self.conv_pre = nn.Conv1d(mel_dim, upsample_initial_channel, 7, 1, padding=3)

        # Wrapped in an inner ModuleList so the checkpoint keys
        # ``ups.{i}.0.weight`` match Qwen2_5OmniToken2WavBigVGANModel.
        self.ups = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        nn.ConvTranspose1d(
                            upsample_initial_channel // (2**i),
                            upsample_initial_channel // (2 ** (i + 1)),
                            k,
                            u,
                            padding=(k - u) // 2,
                        )
                    ]
                )
                for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes))
            ]
        )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(AMPBlock(ch, k, tuple(d)))

        self.activation_post = TorchActivation1d(activation=SnakeBeta(ch))
        self.conv_post = nn.Conv1d(ch, 1, 7, 1, padding=3, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            for up in self.ups[i]:
                x = up(x)
            xs = sum(self.resblocks[i * self.num_kernels + j](x) for j in range(self.num_kernels))
            x = xs / self.num_kernels
        x = self.conv_post(self.activation_post(x))
        return torch.clamp(x, min=-1.0, max=1.0)

    def decode_mel(self, mel: Tensor) -> Tensor:
        """[T, num_mels] mel → [1, T] wav."""
        target_device = next(self.parameters()).device
        mel = mel.transpose(0, 1).unsqueeze(0).to(target_device)
        return self(mel).squeeze(0)

    @classmethod
    def load_from_hf(cls, device, repo_id: str = DEFAULT_HF_REPO) -> "KimiBigVGAN":
        config_path = hf_hub_download(repo_id, "config.json")
        weights_path = hf_hub_download(repo_id, "model.safetensors")
        with open(config_path) as f:
            cfg = json.load(f)
        model = cls(
            mel_dim=cfg["mel_dim"],
            upsample_initial_channel=cfg["upsample_initial_channel"],
            upsample_rates=cfg["upsample_rates"],
            upsample_kernel_sizes=cfg["upsample_kernel_sizes"],
            resblock_kernel_sizes=cfg["resblock_kernel_sizes"],
            resblock_dilation_sizes=cfg["resblock_dilation_sizes"],
        )
        state = load_file(weights_path)
        # Filter buffers (kaiser sinc) are non-persistent in Qwen's
        # UpSample1d/DownSample1d, so they're absent from the safetensors
        # by design. They're re-derived from kernel_size on init.
        model.load_state_dict(state, strict=False)
        logger.info("Loaded Kimi BigVGAN from HF repo %s", repo_id)
        return model.to(device).eval()
