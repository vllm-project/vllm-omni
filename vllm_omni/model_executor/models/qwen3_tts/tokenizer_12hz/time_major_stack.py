# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Time-major upsample + decoder conv stack for the Qwen3-TTS 12 Hz codec.

Same math as ``Qwen3TTSTokenizerV2Decoder.upsample``/``.decoder`` on ``[B, T, C]``
activations: dense causal convs are implicit GEMMs (tensor cores, no im2col
buffer, residual adds in the epilogue), SnakeBeta one pass per activation,
transposed convs a GEMM plus an overlap-add. See ``time_major_conv.py``.
"""

from __future__ import annotations

import torch
from torch import nn

from vllm_omni.model_executor.models.qwen3_tts.tokenizer_12hz.time_major_conv import (
    causal_conv,
    conv_gemm_weight,
    snake_im2col,
    transposed_gemm_weight,
    transposed_overlap_add,
)


def _snake(module) -> tuple[torch.Tensor, torch.Tensor]:
    if module._exp_alpha is None:
        module.precompute_exp_cache()
    return module._exp_alpha, module._inv_beta


class _Conv:
    def __init__(self, causal_conv) -> None:
        conv = causal_conv.conv
        if conv.stride[0] != 1 or conv.groups != 1:
            raise ValueError("time-major decoder convs must be dense with stride 1")
        self.weight = conv_gemm_weight(conv.weight)
        self.bias = conv.bias.detach()
        self.kernel_size = conv.kernel_size[0]
        self.dilation = conv.dilation[0]

    def __call__(self, x: torch.Tensor, residual: torch.Tensor | None = None) -> torch.Tensor:
        return causal_conv(x, self.weight, self.bias, self.kernel_size, self.dilation, residual)


class _Transposed:
    def __init__(self, causal_trans_conv) -> None:
        conv = causal_trans_conv.conv
        self.weight = transposed_gemm_weight(conv.weight)
        self.bias = conv.bias.detach()
        self.kernel_size = conv.kernel_size[0]
        self.stride = conv.stride[0]
        self.out_channels = conv.out_channels

    def __call__(self, x2d: torch.Tensor, bsz: int, t: int) -> torch.Tensor:
        z = torch.mm(x2d, self.weight).view(bsz, t, self.kernel_size, self.out_channels)
        return transposed_overlap_add(z, self.stride, self.bias)


class TimeMajorConvStack:
    """Callable replacing the NCT conv loops; holds GEMM-layout copies of the weights."""

    def __init__(self, decoder: nn.Module) -> None:
        self.upsample = [(_Transposed(trans), block) for trans, block in decoder.upsample]
        self.conv_in = _Conv(decoder.decoder[0])
        self.blocks = []
        for block in decoder.decoder[1:-2]:
            snake, trans, *units = block.block
            residuals = [(_snake(u.act1), _Conv(u.conv1), _snake(u.act2), _Conv(u.conv2)) for u in units]
            self.blocks.append((_snake(snake), _Transposed(trans), residuals))
        self.snake_out = _snake(decoder.decoder[-2])
        self.conv_out = _Conv(decoder.decoder[-1])

    @staticmethod
    def _convnext(x: torch.Tensor, block) -> torch.Tensor:
        dw = block.dwconv.conv
        k, t = dw.kernel_size[0], x.shape[1]
        xp = torch.nn.functional.pad(x, (0, 0, k - 1, 0))
        weight = dw.weight[:, 0, :].float()
        h = dw.bias.float() + sum(xp[:, j : j + t].float() * weight[:, j] for j in range(k))
        h = block.pwconv2(block.act(block.pwconv1(block.norm(h.to(x.dtype)))))
        return x + block.gamma * h

    def __call__(self, hidden: torch.Tensor) -> torch.Tensor:
        """``hidden`` ``[B, T, latent]`` -> unclamped waveform ``[B, 1, T * upsample]``."""
        bsz = hidden.shape[0]

        def snake(x, params):
            return snake_im2col(x, 1, snake=params).view(x.shape)

        for trans, block in self.upsample:
            t = hidden.shape[1]
            hidden = self._convnext(trans(hidden.reshape(bsz * t, -1), bsz, t), block)
        hidden = self.conv_in(hidden)
        for block_snake, trans, residuals in self.blocks:
            t = hidden.shape[1]
            hidden = trans(snake_im2col(hidden, 1, snake=block_snake), bsz, t)
            for act1, conv1, act2, conv2 in residuals:
                hidden = conv2(snake(conv1(snake(hidden, act1)), act2), residual=hidden)
        wav = self.conv_out(snake(hidden, self.snake_out))
        return wav.view(bsz, 1, -1)
