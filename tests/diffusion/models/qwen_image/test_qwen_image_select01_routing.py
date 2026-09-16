# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
import torch.nn as nn

from vllm_omni.diffusion.models.qwen_image import qwen_image_transformer
from vllm_omni.diffusion.models.qwen_image.qwen_image_transformer import QwenImageTransformerBlock

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


class _StaticOutput(nn.Module):
    def __init__(self, output: torch.Tensor):
        super().__init__()
        self.output = output

    def forward(self, _: torch.Tensor) -> torch.Tensor:
        return self.output


class _RecordingNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def forward(self, x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        self.calls.append((x, scale, shift))
        return x


class _ZeroAttention(nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        **_: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.zeros_like(hidden_states), torch.zeros_like(encoder_hidden_states)


class _ZeroMLP(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)


def test_non_cuda_select01_uses_adalayernorm_dispatch(monkeypatch):
    hidden_size = 4
    block = QwenImageTransformerBlock.__new__(QwenImageTransformerBlock)
    nn.Module.__init__(block)
    block.zero_cond_t = True
    block.img_mod = _StaticOutput(torch.zeros(2, 6 * hidden_size))
    block.txt_mod = _StaticOutput(torch.zeros(1, 6 * hidden_size))
    block.img_norm1 = _RecordingNorm()
    block.img_norm2 = _RecordingNorm()
    block.txt_norm1 = _RecordingNorm()
    block.txt_norm2 = _RecordingNorm()
    block.attn = _ZeroAttention()
    block.img_mlp = _ZeroMLP()
    block.txt_mlp = _ZeroMLP()

    monkeypatch.setattr(qwen_image_transformer, "can_use_qwen_select01_triton", lambda _: False)

    def fail_if_called(*args, **kwargs):
        pytest.fail("non-CUDA select01 must preserve the AdaLayerNorm dispatch")

    monkeypatch.setattr(qwen_image_transformer, "fused_layernorm_select01", fail_if_called)
    monkeypatch.setattr(qwen_image_transformer, "fused_residual_layernorm_select01", fail_if_called)

    hidden_states = torch.randn(1, 3, hidden_size)
    encoder_hidden_states = torch.randn(1, 2, hidden_size)
    modulate_index = torch.tensor([[0, 1, 0]])
    block(
        hidden_states,
        encoder_hidden_states,
        torch.ones(1, 2),
        torch.zeros(2, hidden_size),
        (torch.empty(0), torch.empty(0)),
        modulate_index=modulate_index,
    )

    assert len(block.img_norm1.calls) == 1
    assert len(block.img_norm2.calls) == 1
    for _, scale, shift in (*block.img_norm1.calls, *block.img_norm2.calls):
        assert scale.shape == hidden_states.shape
        assert shift.shape == hidden_states.shape
