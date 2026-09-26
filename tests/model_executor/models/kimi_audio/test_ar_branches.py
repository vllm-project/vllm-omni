# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Dual-stream routing with layers that mutate hidden and residual buffers."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_omni.model_executor.models.kimi_audio import kimi_audio_ar_stage


class InplaceLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(4, 4)

    def forward(self, positions, hidden_states, residual):
        if residual is None:
            residual = hidden_states.clone()
        else:
            residual.add_(hidden_states)
        hidden_states.copy_(self.projection(residual))
        return hidden_states, residual


class InplaceNorm(nn.Module):
    def forward(self, hidden_states, residual):
        residual.add_(hidden_states)
        hidden_states.copy_(residual)
        return hidden_states, residual


@pytest.mark.parametrize("branch", ["shared", "text", "audio"])
@pytest.mark.parametrize("num_tokens", [1, 3])
@torch.inference_mode()
def test_forward_branch_isolation(monkeypatch, branch, num_tokens):
    monkeypatch.setattr(
        kimi_audio_ar_stage, "get_pp_group", lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True)
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        stage = kimi_audio_ar_stage.KimiAudioARStage.__new__(kimi_audio_ar_stage.KimiAudioARStage)
        nn.Module.__init__(stage)
        stage.config = SimpleNamespace(hidden_size=4)
        stage.start_layer, stage.end_layer, stage.branch_layer = 0, 3, 0
        stage.mimo_start_layer, stage.mimo_end_layer = 0, 1
        stage.layers = nn.ModuleList(InplaceLayer() for _ in range(3))
        stage.mimo_layers = nn.ModuleList([InplaceLayer()])
        stage.norm, stage.mimo_norm = InplaceNorm(), InplaceNorm()
        inputs = torch.randn(num_tokens, 4)

    positions = torch.arange(num_tokens)
    before = stage(None, positions, inputs_embeds=inputs.clone())
    layer = {"shared": stage.layers[0], "text": stage.layers[1], "audio": stage.mimo_layers[0]}[branch]
    layer.projection.bias.add_(1)
    after = stage(None, positions, inputs_embeds=inputs.clone())

    assert before.shape == after.shape == (num_tokens, 8)
    for index, stream in enumerate(("text", "audio")):
        expected_change = branch in ("shared", stream)
        start = index * 4
        assert torch.equal(before[:, start : start + 4], after[:, start : start + 4]) != expected_change
