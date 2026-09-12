# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_omni.model_executor.models.breeze_tts_2.modeling_breeze_tts_2_talker import (
    BreezeTTS2TalkerForGeneration,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _PaddedHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(2_112, 32), requires_grad=False)


class _DepthDecoder:
    def __init__(self) -> None:
        self.loaded: list[tuple[str, torch.Tensor]] = []

    def load_weights(self, weights):
        self.loaded.extend(weights)
        return {"model.embed_tokens.weight"}


class _TalkerHost(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(tie_codebooks_embeddings=True)
        self.codebook_vocab_size = 2_051
        self.hidden_size = 32
        self.lm_head = _PaddedHead()
        self.embed_audio_tokens = nn.Embedding(4 * 2_051, 32)
        self.depth_decoder = _DepthDecoder()


def _padded_loader(parameter: nn.Parameter, weight: torch.Tensor) -> None:
    assert parameter.shape != weight.shape
    parameter[: weight.shape[0]].copy_(weight)


def test_lm_head_loader_accepts_vocab_padding():
    host = _TalkerHost()
    host.lm_head.weight.weight_loader = _padded_loader
    checkpoint_weight = torch.full((2_052, 32), 1.5)

    loaded = BreezeTTS2TalkerForGeneration.load_weights(
        host,
        iter([("lm_head.weight", checkpoint_weight)]),
    )

    assert "lm_head.weight" in loaded
    assert torch.equal(host.lm_head.weight[:2_052], checkpoint_weight)
    assert torch.count_nonzero(host.lm_head.weight[2_052:]).item() == 0


def test_tied_audio_embedding_reports_both_strict_loading_aliases():
    host = _TalkerHost()
    checkpoint_weight = torch.full((4 * 2_051, 32), 2.0)

    loaded = BreezeTTS2TalkerForGeneration.load_weights(
        host,
        iter([("depth_decoder.model.embed_tokens.weight", checkpoint_weight)]),
    )

    assert "embed_audio_tokens.weight" in loaded
    assert "depth_decoder.model.embed_tokens.weight" in loaded
