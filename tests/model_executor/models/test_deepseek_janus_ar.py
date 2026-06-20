# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Iterable

import pytest
import torch
from torch import nn

from vllm_omni.model_executor.models.deepseek_janus.deepseek_janus_ar import JanusForImageGeneration

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_janus_ar_load_weights_materializes_single_pass_iterable(monkeypatch: pytest.MonkeyPatch) -> None:
    model = JanusForImageGeneration.__new__(JanusForImageGeneration)
    nn.Module.__init__(model)
    model.gen_head = torch.nn.Sequential(
        torch.nn.Linear(2, 3),
        torch.nn.GELU(),
        torch.nn.Linear(3, 4),
    )
    model.gen_embed = torch.nn.Embedding(5, 2)
    model.gen_aligner = torch.nn.Linear(2, 2)

    seen_language_weights: list[list[str]] = []

    def fake_super_load_weights(self, rows: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        captured = [name for name, _ in rows]
        seen_language_weights.append(captured)
        return set(captured)

    monkeypatch.setattr(
        "vllm.model_executor.models.llama.LlamaForCausalLM.load_weights",
        fake_super_load_weights,
    )

    weights = iter(
        [
            ("language_model.model.weight", torch.ones((2, 2))),
            ("gen_embed.weight", torch.full((5, 2), 7.0)),
        ]
    )

    loaded = JanusForImageGeneration.load_weights(model, weights)

    assert seen_language_weights == [["model.weight"]]
    assert "model.weight" in loaded
    assert "gen_embed.weight" in loaded
    assert torch.equal(model.gen_embed.weight, torch.full((5, 2), 7.0))
