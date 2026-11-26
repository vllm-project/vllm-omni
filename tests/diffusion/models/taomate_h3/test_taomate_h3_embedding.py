# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Check that prepared TaoMate row selections reuse MiniMax embedding math."""

import pytest
import torch
import torch.nn as nn

from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import MiniMaxH3DiTModel
from vllm_omni.diffusion.models.taomate_h3.transformer import LocalEmbedPlan, TaoMateH3DiTModel, local_embed_plan

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


class _Projection(nn.Module):
    def forward(self, x):
        return x * 0.75 + 0.125, None


class _Refiner(nn.Module):
    def forward(self, x, **kwargs):
        return x + 0.0625


def _model(cls):
    model = object.__new__(cls)
    nn.Module.__init__(model)
    model.hidden_size = 2
    model.video_patch_proj = _Projection()
    model.audio_patch_proj = _Projection()
    model.condition_proj = _Projection()
    model.token_refiner = _Refiner()
    model.time_embedder = nn.Identity()
    return model


def _inputs(local_span):
    return {
        "x": torch.arange(32, dtype=torch.float32).reshape(1, 16, 2) / 7,
        "audio_x": torch.arange(32, dtype=torch.float32).reshape(1, 16, 2) / 11,
        "text_embeddings_selected": torch.arange(8, dtype=torch.bfloat16).reshape(4, 2) / 3,
        "unique_timesteps": torch.tensor([0.2, 0.7]),
        "img_pos": torch.tensor([1, 2, 6, 7, 11, 12]),
        "audio_pos": torch.tensor([3, 4, 8, 9, 13, 14]),
        "text_pos": torch.tensor([0, 5, 10, 15]),
        "refiner_cu_seqlens": torch.tensor([0, 4, 4], dtype=torch.int32),
        "refiner_max_seqlen": 4,
        "seq_len": 16,
        "device": torch.device("cpu"),
        "local_span": local_span,
    }


def _plan(inputs):
    return LocalEmbedPlan.build(
        **{name: inputs[name] for name in ("img_pos", "audio_pos", "text_pos", "seq_len", "local_span", "device")}
    )


@pytest.mark.parametrize(
    ("local_span", "plan_span"),
    [(span, span) for span in [(0, 16), (0, 8), (8, 8), *((index * 2, 2) for index in range(8))]]
    + [((8, 8), (0, 8))],  # A plan for another span must fall back to the current rank's selection.
)
def test_prepared_and_unprepared_embeddings_match_minimax(local_span, plan_span):
    inputs = _inputs(local_span)
    expected = _model(MiniMaxH3DiTModel)._embed(**inputs)
    model = _model(TaoMateH3DiTModel)
    actual = model._embed(**inputs)
    with local_embed_plan(_plan(_inputs(plan_span))):
        prepared = model._embed(**inputs)
    for reference, eager, graphed in zip(expected, actual, prepared, strict=True):
        assert torch.equal(reference, eager)
        assert torch.equal(reference, graphed)


def test_prepared_embedding_uses_static_indices(monkeypatch):
    inputs = _inputs((0, 8))
    expected = _model(MiniMaxH3DiTModel)._embed(**inputs)

    def unexpected_selection(*args, **kwargs):
        raise AssertionError("a prepared embedding must not recompute boolean row selections")

    monkeypatch.setattr(MiniMaxH3DiTModel, "_embed_row_indices", unexpected_selection)
    with local_embed_plan(_plan(inputs)):
        actual = _model(TaoMateH3DiTModel)._embed(**inputs)
    for reference, value in zip(expected, actual, strict=True):
        assert torch.equal(reference, value)


def test_embedding_plan_rejects_different_row_counts():
    inputs = _inputs((0, 8))
    plan = _plan(inputs)
    inputs["img_pos"] = inputs["img_pos"][:-1]
    with local_embed_plan(plan), pytest.raises(RuntimeError, match="does not describe this document"):
        _model(TaoMateH3DiTModel)._embed(**inputs)
