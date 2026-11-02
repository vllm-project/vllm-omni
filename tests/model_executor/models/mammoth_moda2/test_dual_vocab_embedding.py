"""Regression test for the shape-static dual-vocabulary embedding lookup.

The mixed text/generation-vocab path must produce the same values as
per-table lookups, stay in-bounds by construction, and involve no
data-dependent host branches (uniform batches take the same code path).
"""

import pytest
import torch
import torch.nn as nn

from vllm_omni.model_executor.models.mammoth_moda2.mammoth_moda2 import (
    MammothModa2Qwen2ForCausalLM,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

HIDDEN = 16
GEN_START = 32
GEN_SIZE = 8


class _FakeModel:
    """Carries just the attributes get_input_embeddings needs."""

    extra_gen_vocab = True
    gen_vocab_start_index = GEN_START

    def __init__(self):
        torch.manual_seed(0)
        self.embed_tokens = nn.Embedding(GEN_START, HIDDEN)
        self.gen_embed_tokens = nn.Embedding(GEN_SIZE, HIDDEN)

    def get_input_embeddings(self, input_ids):
        return MammothModa2Qwen2ForCausalLM.get_input_embeddings(self, input_ids)


def _reference(model, input_ids):
    out = torch.empty(input_ids.shape + (HIDDEN,))
    flat = input_ids.reshape(-1)
    ref = out.reshape(-1, HIDDEN)
    for i, tid in enumerate(flat.tolist()):
        if tid >= GEN_START:
            ref[i] = model.gen_embed_tokens.weight[tid - GEN_START]
        else:
            ref[i] = model.embed_tokens.weight[tid]
    return out


@pytest.mark.parametrize(
    "ids",
    [
        # Mixed batch with both vocab boundaries exercised.
        [0, GEN_START - 1, GEN_START, GEN_START + GEN_SIZE - 1, 5, GEN_START + 3],
        # Uniform text-only batch.
        [1, 2, 3, GEN_START - 1],
        # Uniform generation-vocab batch.
        [GEN_START, GEN_START + 1, GEN_START + GEN_SIZE - 1],
        # Single token, both vocabs.
        [0],
        [GEN_START],
    ],
)
def test_dual_vocab_embedding_matches_reference(ids):
    model = _FakeModel()
    input_ids = torch.tensor(ids, dtype=torch.long)
    out = model.get_input_embeddings(input_ids)
    assert out.shape == (len(ids), HIDDEN)
    torch.testing.assert_close(out, _reference(model, input_ids))


def test_dual_vocab_embedding_2d_input():
    model = _FakeModel()
    input_ids = torch.tensor([[0, GEN_START], [GEN_START + 2, 7]], dtype=torch.long)
    out = model.get_input_embeddings(input_ids)
    assert out.shape == (2, 2, HIDDEN)
    torch.testing.assert_close(out, _reference(model, input_ids))
