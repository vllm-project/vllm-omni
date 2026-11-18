# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import pytest
import torch

from vllm_omni.model_executor.models.personaplex.configuration_personaplex import (
    PersonaPlexDepformerConfig,
)
from vllm_omni.model_executor.models.personaplex.personaplex_depformer import PersonaPlexDepformer

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _tiny_depformer() -> PersonaPlexDepformer:
    torch.manual_seed(0)
    config = PersonaPlexDepformerConfig(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        head_dim=8,
        num_key_value_heads=4,
        intermediate_size=48,
        dep_q=16,
        num_active_codebooks=8,
        card=64,
    )
    model = PersonaPlexDepformer(config, temporal_hidden_size=24, text_card=100)
    for param in model.parameters():
        torch.nn.init.normal_(param, std=0.2)
    return model.eval()


def test_num_steps_matches_prefix_of_full_run_with_teacher_forcing():
    model = _tiny_depformer()
    text = torch.tensor([3, 7, 11])
    hidden = torch.randn(3, 1, 24)
    tokens = torch.randint(0, 64, (3, 16))
    provided = torch.zeros(3, 16, dtype=torch.bool)
    provided[0, 1:8] = True
    provided[:, 8:] = True

    full, full_logits = model(text, hidden, tokens, provided, return_logits=True)
    head, head_logits = model(text, hidden, tokens, provided, return_logits=True, num_steps=8)

    assert full.shape == (3, 16)
    assert head.shape == (3, 8)
    assert torch.equal(head, full[:, :8])
    assert torch.equal(head_logits, full_logits[:, :8])


@pytest.mark.parametrize("num_steps", [0, 17])
def test_num_steps_out_of_range_is_rejected(num_steps):
    model = _tiny_depformer()
    with pytest.raises(ValueError, match="num_steps"):
        model(torch.tensor([1]), torch.randn(1, 1, 24), num_steps=num_steps)
