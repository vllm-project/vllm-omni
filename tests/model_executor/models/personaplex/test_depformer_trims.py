# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import pytest
import torch

from vllm_omni.model_executor.models.personaplex.configuration_personaplex import (
    PersonaPlexDepformerConfig,
)
from vllm_omni.model_executor.models.personaplex.personaplex_depformer import (
    PersonaPlexDepformer,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

HIDDEN = 24


def _config(dep_q: int) -> PersonaPlexDepformerConfig:
    return PersonaPlexDepformerConfig(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        head_dim=8,
        num_key_value_heads=4,
        intermediate_size=48,
        dep_q=dep_q,
        num_active_codebooks=8,
        card=64,
    )


def _checkpoint(sets: int) -> dict[str, torch.Tensor]:
    """A raw checkpoint with ``sets`` depformer weight sets.

    ``nvidia/personaplex-7b-v1`` ships all 16 (15 ``depformer_emb`` tables).
    """
    torch.manual_seed(0)
    source = PersonaPlexDepformer(_config(sets), temporal_hidden_size=HIDDEN, text_card=100)
    state: dict[str, torch.Tensor] = {}
    for step in range(sets):
        state[f"depformer_in.{step}.weight"] = torch.randn_like(source.depformer_in[step].weight) * 0.3
        state[f"linears.{step}.weight"] = torch.randn_like(source.linears[step].weight) * 0.3
    for step in range(sets - 1):
        state[f"depformer_emb.{step}.weight"] = torch.randn_like(source.depformer_emb[step].weight) * 0.3
    state["depformer_text_emb.weight"] = torch.randn_like(source.depformer_text_emb.weight) * 0.3
    for index, layer in enumerate(source.layers):
        prefix = f"depformer.layers.{index}"
        state[f"{prefix}.self_attn.in_proj_weight"] = torch.randn_like(layer.in_proj_weight) * 0.3
        state[f"{prefix}.self_attn.out_proj.weight"] = torch.randn_like(layer.out_proj_weight) * 0.3
        state[f"{prefix}.norm1.alpha"] = torch.rand(1, 1, 32) + 0.5
        state[f"{prefix}.norm2.alpha"] = torch.rand(1, 1, 32) + 0.5
        for step in range(sets):
            state[f"{prefix}.gating.{step}.linear_in.weight"] = torch.randn_like(layer.gating_in[step]) * 0.3
            state[f"{prefix}.gating.{step}.linear_out.weight"] = torch.randn_like(layer.gating_out[step]) * 0.3
    return state


def _loaded(dep_q: int, state: dict[str, torch.Tensor]) -> PersonaPlexDepformer:
    model = PersonaPlexDepformer(_config(dep_q), temporal_hidden_size=HIDDEN, text_card=100)
    loaded = model.load_weights(state)
    assert loaded == {name for name, _ in model.named_parameters()}
    return model.eval()


def _inputs() -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(3)
    text = torch.randint(0, 100, (5,), generator=generator)
    hidden = torch.randn(5, 1, HIDDEN, generator=generator)
    tokens = torch.randint(0, 64, (5, 16), generator=generator)
    provided = torch.rand(5, 16, generator=generator) > 0.5
    return text, hidden, tokens, provided


def test_an_eight_step_depformer_matches_the_agent_steps_of_the_full_one() -> None:
    state = _checkpoint(16)
    full, agent = _loaded(16, state), _loaded(8, state)
    text, hidden, tokens, provided = _inputs()

    full_codes, full_logits = full(text, hidden, tokens, provided, return_logits=True, num_steps=8)
    codes, logits = agent(text, hidden, tokens, provided, return_logits=True, num_steps=8)

    assert torch.equal(codes, full_codes)
    assert torch.equal(logits, full_logits)
