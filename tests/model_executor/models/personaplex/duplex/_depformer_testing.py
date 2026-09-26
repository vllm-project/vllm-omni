# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import torch
from torch import nn

from vllm_omni.model_executor.models.personaplex.configuration_personaplex import (
    PersonaPlexDepformerConfig,
)
from vllm_omni.model_executor.models.personaplex.personaplex_depformer import (
    PersonaPlexDepformer,
)

HIDDEN = 32
TEMPORAL = 32
DEP_Q = 4
CARD = 16
TEXT_CARD = 32


def tiny_depformer_config() -> PersonaPlexDepformerConfig:
    return PersonaPlexDepformerConfig(
        hidden_size=HIDDEN,
        num_hidden_layers=2,
        num_attention_heads=4,
        head_dim=8,
        num_key_value_heads=4,
        intermediate_size=64,
        dep_q=DEP_Q,
        card=CARD,
        rms_norm_eps=1e-8,
    )


def init_depformer_weights(module: nn.Module, seed: int = 0) -> None:
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for param in module.parameters():
            param.copy_(torch.randn(param.shape, generator=g, dtype=param.dtype) * 0.02)


def make_depformer(
    device: torch.device | None = None,
    *,
    max_graph_batch_size: int = 8,
    seed: int = 0,
) -> PersonaPlexDepformer:
    model = PersonaPlexDepformer(
        tiny_depformer_config(),
        temporal_hidden_size=TEMPORAL,
        text_card=TEXT_CARD,
        max_graph_batch_size=max_graph_batch_size,
    )
    if device is not None:
        model.to(device)
    init_depformer_weights(model, seed=seed)
    return model


def clone_depformer(
    src: PersonaPlexDepformer,
    device: torch.device | None = None,
) -> PersonaPlexDepformer:
    dst = make_depformer(
        device,
        max_graph_batch_size=src.max_graph_batch_size,
        seed=1,
    )
    dst.load_state_dict(src.state_dict())
    dst.eval()
    return dst


def frame(
    batch: int,
    seed: int,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    text = torch.randint(0, TEXT_CARD, (batch,), generator=g)
    hidden = torch.randn(batch, 1, TEMPORAL, generator=g)
    tokens = torch.randint(0, CARD, (batch, DEP_Q), generator=g)
    provided = torch.zeros(batch, DEP_Q, dtype=torch.bool)
    provided[:, 0] = True
    if device is not None:
        text = text.to(device)
        hidden = hidden.to(device)
        tokens = tokens.to(device)
        provided = provided.to(device)
    return text, hidden, tokens, provided
