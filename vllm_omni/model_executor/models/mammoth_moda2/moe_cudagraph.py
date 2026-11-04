# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Mask-keyed CUDA-graph specialty for MammothModa2 AR MoE (#9865).

#9764 records both und/gen experts under capture so host-side mask branches
cannot bake the wrong path. That dual-expert tax hits all-und / all-gen
batches that share the same FULL graph.

This module defines the routing key used to capture three specialized graphs
per batch size (similar spirit to ``cudagraph_specialize_lora``):

| Key | Capture records | Host selects when |
| --- | --- | --- |
| BLEND | both + ``torch.where`` | mixed (or unknown) |
| ALL_UND | ``und_expert`` only | mask all false |
| ALL_GEN | ``gen_expert`` only | mask all true |

Host ``.any()`` / ``.all()`` runs **outside** the captured region (graph
dispatch). Inside the graph, mixed still uses the #9764 blend path.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from enum import IntEnum
from typing import Iterator

import torch


class MoERoutingKey(IntEnum):
    """CUDA-graph specialty for MammothModa2 MoE layers."""

    BLEND = 0
    ALL_UND = 1
    ALL_GEN = 2


# Active specialty for the current capture / replay. Default BLEND preserves
# #9764 behaviour until the runner wires mask-keyed dispatch.
_moe_routing_key: ContextVar[MoERoutingKey] = ContextVar(
    "mammoth_moda2_moe_routing_key",
    default=MoERoutingKey.BLEND,
)


def get_moe_routing_key() -> MoERoutingKey:
    return _moe_routing_key.get()


def set_moe_routing_key(key: MoERoutingKey) -> None:
    _moe_routing_key.set(key)


@contextmanager
def moe_routing_key_scope(key: MoERoutingKey) -> Iterator[None]:
    """Temporarily force the MoE capture/replay specialty."""
    token = _moe_routing_key.set(key)
    try:
        yield
    finally:
        _moe_routing_key.reset(token)


def moe_routing_key_from_mask(gen_token_mask: torch.Tensor | None) -> MoERoutingKey:
    """Classify a batch mask for host-side graph dispatch (outside capture).

    Missing / empty masks fall back to BLEND so dispatch stays fail-closed
    toward the dual-expert graph rather than guessing ALL_UND.
    """
    if gen_token_mask is None or gen_token_mask.numel() == 0:
        return MoERoutingKey.BLEND
    flat = gen_token_mask.reshape(-1)
    # Host sync is intentional: this runs before graph replay selection.
    if not bool(flat.any().item()):
        return MoERoutingKey.ALL_UND
    if bool(flat.all().item()):
        return MoERoutingKey.ALL_GEN
    return MoERoutingKey.BLEND


def moe_routing_key_from_input_ids(
    input_ids: torch.Tensor | None,
    gen_vocab_start_index: int,
) -> MoERoutingKey:
    """Derive the dispatch key from raw token ids (runner path)."""
    if input_ids is None or input_ids.numel() == 0:
        return MoERoutingKey.BLEND
    return moe_routing_key_from_mask(input_ids >= gen_vocab_start_index)


def capture_mask_for_routing_key(
    num_tokens: int,
    key: MoERoutingKey,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Dummy ``gen_token_mask`` matching the specialty being captured."""
    if key is MoERoutingKey.ALL_GEN:
        return torch.ones(num_tokens, dtype=torch.bool, device=device)
    if key is MoERoutingKey.ALL_UND:
        return torch.zeros(num_tokens, dtype=torch.bool, device=device)
    # BLEND: mixed pattern so torch.where stays live in the recorded graph.
    mask = torch.zeros(num_tokens, dtype=torch.bool, device=device)
    if num_tokens > 0:
        mask[0] = True
    if num_tokens > 1:
        mask[1] = False
    if num_tokens > 2:
        mask[2:] = True
    return mask
