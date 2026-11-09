# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Owned CPU tensor snapshots for the untyped turn-mode request envelope."""

import torch

from vllm_omni.distributed.omni_connectors.utils.serialization import OmniSerializer


def pack_handoff_tensor(value: torch.Tensor) -> bytes:
    """Keep legacy list-to-float32 values without creating Python scalar lists.

    EngineCore's ``model_intermediate_buffer`` has Any-valued fields, so raw
    tensors would lose their type on its msgpack wire. The existing Omni codec
    packs an owned byte snapshot and restores the dtype/shape at the consumer.
    """
    tensor = value.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if tensor.numel() == 0:
        # Empty transposed tensors may report contiguous with a nonunit last
        # stride; the shared codec reinterprets that dimension as bytes.
        tensor = torch.empty(tensor.shape, dtype=torch.float32)
    return OmniSerializer.serialize(tensor)


def unpack_handoff_tensor(value: object) -> object:
    """Accept existing local tensors/lists as well as packed turn handoffs."""
    if not isinstance(value, bytes):
        return value
    tensor = OmniSerializer.deserialize(value)
    if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.float32 or tensor.device.type != "cpu":
        raise ValueError("MiniCPM-o packed handoff must contain a CPU float32 tensor")
    return tensor


def ends_with_own_row(payload: object, position: int, token_id: int) -> bool:
    """Whether a forward's rows end at ``token_id``'s own row at ``position``.

    The rows must be one per consecutive input position, so a row predicting
    the token, a duplicated row or a rejected draft's row does not qualify.
    """
    if not isinstance(payload, dict):
        return False
    ids, positions, latent = (payload.get(k) for k in ("latent_input_ids", "latent_positions", "latent"))
    if not all(isinstance(v, torch.Tensor) for v in (ids, positions, latent)):
        return False
    ids, positions = ids.reshape(-1).tolist(), positions.reshape(-1).tolist()
    n = len(ids)
    return (
        n > 0
        and len(positions) == latent.shape[0] == n
        and positions == list(range(position - n + 1, position + 1))
        and int(ids[-1]) == token_id
    )


def has_no_rows(payload: object) -> bool:
    """Whether a forward's ledger is present and empty: every row was already delivered."""
    if not isinstance(payload, dict):
        return False
    rows = [payload.get(k) for k in ("latent_input_ids", "latent_positions", "latent")]
    return all(isinstance(v, torch.Tensor) and v.shape[0] == 0 for v in rows)
