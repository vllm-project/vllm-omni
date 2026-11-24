# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Strict Ulysses all-to-all over FlashInfer's ``UlyssesCommunicator``.

Selected with ``ulysses_a2a_backend="flashinfer"`` (on top of ``ulysses_a2a_permute``). On NVLink nodes FlashInfer
runs one-sided kernels; outputs registered with ``allocate_output()`` take its direct route, where every rank writes
its blocks straight into the peers' outputs. Those outputs belong to the communicator and are reused by every call,
so a returned tensor stays valid until the next exchange of the same kind on the group.
"""

from __future__ import annotations

import torch
from torch.distributed.distributed_c10d import _resolve_process_group

_COMMUNICATORS: dict[str, tuple[int, torch.dtype, object]] = {}
_OUTPUTS: dict[tuple, torch.Tensor] = {}


def ensure_flashinfer_ulysses_available() -> None:
    if not torch.cuda.is_available():
        raise ValueError("ulysses_a2a_backend='flashinfer' requires CUDA")
    try:
        from flashinfer.comm import UlyssesCommunicator  # noqa: F401
    except ImportError as exc:
        raise ValueError("ulysses_a2a_backend='flashinfer' requires flashinfer with UlyssesCommunicator") from exc


def _communicator(group_name: str, nbytes: int, dtype: torch.dtype):
    from flashinfer.comm import UlyssesCommunicator

    cached = _COMMUNICATORS.get(group_name)
    if cached is not None and cached[0] >= nbytes and cached[1] == dtype:
        return cached[2]
    if cached is not None:
        for key in [k for k in _OUTPUTS if k[0] == group_name]:
            del _OUTPUTS[key]
        cached[2].close()
    comm = UlyssesCommunicator(
        _resolve_process_group(group_name),
        max_bytes=nbytes,
        dtype=dtype,
        backend="auto",
        device=torch.accelerator.current_device_index(),
    )
    _COMMUNICATORS[group_name] = (nbytes, dtype, comm)
    return comm


def _output(comm, group_name: str, x: torch.Tensor, op: str, slot: int) -> torch.Tensor:
    key = (group_name, op, slot, tuple(x.shape), x.dtype)
    out = _OUTPUTS.get(key)
    if out is None:
        out = _OUTPUTS[key] = comm.allocate_output(x, op)
    return out


@torch.library.custom_op("vllm_omni_a2a::ulysses_fi_qkv", mutates_args=(), device_types="cuda")
def ulysses_fi_qkv(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, group_name: str, world_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(B, S/P, H, D) Q/K/V -> (B, S, H/P, D) each."""
    tensors = [t.contiguous() for t in (query, key, value)]
    comm = _communicator(group_name, max(t.numel() * t.element_size() for t in tensors), query.dtype)
    q, k, v = (
        comm.scatter_heads(t, out=_output(comm, group_name, t, "scatter_heads", slot)) for slot, t in enumerate(tensors)
    )
    return q, k, v


@ulysses_fi_qkv.register_fake
def _(query, key, value, group_name, world_size):
    def out(t):
        b, s, h, d = t.shape
        return t.new_empty(b, s * world_size, h // world_size, d)

    return out(query), out(key), out(value)


@torch.library.custom_op("vllm_omni_a2a::ulysses_fi_o", mutates_args=(), device_types="cuda")
def ulysses_fi_o(y: torch.Tensor, group_name: str, world_size: int) -> torch.Tensor:
    """(B, S, H/P, D) -> (B, S/P, H, D)."""
    y = y.contiguous()
    comm = _communicator(group_name, y.numel() * y.element_size(), y.dtype)
    return comm.gather_heads(y, out=_output(comm, group_name, y, "gather_heads", 0))


@ulysses_fi_o.register_fake
def _(y, group_name, world_size):
    b, s, h, d = y.shape
    return y.new_empty(b, s // world_size, h * world_size, d)
