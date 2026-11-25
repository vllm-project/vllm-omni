# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Strict Ulysses all-to-all over FlashInfer's ``UlyssesCommunicator``.

Selected with ``ulysses_a2a_backend="flashinfer"`` (on top of ``ulysses_a2a_permute``). On NVLink nodes FlashInfer
runs one-sided kernels; outputs registered with ``allocate_output()`` take its direct route, where every rank writes
its blocks straight into the peers' outputs. Those outputs belong to the communicator and are reused by every call:
Q/K/V are returned as those outputs because attention consumes them inside the Ulysses strategy, while the attention
output, which goes back to model code, is copied out.
"""

from __future__ import annotations

import torch
from torch.distributed.distributed_c10d import _resolve_process_group

_COMMUNICATORS: dict[str, tuple[int, torch.dtype, object]] = {}
# One output per (group, op, slot). On the direct route FlashInfer registers each output at the communicator's full
# capacity until close(), so new request shapes reuse that allocation instead of registering another one.
_OUTPUTS: dict[tuple[str, str, int], torch.Tensor] = {}


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
        # Like a2a_permute's grow-only workspace: earlier kernels (the previous layer's attention reading Q/K/V from
        # the registered outputs) may still be running, so finish them before the old buffers are released.
        torch.accelerator.synchronize()
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


def _output_shape(x: torch.Tensor, op: str, world_size: int) -> tuple[int, int, int, int]:
    b, s, h, d = x.shape
    if op == "scatter_heads":
        return b, s * world_size, h // world_size, d
    return b, s // world_size, h * world_size, d


def _output(comm, group_name: str, x: torch.Tensor, op: str, slot: int) -> torch.Tensor:
    key = (group_name, op, slot)
    shape = _output_shape(x, op, comm.world_size)
    out = _OUTPUTS.get(key)
    if out is not None and out.dtype == x.dtype:
        if tuple(out.shape) == shape:
            return out
        numel = torch.Size(shape).numel()
        storage_elems = out.untyped_storage().nbytes() // out.element_size() - out.storage_offset()
        if numel <= storage_elems:
            # Same base pointer, so the direct route still recognizes it; every rank re-views it identically.
            return out.as_strided(shape, torch.empty(shape, device="meta").stride(), out.storage_offset())
    # First call for this slot, or an output too small to hold the new shape (an unregistered staged output, which
    # is freed when replaced). The communicator only grows by being rebuilt, which drops every cached output.
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
def ulysses_fi_o(y: torch.Tensor, group_name: str, world_size: int, copy_out: bool = True) -> torch.Tensor:
    """(B, S, H/P, D) -> (B, S/P, H, D)."""
    y = y.contiguous()
    comm = _communicator(group_name, y.numel() * y.element_size(), y.dtype)
    out = comm.gather_heads(y, out=_output(comm, group_name, y, "gather_heads", 0))
    # The output goes back to model code, so by default it gets its own storage instead of the reused
    # communicator buffer.
    return out.clone() if copy_out else out


@ulysses_fi_o.register_fake
def _(y, group_name, world_size, copy_out=True):
    b, s, h, d = y.shape
    return y.new_empty(b, s // world_size, h * world_size, d)
