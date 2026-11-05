# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Bind fast-ulysses copy-engine transport to Omni's Ulysses subgroups."""

from __future__ import annotations

import socket
from threading import Lock
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

from vllm_omni.diffusion.distributed.group_coordinator import (
    SequenceParallelGroupCoordinator,
)

_GROUPS: dict[str, Any] = {}
_GROUPS_LOCK = Lock()
_OUTPUT_BUFFERS: dict[tuple[str, int], tuple[tuple[Any, ...], torch.Tensor]] = {}
_OUTPUT_BUFFERS_LOCK = Lock()


def _load_fast_ulysses() -> Any:
    try:
        from fast_ulysses import UlyssesGroup
    except ImportError as exc:
        raise RuntimeError(
            "CE Ulysses requires the fast-ulysses wheel matching PyTorch/CUDA: "
            "https://triple-mu.github.io/fast-ulysses/whl/"
        ) from exc
    return UlyssesGroup


def _group_for(sp_group: SequenceParallelGroupCoordinator, device: torch.device) -> Any:
    """Return the cached fast-ulysses group for this Omni SP process group."""
    # The group name identifies the same symmetric-memory rendezvous on every rank.
    key = sp_group.ulysses_group.group_name
    with _GROUPS_LOCK:
        group = _GROUPS.get(key)
        if group is None:
            UlyssesGroup = _load_fast_ulysses()
            hosts = [None] * sp_group.ulysses_world_size
            dist.all_gather_object(hosts, socket.gethostname(), group=sp_group.ulysses_group)
            if len(set(hosts)) != 1:
                raise ValueError("CE Ulysses requires all ranks of each Ulysses group on one NVLink-connected node")
            if symm_mem.get_backend(device) != "CUDA":
                symm_mem.set_backend("CUDA")
            group = UlyssesGroup(sp_group.ulysses_group, device)
            _GROUPS[key] = group
        return group


def _get_output_buffer(
    tensor: torch.Tensor, group_name: str, world_size: int, mode: int, slot: int
) -> torch.Tensor | None:
    """Get one reusable symmetric output, or miss while CUDA Graph capture is active."""
    if slot < 0 or slot > 3:
        raise ValueError(f"CE Ulysses output slot must be in [0, 3], got {slot}")
    signature = (tuple(tensor.shape), tensor.dtype, tensor.device, world_size, mode)
    key = (group_name, slot)
    with _OUTPUT_BUFFERS_LOCK:
        cached = _OUTPUT_BUFFERS.get(key)
        if cached is not None and cached[0] == signature:
            return cached[1]
        if torch.cuda.is_current_stream_capturing():
            # Collective symmetric allocation must happen during eager warmup.
            return None
        output = _GROUPS[group_name].empty_output(tensor, mode=mode)
        _OUTPUT_BUFFERS[key] = (signature, output)
        return output


@torch.library.custom_op("vllm_omni_ce::ulysses_all_to_all_4d", mutates_args=(), device_types="cuda")
def ulysses_all_to_all_4d(
    tensor: torch.Tensor, group_name: str, world_size: int, mode: int, slot: int = -1
) -> torch.Tensor:
    """Exchange op opaque to torch.compile; slot -1 owns its output, 0..3 reuse Q/K/V/O buffers."""
    group = _GROUPS.get(group_name)
    if group is None:
        raise RuntimeError(f"CE Ulysses group {group_name!r} was not initialized by Omni")
    output = _get_output_buffer(tensor, group_name, world_size, mode, slot) if slot != -1 else None
    return group.all_to_all_4d(tensor, mode=mode, out=output)


@ulysses_all_to_all_4d.register_fake
def _ulysses_all_to_all_4d_fake(
    tensor: torch.Tensor, group_name: str, world_size: int, mode: int, slot: int = -1
) -> torch.Tensor:
    del group_name
    if slot < -1 or slot > 3:
        raise ValueError(f"CE Ulysses output slot must be in [-1, 3], got {slot}")
    batch, sequence, heads, head_size = tensor.shape
    if mode == 0:
        return tensor.new_empty((batch, sequence * world_size, heads // world_size, head_size))
    if mode == 1:
        return tensor.new_empty((batch, sequence // world_size, heads * world_size, head_size))
    raise ValueError(f"CE Ulysses mode must be 0 or 1, got {mode}")


def clear_ulysses_groups() -> None:
    """Collectively release fast-ulysses resources during worker shutdown."""
    with _OUTPUT_BUFFERS_LOCK:
        for device in {buffer.device for _, buffer in _OUTPUT_BUFFERS.values()}:
            torch.accelerator.synchronize(device)
        _OUTPUT_BUFFERS.clear()
    with _GROUPS_LOCK:
        groups = list(_GROUPS.values())
        _GROUPS.clear()
    for group in groups:
        group.destroy()


def initialize_ulysses_groups(sp_group: SequenceParallelGroupCoordinator) -> None:
    """Initialize/validate CE groups during worker model initialization."""
    if sp_group.ulysses_world_size > 1:
        _group_for(sp_group, torch.device("cuda", torch.accelerator.current_device_index()))
