# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Portable destination-major Ulysses exchanges for equal-shaped Q/K/V.

These inference-only helpers use the process group's ordinary all-to-all
implementation, including RCCL on ROCm. They do not use peer-memory mappings,
persistent scratch buffers, or any CUDA-specific communication implementation.
"""

from __future__ import annotations

import torch
import torch.distributed as dist


def packed_qkv_eligible(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, world_size: int) -> bool:
    """The strict S<->H layout supports equal-shaped, equally typed Q/K/V.

    Different local sequence lengths across ranks are not supported, just as
    with the existing strict fixed-split Ulysses exchange. Padded sequence rows
    are ordinary rows and retain their existing positions and masks.
    """
    return (
        world_size > 1
        and query.ndim == 4
        and query.shape == key.shape == value.shape
        and query.dtype == key.dtype == value.dtype
        and query.device == key.device == value.device
        and all(size > 0 for size in query.shape)
        and query.shape[2] % world_size == 0
        and not (query.requires_grad or key.requires_grad or value.requires_grad)
    )


def _synchronize(tensor: torch.Tensor, use_sync: bool) -> None:
    if use_sync and tensor.is_cuda:
        torch.accelerator.synchronize(tensor.device)


def packed_qkv_all_to_all(
    group: dist.ProcessGroup,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    use_sync: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exchange three [B,S_local,H,D] tensors in one collective.

    Unlike stacking Q/K/V before all_to_all_5D, the three strided producers
    copy directly into the wire buffer without a full stacked intermediate.
    The wire order is [destination,B,S_local,H_local,QKV,D]. On receive,
    source-rank order forms the global sequence dimension. Explicit copy_ calls
    accept strided Q/K/V views without assuming a common producer/storage.
    Every call owns its send/receive buffers; returned tensors are independent,
    contiguous allocations and remain valid after later calls.
    """
    world_size = dist.get_world_size(group)
    if not packed_qkv_eligible(query, key, value, world_size):
        raise ValueError("Packed Ulysses Q/K/V requires equal 4D inference tensors and divisible heads")
    batch, tokens, heads, width = query.shape
    local_heads = heads // world_size
    send = torch.empty((world_size, batch, tokens, local_heads, 3, width), dtype=query.dtype, device=query.device)
    for slot, tensor in enumerate((query, key, value)):
        # reshape may copy for exotic strides; the following copy remains
        # correct for those layouts as well as normal fused-QKV views.
        send[..., slot, :].copy_(tensor.reshape(batch, tokens, world_size, local_heads, width).permute(2, 0, 1, 3, 4))
    receive = torch.empty_like(send)
    dist.all_to_all_single(receive, send, group=group)
    _synchronize(query, use_sync)
    outputs = [
        receive[..., slot, :]
        .permute(1, 0, 2, 3, 4)
        .reshape(batch, world_size * tokens, local_heads, width)
        .contiguous()
        for slot in range(3)
    ]
    return outputs[0], outputs[1], outputs[2]


def packed_output_all_to_all(group: dist.ProcessGroup, output: torch.Tensor, use_sync: bool = False) -> torch.Tensor:
    """Reverse [B,S_global,H_local,D] to [B,S_local,H,D] in one collective."""
    world_size = dist.get_world_size(group)
    if output.ndim != 4 or output.shape[1] % world_size or output.requires_grad:
        raise ValueError("Packed Ulysses output requires a 4D inference tensor and divisible sequence length")
    batch, tokens, local_heads, width = output.shape
    local_tokens = tokens // world_size
    # Destination rank owns a contiguous segment of global sequence rows.
    send = output.reshape(batch, world_size, local_tokens, local_heads, width).permute(1, 0, 2, 3, 4).contiguous()
    receive = torch.empty_like(send)
    dist.all_to_all_single(receive, send, group=group)
    _synchronize(output, use_sync)
    # Source rank owns its original head shard, preserving head order.
    return receive.permute(1, 2, 0, 3, 4).reshape(batch, local_tokens, world_size * local_heads, width).contiguous()
