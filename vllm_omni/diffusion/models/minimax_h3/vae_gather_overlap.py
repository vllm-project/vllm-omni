# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Opt-in stream separation for the unchanged paired full-H3 VAE gather.

This changes scheduling only: tensors, collective geometry, decoder batches,
and spatial/temporal blending remain owned by the original implementations.
"""

import os

import torch

ENV = "VLLM_OMNI_H3_VAE_GATHER_OVERLAP"


def enabled(*, paired: bool, mixed: bool) -> bool:
    value = os.environ.get(ENV, "0")
    if value not in ("0", "1"):
        raise ValueError(f"{ENV} must be 0 or 1")
    if value == "1" and not (paired and mixed):
        raise ValueError(f"{ENV}=1 requires paired full-H3 VAE and mixed B2/B1")
    return value == "1"


def prepare_stream(device, *, paired: bool, mixed: bool, owner):
    """Run during existing all-rank sink/setup admission, before any gather."""
    if not enabled(paired=paired, mixed=mixed):
        return None
    if device.type != "cuda":
        raise ValueError(f"{ENV}=1 requires CUDA")
    for name in ("TORCH_NCCL_BLOCKING_WAIT", "NCCL_BLOCKING_WAIT"):
        if os.environ.get(name, "0") != "0":
            raise ValueError(f"{ENV}=1 requires {name} unset or 0")
    # This VAE already requires serial decode requests (mixed_decode temporarily
    # wraps model.decode). Reuse one stream per instance/device after the prior
    # request's unconditional drain, so allocator blocks retain a stable owner.
    index = device.index if device.index is not None else torch.accelerator.current_device_index()
    streams = getattr(owner, "_h3_vae_gather_streams", None)
    if streams is None:
        streams = {}
        owner._h3_vae_gather_streams = streams
    if index not in streams:
        streams[index] = torch.get_device_module().Stream(device=device)
    return streams[index]


def gather_on_stream(packed, group, stream):
    """Keep the NCCL completion dependency off the decoder caller stream.

    The existing synchronous Python gather waits for NCCL on its current CUDA
    stream. Here that current stream is dedicated to gather, so the subsequent
    decoder round may be submitted on the original caller stream. The leader
    must wait on this gather stream before consuming the returned tensor.
    """
    from .vae_collectives import _gather_stack_to_rank_zero

    caller = torch.get_device_module().current_stream(packed.device)
    if stream == caller:
        raise RuntimeError("VAE gather overlap requires a distinct stream")
    stream.wait_stream(caller)
    # `packed` was allocated/written on caller. Its Python lifetime ends after
    # this round is submitted, potentially before the gather stream finishes.
    packed.record_stream(stream)
    with torch.get_device_module().stream(stream):
        return _gather_stack_to_rank_zero(packed, group)
