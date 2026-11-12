# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import os

import torch
from vllm.utils.network_utils import (
    aiter_requires_tcp_store,
    get_distributed_init_method,
    get_file_store_init_method,
    get_ip,
    get_loopback_ip,
    get_open_port,
)

from vllm_omni.platforms import current_omni_platform


def get_diffusion_worker_init_method(*, use_loopback: bool = True) -> str:
    """Select a rendezvous for a local diffusion worker.

    Mirror vLLM's executors: multiprocess workers use the loopback address,
    while the in-process worker uses the host IP. FileStore avoids reserving a
    TCP rendezvous port; AITER custom all-reduce requires a TCPStore, so use an
    explicit TCP rendezvous when that implementation is in use.
    """
    if aiter_requires_tcp_store():
        host = get_loopback_ip() if use_loopback else get_ip()
        return get_distributed_init_method(host, get_open_port())
    return get_file_store_init_method()


def get_local_device() -> torch.device:
    """Return the torch device for the current rank based on detected device type."""
    if current_omni_platform.is_initialized():
        return current_omni_platform.get_torch_device(current_omni_platform.current_device())
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    return current_omni_platform.get_torch_device(local_rank)


def build_local_sp_padding_mask(
    batch_size: int,
    local_seq_len: int,
    sp_original_seq_len: int | None,
    sp_padding_size: int,
    sequence_parallel_rank: int,
    device,
):
    """Build a per-rank SP padding mask that matches the local shard shape."""
    if sp_original_seq_len is None or sp_padding_size <= 0:
        return None

    shard_start = sequence_parallel_rank * local_seq_len
    shard_end = shard_start + local_seq_len
    valid_tokens = max(0, min(sp_original_seq_len, shard_end) - shard_start)

    if valid_tokens >= local_seq_len:
        return None

    token_mask = torch.arange(local_seq_len, device=device) < valid_tokens
    return token_mask.unsqueeze(0).expand(batch_size, -1)
