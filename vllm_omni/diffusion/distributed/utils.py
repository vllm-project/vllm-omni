# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import torch

from vllm_omni.platforms import current_omni_platform


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
