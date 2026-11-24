# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import os
from collections.abc import Iterator
from contextlib import contextmanager

import torch
import torch.distributed as dist

from vllm_omni.platforms import current_omni_platform


def get_local_device() -> torch.device:
    """Return the torch device for the current rank based on detected device type."""
    if current_omni_platform.is_initialized():
        return current_omni_platform.get_torch_device(current_omni_platform.current_device())
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    return current_omni_platform.get_torch_device(local_rank)


@contextmanager
def sync_stage_errors(stage: str, cpu_group: dist.ProcessGroup, status_tensor: torch.Tensor) -> Iterator[None]:
    """Agree on local-stage failures before entering the next collective."""
    status = 0
    message = ""
    try:
        yield
    except torch.OutOfMemoryError as error:
        status = 1
        message = str(error)
    except Exception as error:
        # A non-retryable error takes precedence over OOM on another rank.
        status = 2
        message = f"{type(error).__name__}: {error}"

    status_tensor.fill_(status)
    dist.all_reduce(status_tensor, op=dist.ReduceOp.MAX, group=cpu_group)
    global_status = status_tensor.item()
    if global_status == 2:
        detail = message if status == 2 else "another rank failed"
        raise RuntimeError(f"{stage} failed: {detail}")
    if global_status == 1:
        detail = message if status == 1 else "another rank ran out of memory"
        raise torch.OutOfMemoryError(f"{stage} failed: {detail}")


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
