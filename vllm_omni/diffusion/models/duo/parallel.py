# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""TP publication consensus, outside captured compute/allocator transactions."""

import torch
from vllm.distributed import get_tp_group


def all_ranks_finite(finite, device):
    """Every rank acknowledges GPU completion before any rank publishes KV.

    Not a distributed exception-recovery protocol: rank loss must terminate the
    worker group. Caller enters this boundary in identical request order.
    """
    group = get_tp_group()
    status = torch.as_tensor(finite, dtype=torch.int32, device=device).reshape(1)
    if group.world_size > 1:
        torch.distributed.all_reduce(status, op=torch.distributed.ReduceOp.MIN, group=group.device_group)
    return bool(status.item())
