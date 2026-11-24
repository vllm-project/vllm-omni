# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""310P replacements for the distributed VAE result exchange.

Two 310P HCCL quirks make the shared executor's exchange unusable:

* ``dist.broadcast`` returns without error but leaves the receivers' buffers
  untouched (measured on Atlas 300I Duo, every dtype), so every non-zero rank
  would hand back uninitialized memory for the decoded tensor and for the
  output shape. 910B HCCL implements broadcast, so this is 310P-only.
  An ``all_reduce`` with rank 0 contributing the payload and every other rank
  contributing zeros sums to exactly the payload for every dtype HCCL accepts
  (FP16/BF16/FP32/INT8/INT16/INT32), which covers the decoded tiles.
* The collective dtype set is INT8/INT16/INT32/FP16/FP32 only. An int64
  ``all_reduce``/``gather`` either fails the operator check or silently
  communicates nothing, and the ranks then agree on no padding shape, pack
  different numbers of tiles and fail the merge. The shared executor creates
  its control tensors (tile counts, padding shape, tile metadata, output shape)
  in ``_control_dtype``, which defaults to int64; here it is int32. Tile counts
  and tensor dims fit int32 by a wide margin.
"""

from __future__ import annotations

import torch
import torch.distributed as dist


def publish_tensor_310p(rank: int, tensor: torch.Tensor, group) -> None:
    """Replicate rank 0's ``tensor`` to every rank of ``group``, in place."""
    if rank != 0:
        tensor.zero_()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)


def apply_patch() -> None:
    from vllm_omni.diffusion.distributed.autoencoders.distributed_vae_executor import (
        DistributedVaeExecutor,
    )

    if getattr(DistributedVaeExecutor, "_310p_publish_patched", False):
        return

    def _publish_tensor(self, tensor: torch.Tensor) -> None:
        publish_tensor_310p(self.rank, tensor, self.group)

    DistributedVaeExecutor._publish_tensor = _publish_tensor  # type: ignore[method-assign]
    DistributedVaeExecutor._control_dtype = torch.int32
    DistributedVaeExecutor._310p_publish_patched = True  # type: ignore[attr-defined]
