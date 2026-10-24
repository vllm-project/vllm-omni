# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Rank-local, bounded batches for the checkpoint's spatial decoder tiles."""

from collections.abc import Callable, Sequence
from datetime import timedelta

import torch
import torch.distributed as dist
from torch import nn


class TileBatchExecutionError(RuntimeError):
    """A failed rank group must be restarted rather than reused for a request."""

    requires_stage_shutdown = True


class TileBatchExecutor:
    """Replace one model instance's task runner, preserving encoder dispatch."""

    def __init__(
        self,
        model: nn.Module,
        batch_size: int,
        cpu_group: dist.ProcessGroup | None = None,
        control_timeout: timedelta = timedelta(seconds=120),
    ) -> None:
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("vae_tile_batch_size must be a positive integer")
        self.model = model
        self.batch_size = batch_size
        self.cpu_group = cpu_group
        self.control_timeout = control_timeout
        self.original = getattr(model, "_run_tile_tasks", None)
        self.decode = getattr(model, "decode", None)
        if not callable(self.original) or not callable(self.decode) or not hasattr(self.decode, "__func__"):
            raise ValueError("MiniMax H3 checkpoint does not expose the supported decoder tile contract")
        if model.training:
            raise ValueError("VAE tile batching requires eval mode")
        # CPU-only status; never allocate a device tensor after a local OOM.
        self.status = torch.zeros(1, dtype=torch.int32, device="cpu")

    def run(
        self,
        tiles: Sequence[torch.Tensor],
        tile_indices: Sequence[int],
        forward_fn: Callable[[torch.Tensor], torch.Tensor],
        stack_tiling: bool,
        cls_agg: object | None = None,
    ) -> list[torch.Tensor]:
        is_decode = (
            getattr(forward_fn, "__self__", None) is self.model
            and getattr(forward_fn, "__func__", None) is self.decode.__func__
        )
        if not is_decode:
            return self.original(tiles, tile_indices, forward_fn, stack_tiling, cls_agg)

        error: Exception | None = None
        outputs: list[torch.Tensor] = []
        try:
            if self.model.training or cls_agg is not None:
                raise ValueError("decoder tile batching requires inference without CLS aggregation")
            for start in range(0, len(tile_indices), self.batch_size):
                indices = tile_indices[start : start + self.batch_size]
                inputs = [tiles[index] for index in indices]
                first = inputs[0]
                if any(x.shape != first.shape or x.dtype != first.dtype or x.device != first.device for x in inputs):
                    raise ValueError("decoder tile batch must have matching shapes, dtypes and devices")
                sample_batch = first.shape[0]
                merged = first if len(inputs) == 1 else torch.cat(inputs, dim=0)
                decoded = forward_fn(merged)
                if decoded.shape[0] != len(inputs) * sample_batch:
                    raise RuntimeError("decoder changed the sample batch dimension")
                outputs.extend(decoded.split(sample_batch, dim=0))
                del merged, decoded, inputs
        except Exception as exc:
            # Preserve the local failure until peers have completed their local
            # work, so no successful peer enters the checkpoint's device gather.
            error = exc
            outputs.clear()
        if self.cpu_group is not None:
            self.status.fill_(int(error is not None))
            try:
                dist.monitored_barrier(group=self.cpu_group, timeout=self.control_timeout, wait_all_ranks=True)
                dist.all_reduce(self.status, op=dist.ReduceOp.MAX, group=self.cpu_group)
            except RuntimeError as exc:
                raise TileBatchExecutionError("MiniMax H3 decoder tile control group failed") from exc
            if self.status.item():
                raise TileBatchExecutionError("MiniMax H3 decoder tile execution failed on a VAE rank") from error
        if error is not None:
            raise error
        return outputs
