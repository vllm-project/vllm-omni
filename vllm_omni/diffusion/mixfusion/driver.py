# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Execute the scheduler's MixFusion batch without admitting or regrouping requests."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import Any

import torch
import torch.distributed as dist

from vllm_omni.diffusion.forward_context import MixFusionStep, get_forward_context
from vllm_omni.diffusion.mixfusion.adapter import MixFusionAdapter
from vllm_omni.diffusion.worker.input_batch import InputBatch


class MixFusionDriver:
    """Arm the approved layout around the pipeline's ordinary denoise step."""

    def __init__(self, pipeline: Any, transformer: Any, adapter: MixFusionAdapter) -> None:
        self.pipeline = pipeline
        self.transformer = transformer
        self.adapter = adapter

    def validate(self, input_batch: InputBatch) -> None:
        """Check the execution contract; disagreement is an error, never a fallback."""
        from vllm_omni.diffusion.mixfusion.setup import packing_unavailable_reason

        spec = input_batch.mixfusion_batch
        if spec is None:
            if isinstance(input_batch.latents, list):
                raise ValueError("Ragged latents require a scheduler-approved MixFusion batch")
            return
        reason = packing_unavailable_reason(self.pipeline)
        if reason is not None:
            raise ValueError(f"Scheduled MixFusion runtime changed: {reason}")
        if not isinstance(input_batch.latents, list):
            raise ValueError("Scheduled MixFusion batch requires per-request latents")
        for request_id, latent in zip(input_batch.request_ids, input_batch.latents, strict=True):
            expected = spec.request_shapes.get(request_id)
            if tuple(latent.shape) != expected:
                raise ValueError(
                    f"MixFusion request {request_id}: scheduled latent shape {expected}, actual {tuple(latent.shape)}"
                )
            if spec.chunk_size <= 0 or latent.shape[1] % spec.chunk_size:
                raise ValueError(f"MixFusion request {request_id}: invalid scheduled chunk_size={spec.chunk_size}")
        self.adapter.validate_inputs(self.transformer, input_batch)

    def run(self, input_batch: InputBatch, states: Sequence[Any], dense_step: Callable[..., Any]) -> Any:
        if input_batch.mixfusion_batch is None:
            self.validate(input_batch)
            return dense_step(input_batch, states=states)
        if self.pipeline.interrupt:
            return None
        error = None
        try:
            self.validate(input_batch)
        except (ValueError, RuntimeError) as exc:
            error = exc
        if _any_rank_failed(error is not None):
            if error is not None:
                raise error
            raise RuntimeError("MixFusion execution contract failed on another DiT rank")
        with self.armed(input_batch.mixfusion_batch.chunk_size):
            return dense_step(input_batch, states=states)

    @contextmanager
    def armed(self, chunk_size: int) -> Iterator[None]:
        ctx = get_forward_context()
        previous = ctx.mixfusion_step
        ctx.mixfusion_step = MixFusionStep(chunk_size=chunk_size)
        try:
            yield
        finally:
            ctx.mixfusion_step = previous


def _any_rank_failed(local_failed: bool) -> bool:
    """Stop all executor ranks before model collectives on contract failure."""
    if not dist.is_initialized():
        return local_failed
    from vllm_omni.diffusion.distributed.parallel_state import get_world_group

    world = get_world_group()
    if world.world_size == 1:
        return local_failed
    # This is host control state, so use the existing Gloo group rather than
    # introducing a device allocation/synchronization on every denoise step.
    failed = torch.tensor(int(local_failed), dtype=torch.int32)
    dist.all_reduce(failed, op=dist.ReduceOp.MAX, group=world.cpu_group)
    return bool(failed.item())
