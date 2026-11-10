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

        if not input_batch.use_mixfusion:
            if isinstance(input_batch.latents, list):
                raise ValueError("Ragged latents require a scheduler-approved MixFusion batch")
            return
        reason = packing_unavailable_reason(self.pipeline)
        if reason is not None:
            raise ValueError(f"Scheduled MixFusion runtime changed: {reason}")
        if not isinstance(input_batch.latents, list):
            raise ValueError("Scheduled MixFusion batch requires per-request latents")
        for request_id, latent, state in zip(
            input_batch.request_ids, input_batch.latents, input_batch.states, strict=True
        ):
            info = state.mixfusion_info
            if state.request_id != request_id or info is None:
                raise ValueError(f"MixFusion request {request_id}: missing or mismatched admission metadata")
            if latent.ndim != 3 or tuple(latent.shape[:2]) != (info.num_images, info.image_tokens):
                raise ValueError(f"MixFusion request {request_id}: actual latents disagree with admitted token counts")
        first = input_batch.latents[0]
        if any(
            latent.shape[2] != first.shape[2] or latent.dtype != first.dtype or latent.device != first.device
            for latent in input_batch.latents
        ):
            raise ValueError("MixFusion latents must share channels, dtype and device")
        self.adapter.validate_inputs(self.transformer, input_batch)

    def run(self, input_batch: InputBatch, states: Sequence[Any], dense_step: Callable[..., Any]) -> Any:
        if not input_batch.use_mixfusion:
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
        with self.armed():
            return dense_step(input_batch, states=states)

    @contextmanager
    def armed(self) -> Iterator[None]:
        ctx = get_forward_context()
        previous = ctx.mixfusion_step
        ctx.mixfusion_step = MixFusionStep()
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
