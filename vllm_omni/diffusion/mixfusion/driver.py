# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Common MixFusion driver: admission, packed-step lifecycle and fallback.

The model-agnostic entry point for a denoise step: it decides whether the
per-request latents may be packed, arms the
:class:`~vllm_omni.diffusion.forward_context.MixFusionStep` around the packed
forward, and otherwise falls back to a serial per-request pass. Model-specific
knowledge (forward signature, token layout, output shape) comes from the
model's ``_mixfusion_plan`` through its adapter (the framework's
:class:`~vllm_omni.diffusion.mixfusion.chunked.ChunkedJointAdapter` for a
declarative plan).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import Any

import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.forward_context import MixFusionStep, get_forward_context
from vllm_omni.diffusion.hooks.mixfusion import plan_foreign_hooks
from vllm_omni.diffusion.mixfusion.adapter import MixFusionAdapter
from vllm_omni.diffusion.mixfusion.execution_plan import plan_chunk_size
from vllm_omni.diffusion.mixfusion.request import (
    DEFAULT_MAX_CHUNKS,
    DEFAULT_MIN_CHUNK_TOKENS,
    RELAXED_STEP_KEY_FIELDS,
)
from vllm_omni.diffusion.worker.input_batch import InputBatch

logger = init_logger(__name__)


class MixFusionDriver:
    """Run one denoise step through the packed path when it is admissible.

    ``adapter`` packs the transformer's root call; ``None`` keeps the driver
    on the serial ragged path.
    """

    def __init__(
        self,
        pipeline: Any,
        transformer: Any,
        adapter: MixFusionAdapter | None,
    ) -> None:
        self.pipeline = pipeline
        self.transformer = transformer
        self.adapter = adapter

    def run(
        self,
        input_batch: InputBatch,
        states: Sequence[Any],
        dense_step: Callable[..., Any],
    ) -> Any:
        """Denoise ``input_batch``, packed when possible.

        Dense batches go straight to ``dense_step``. Ragged batches either
        pack or fall back to one dense call per compatible request group. The
        packed branch still calls ``dense_step`` inside the armed step (the
        root hook rewrites the call into the packed layout), so both paths
        share one keyword builder.
        """
        if not isinstance(input_batch.latents, list):
            return dense_step(input_batch, states=states)
        if self.pipeline.interrupt:
            return None

        min_chunk_tokens, max_chunks = self._chunk_policy(input_batch)
        accepted, reason = self.admit(input_batch, min_chunk_tokens=min_chunk_tokens, max_chunks=max_chunks)
        if not accepted:
            logger.warning(
                "MixFusion disabled for this batch: %s. Falling back to per-request denoise.",
                reason,
            )
            return self.serial_ragged(input_batch, states, dense_step)

        with self.armed():
            return dense_step(input_batch, states=states)

    def admit(
        self,
        input_batch: InputBatch,
        *,
        min_chunk_tokens: int,
        max_chunks: int,
    ) -> tuple[bool, str]:
        """Whether this ragged batch may take the packed path.

        The single admission point: once accepted, the armed hooks pack
        unconditionally. Returns ``(accepted, reason)``.
        """
        if self.adapter is None:
            return False, f"{type(self.transformer).__name__} declares no packable MixFusion plan"
        if not getattr(self.transformer, "_mixfusion_hooks_applied", False):
            return False, "mixfusion hooks not applied"
        # The packed forward has no ``attention_kwargs`` channel; the serial
        # path threads them into every block.
        if getattr(self.pipeline, "attention_kwargs", None):
            return False, "attention_kwargs is set"
        parallel_config = getattr(self.pipeline, "parallel_config", None)
        # Packing stacks the image stream itself; SP would shard rows again.
        if parallel_config is not None and parallel_config.sequence_parallel_size > 1:
            return False, "sequence parallelism is enabled"
        # CFG parallel all-gathers tensor predictions per rank, which has no
        # per-request list semantics. Sequential true-CFG is fine.
        if parallel_config is not None and parallel_config.cfg_parallel_size > 1:
            return False, "CFG parallel is enabled"
        # Hooks on modules the packed path drives would see a layout they were
        # not written for (layerwise offload, FSDP2, cache hooks).
        foreign = plan_foreign_hooks(self.transformer, getattr(self.transformer, "_mixfusion_plan", None))
        if foreign:
            return False, f"foreign hooks on the packed path ({', '.join(foreign)})"
        packable, pack_reason = self.adapter.can_pack(self.transformer, input_batch)
        if not packable:
            return False, f"adapter declined packing: {pack_reason}"

        accepted, reason, _, _ = plan_chunk_size(
            input_batch.latents,
            min_chunk_tokens=min_chunk_tokens,
            max_chunks=max_chunks,
        )
        return accepted, reason

    def serial_ragged(
        self,
        input_batch: InputBatch,
        states: Sequence[Any],
        dense_step: Callable[..., Any],
    ) -> list[Any]:
        """Denoise the ragged batch through the dense path, one call per group.

        A dense call takes one image geometry, so requests share a group only
        if they also matched on the step-batch key fields MixFusion relaxed
        (and on ``img_shapes``): equal-area images with different aspect
        ratios share ``S`` but not their positional embedding. CFG/guidance
        settings must match too. The batch dimension is per-image, so
        different ``num_images`` can share a call. Rebuilding the batch per
        group keeps per-image conditioning rows aligned with its latents.
        Outputs are returned per request, in caller order, whether
        ``dense_step`` returns a tensor or a list.
        """
        groups: dict[tuple, list[int]] = {}
        for req_idx, latents in enumerate(input_batch.latents):
            state = states[req_idx]
            group_key = (
                tuple(latents.shape[1:]),
                _freeze(getattr(state, "img_shapes", None)),
                *(getattr(state.sampling, name, None) for name in RELAXED_STEP_KEY_FIELDS),
                state.do_true_cfg,
                getattr(state.sampling, "true_cfg_scale", None) or 4.0,
                bool(getattr(state.sampling, "cfg_normalize", False)),
                state.guidance is None,
            )
            groups.setdefault(group_key, []).append(req_idx)

        outputs: list[Any] = [None] * len(input_batch.latents)
        for indices in groups.values():
            sub_batch = InputBatch.make_batch(
                states,
                idx_mapping=torch.tensor(indices, dtype=torch.int32),
            )
            if isinstance(sub_batch.latents, list):
                # Ragged latents are one [num_images, S, C] tensor per request;
                # the group key guarantees the same S and C.
                sub_batch.latents = torch.cat(sub_batch.latents, dim=0)
            pred = dense_step(sub_batch, states=sub_batch.states)
            if isinstance(pred, (list, tuple)):
                for offset, req_idx in enumerate(indices):
                    outputs[req_idx] = pred[offset]
                continue
            offset = 0
            for req_idx in indices:
                rows = input_batch.latents[req_idx].shape[0]
                outputs[req_idx] = pred[offset : offset + rows]
                offset += rows
        return outputs

    @contextmanager
    def armed(self) -> Iterator[None]:
        """Activate the packed layout for the wrapped forward call."""
        ctx = get_forward_context()
        previous = ctx.mixfusion_step
        ctx.mixfusion_step = MixFusionStep()
        try:
            yield
        finally:
            # CFG paths may call the transformer more than once in one step, so
            # restore rather than relying on a per-step context rebuild.
            ctx.mixfusion_step = previous

    @staticmethod
    def _chunk_policy(input_batch: InputBatch) -> tuple[int, int]:
        """Aggregate the per-request chunk policy for the whole batch.

        The strictest per-request thresholds govern the fused pass.
        """
        request_extras = input_batch.request_extras or []
        if not request_extras:
            return DEFAULT_MIN_CHUNK_TOKENS, DEFAULT_MAX_CHUNKS
        min_chunk_tokens = max(
            int(extra.get("mixfusion_min_chunk_tokens", DEFAULT_MIN_CHUNK_TOKENS)) for extra in request_extras
        )
        max_chunks = min(int(extra.get("mixfusion_max_chunks", DEFAULT_MAX_CHUNKS)) for extra in request_extras)
        return min_chunk_tokens, max_chunks


def _freeze(value: Any) -> Any:
    """Hashable form of nested ``list``/``tuple`` metadata such as ``img_shapes``."""
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value
