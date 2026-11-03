# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Common MixFusion driver: admission, packed-step lifecycle and fallback.

The driver is the model-agnostic entry point for a denoise step. It decides
whether the step's per-request latents may be packed, arms the
:class:`~vllm_omni.diffusion.forward_context.MixFusionStep` around the model's
packed forward call, and otherwise falls back to a serial per-request pass.

It knows nothing about a model's forward signature, token layout or output
shape: those live in the model's
:class:`~vllm_omni.diffusion.mixfusion.adapter.MixFusionAdapter`, which the
driver only calls. Adding a model therefore needs an adapter (and its
``_mixfusion_plan``), not changes here.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import Any

import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.forward_context import MixFusionStep, get_forward_context
from vllm_omni.diffusion.hooks.mixfusion import MixFusionPackingError, plan_foreign_hooks
from vllm_omni.diffusion.mixfusion.adapter import MixFusionAdapter
from vllm_omni.diffusion.mixfusion.execution_plan import plan_chunk_size
from vllm_omni.diffusion.mixfusion.request import DEFAULT_MAX_CHUNKS, DEFAULT_MIN_CHUNK_TOKENS
from vllm_omni.diffusion.worker.input_batch import InputBatch

logger = init_logger(__name__)


class MixFusionDriver:
    """Run one denoise step through the packed path when it is admissible.

    ``transformer`` is the DiT the admitted call runs on; ``adapter`` is its
    model-side MixFusion adapter (``None`` keeps the driver on the serial
    ragged path, which only needs the pipeline's dense ``denoise_step``).
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

        Dense batches (tensor latents) go straight to ``dense_step``. Ragged
        batches (list latents) either pack or fall back to one dense call per
        compatible group of requests. The packed branch still calls
        ``dense_step`` inside the armed step: the pipeline's keyword builder
        accepts list latents and the adapter's root hook rewrites the call
        into the packed layout, so packed and serial steps share one path.

        Admission preflights ``adapter.can_pack``, so the armed root hook
        normally has nothing to refuse. If it nevertheless declines (an
        adapter whose answer drifted between admission and the forward), it
        raises :class:`MixFusionPackingError`; rerun the batch serially rather
        than let list latents reach the model forward.
        """
        if not isinstance(input_batch.latents, list):
            return dense_step(input_batch, states=states)

        if self.pipeline.interrupt:
            return None
        # Model code reads ``pipeline._current_timestep``. The delegated dense
        # step publishes it too, but set it before dispatch so a serial
        # fallback that rebuilds sub-batches never observes a stale step.
        self.pipeline._current_timestep = input_batch.timesteps

        min_chunk_tokens, max_chunks = self._chunk_policy(input_batch)
        accepted, reason = self.admit(input_batch, min_chunk_tokens=min_chunk_tokens, max_chunks=max_chunks)
        if not accepted:
            logger.warning(
                "MixFusion disabled for this batch: %s. Falling back to per-request denoise.",
                reason,
            )
            return self.serial_ragged(input_batch, states, dense_step)

        try:
            with self.armed(min_chunk_tokens=min_chunk_tokens, max_chunks=max_chunks):
                return dense_step(input_batch, states=states)
        except MixFusionPackingError as exc:
            logger.warning(
                "MixFusion packing failed after admission: %s. Falling back to per-request denoise.",
                exc,
            )
            return self.serial_ragged(input_batch, states, dense_step)

    def admit(
        self,
        input_batch: InputBatch,
        *,
        min_chunk_tokens: int,
        max_chunks: int,
    ) -> tuple[bool, str]:
        """Whether this ragged batch may take the packed path.

        Returns ``(accepted, reason)``. Rejection covers both features the
        packed layout cannot express (kept on the serial path) and batches the
        packer declines on size grounds.
        """
        if self.adapter is None:
            return False, f"{type(self.transformer).__name__} declares no MixFusion adapter"
        if not getattr(self.transformer, "_mixfusion_hooks_applied", False):
            logger.warning_once(
                "MixFusion hooks are not installed on %s; falling back to the serial "
                "per-request path. Check that enable_mixfusion is set and the model "
                "declares a _mixfusion_plan.",
                type(self.transformer).__name__,
            )
            return False, "mixfusion hooks not applied"
        # The packed forward drives blocks through the layout-aware
        # ``block.forward``, which has no ``attention_kwargs`` channel; the
        # serial path threads them into every block, so honor a caller's
        # per-attention arguments there instead of silently dropping them.
        if getattr(self.pipeline, "attention_kwargs", None):
            return False, "attention_kwargs is set"
        # Packing splits and stacks the image stream itself; SP would shard
        # rows a second time.
        parallel_config = getattr(self.pipeline, "parallel_config", None)
        if parallel_config is not None and parallel_config.sequence_parallel_size > 1:
            return False, "sequence parallelism is enabled"
        if input_batch.do_true_cfg:
            return False, "do_true_cfg is enabled"
        if input_batch.image_latents is not None:
            return False, "image_latents is set"
        # The packed forward bypasses some module calls (boundaries, block
        # forwards); foreign hooks there would run on a layout they were not
        # written for (layerwise offload, FSDP2, cache hooks), so keep the
        # serial path while any of them is installed.
        foreign = plan_foreign_hooks(
            self.transformer,
            getattr(self.transformer, "_mixfusion_plan", None),
        )
        if foreign:
            return False, f"foreign hooks on the packed path ({', '.join(foreign)})"

        # Preflight the adapter on the exact call the driver would make. A
        # refusal must become the serial fallback *before* the armed forward
        # starts: once the root hook runs, the model is already inside
        # ``transformer.forward`` with list latents and cannot fall back.
        # ``build_packed_call`` only shapes arguments (no projection compute),
        # so binding it costs nothing next to running the model.
        try:
            bound = inspect.signature(self.transformer.forward).bind(**self.adapter.build_packed_call(input_batch))
            packable, pack_reason = self.adapter.can_pack(self.transformer, bound)
        except TypeError as exc:
            return False, f"adapter cannot bind the packed call: {exc}"
        if not packable:
            return False, f"adapter declined packing: {pack_reason or 'no reason given'}"

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

        Requests that ``InputBatch.make_batch`` can express together (same
        latent token shape and same CFG/guidance settings) are stacked into
        one dense batch: the batch dimension is per-image, so requests with
        different ``num_images`` share a call as long as ``S`` and ``C``
        match. A group of one request is exactly the old per-request path.

        Rebuilding the step-local batch per group keeps every per-image
        conditioning row (prompts, guidance, condition images) aligned with
        the group's latents, which a flattened dense batch cannot express.
        The output is per request, in the caller's request order, whether
        ``dense_step`` returns one tensor for the group or a list of them.
        """
        groups: dict[tuple, list[int]] = {}
        for req_idx, latents in enumerate(input_batch.latents):
            state = states[req_idx]
            group_key = (
                tuple(latents.shape[1:]),
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
                # Ragged latents come back as one [num_images, S, C] tensor per
                # request; stacking the group restores a dense batch. The
                # group key guarantees every tensor has the same S and C.
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
    def armed(self, *, min_chunk_tokens: int, max_chunks: int) -> Iterator[None]:
        """Activate the packed layout for the wrapped forward call."""
        ctx = get_forward_context()
        previous = ctx.mixfusion_step
        ctx.mixfusion_step = MixFusionStep(
            min_chunk_tokens=min_chunk_tokens,
            max_chunks=max_chunks,
            adapter=self.adapter,
        )
        try:
            yield
        finally:
            # CFG paths may call the transformer more than once in one step, so
            # restore rather than relying on a per-step context rebuild.
            ctx.mixfusion_step = previous

    @staticmethod
    def _chunk_policy(input_batch: InputBatch) -> tuple[int, int]:
        """Aggregate the per-request chunk policy for the whole batch.

        The strictest per-request thresholds govern the fused pass, so no
        request's limits are violated by packing the batch together.
        """
        request_extras = input_batch.request_extras or []
        if not request_extras:
            return DEFAULT_MIN_CHUNK_TOKENS, DEFAULT_MAX_CHUNKS
        min_chunk_tokens = max(
            int(extra.get("mixfusion_min_chunk_tokens", DEFAULT_MIN_CHUNK_TOKENS)) for extra in request_extras
        )
        max_chunks = min(int(extra.get("mixfusion_max_chunks", DEFAULT_MAX_CHUNKS)) for extra in request_extras)
        return min_chunk_tokens, max_chunks
