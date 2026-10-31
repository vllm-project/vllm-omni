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

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import Any

import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.forward_context import MixFusionStep, get_forward_context
from vllm_omni.diffusion.hooks.mixfusion import plan_foreign_hooks
from vllm_omni.diffusion.mixfusion.adapter import MixFusionAdapter
from vllm_omni.diffusion.mixfusion.execution_plan import plan_chunk_size
from vllm_omni.diffusion.worker.input_batch import InputBatch

logger = init_logger(__name__)

# Defaults for requests that do not carry per-request MixFusion extras.
DEFAULT_MIN_CHUNK_TOKENS = 256
DEFAULT_MAX_CHUNKS = 128


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
        request. The packed branch must not call ``dense_step``: the dense
        keyword builder only accepts a tensor of latents.
        """
        if not isinstance(input_batch.latents, list):
            return dense_step(input_batch, states=states)

        if self.pipeline.interrupt:
            return None
        # The packed branch calls the transformer directly, bypassing the
        # pipeline's dense wrapper that records the current timestep for model
        # code reading ``pipeline._current_timestep``.
        self.pipeline._current_timestep = input_batch.timesteps

        min_chunk_tokens, max_chunks = self._chunk_policy(input_batch)
        accepted, reason = self.admit(input_batch, min_chunk_tokens=min_chunk_tokens, max_chunks=max_chunks)
        if not accepted:
            logger.warning(
                "MixFusion disabled for this batch: %s. Falling back to per-request denoise.",
                reason,
            )
            return self.serial_ragged(input_batch, states, dense_step)

        with self.armed(min_chunk_tokens=min_chunk_tokens, max_chunks=max_chunks):
            return self.transformer(**self.adapter.build_packed_call(input_batch))

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
        """Denoise each request on its own through the dense path.

        Rebuilding the step-local batch per request keeps every per-image
        conditioning row (prompts, guidance, condition images) aligned with
        the request's latents, which a flattened dense batch cannot express.
        """
        outputs: list[Any] = []
        num_requests = len(input_batch.latents)
        for req_idx in range(num_requests):
            sub_batch = InputBatch.make_batch(
                states,
                idx_mapping=torch.tensor([req_idx], dtype=torch.int32),
            )
            if isinstance(sub_batch.latents, list):
                # A single ragged request still reports list latents; flatten
                # it so the dense path sees the request's [num_images, S, C]
                # tensor (and does not recurse back here).
                sub_batch.latents = sub_batch.latents[0]
            outputs.append(dense_step(sub_batch, states=sub_batch.states))
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
