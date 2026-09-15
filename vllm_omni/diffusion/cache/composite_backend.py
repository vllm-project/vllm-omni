# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Composite cache backend combining inter_request and cache_dit.

This allows both cross-request latent reuse (inter_request) and intra-request
block-level caching (cache_dit) to work together.

Usage:
    omni = Omni(
        model="Qwen/Qwen-Image",
        cache_backend="inter_request+cache_dit",
        cache_config={
            # inter_request params
            "inter_request_clip_model_path": "/path/to/clip",
            "inter_request_clip_threshold": 0.65,
            # cache_dit params
            "Fn_compute_blocks": 1,
            "max_warmup_steps": 4,
        },
    )
"""

from __future__ import annotations

import logging
from typing import Any

from vllm_omni.diffusion.cache.cachedit.backend import CacheDiTBackend
from vllm_omni.diffusion.cache.inter_request.backend import InterRequestCacheBackend

logger = logging.getLogger(__name__)


class CompositeCacheBackend(InterRequestCacheBackend):
    """
    Composite backend that combines inter_request with cache_dit.

    Inherits from InterRequestCacheBackend so all polymorphic cache hooks
    (short_circuit_requests / post_forward_store / ...) work transparently.
    Internally creates and manages a CacheDiTBackend for block-level caching
    within each denoising step.

    Coordination logic:
    - enable(): enables cache_dit on transformer first, then inter_request recorder
    - refresh(): forwards the FULL num_inference_steps to cache_dit. inter_request
      resumes by skipping the first N steps at the pipeline loop level (step
      hook); cache_dit operates inside the transformer and counts its own
      executed steps from zero, so it must keep the original step count.
    - All other methods (lookup, store, before_diffuse, after_diffuse) inherit
      from InterRequestCacheBackend unchanged — cache_dit operates
      automatically via transformer hooks.
    """

    def __init__(self, config: Any):
        super().__init__(config)
        self._cache_dit_backend = CacheDiTBackend(config)
        logger.info(
            "CompositeCacheBackend initialized: inter_request + cache_dit (Fn=%d, Bn=%d, warmup=%d)",
            config.Fn_compute_blocks,
            config.Bn_compute_blocks,
            config.max_warmup_steps,
        )

    def enable(self, pipeline: Any) -> None:
        # Enable cache_dit first (modifies transformer forward behavior)
        self._cache_dit_backend.enable(pipeline)
        logger.info("cache_dit enabled on transformer within composite backend")

        # Then enable inter_request (attaches StepLatentsRecorder)
        super().enable(pipeline)

    def refresh(
        self,
        pipeline: Any,
        num_inference_steps: int,
        verbose: bool = True,
    ) -> None:
        """Refresh cache_dit context with the full step count.

        inter_request resumes by skipping the first N steps at the pipeline
        loop level; cache_dit sees the original step indices inside the
        transformer, so the step count is intentionally NOT reduced by
        resume_from_step.
        """
        self._cache_dit_backend.refresh(pipeline, num_inference_steps, verbose)

    @property
    def cache_dit_backend(self) -> CacheDiTBackend:
        """Access the internal cache_dit backend for summary/debugging."""
        return self._cache_dit_backend
