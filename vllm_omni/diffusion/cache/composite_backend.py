# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Composite cache backend combining inter_request and cache_dit.

This allows both cross-request latent reuse (inter_request) and intra-request
block-level caching (cache_dit) to work together. When inter_request resumes
from step N, cache_dit operates on the remaining (total - N) steps.

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

from vllm_omni.diffusion.cache.inter_request.backend import InterRequestCacheBackend

logger = logging.getLogger(__name__)


class CompositeCacheBackend(InterRequestCacheBackend):
    """
    Composite backend that combines inter_request with cache_dit.

    Inherits from InterRequestCacheBackend so all isinstance() checks in the
    runner work transparently. Internally creates and manages a CacheDiTBackend
    for block-level caching within each denoising step.

    Coordination logic:
    - enable(): enables cache_dit on transformer first, then inter_request recorder
    - refresh(): when resume_from_step > 0, tells cache_dit to use (total - resume) steps
    - All other methods (lookup, store, before_diffuse, after_diffuse) inherit from
      InterRequestCacheBackend unchanged - cache_dit operates automatically via
      transformer hooks.
    """

    def __init__(self, config: Any):
        super().__init__(config)
        from vllm_omni.diffusion.cache.cache_dit_backend import CacheDiTBackend
        self._cache_dit_backend = CacheDiTBackend(config)
        logger.info(
            "CompositeCacheBackend initialized: inter_request + cache_dit "
            "(Fn=%d, Bn=%d, warmup=%d)",
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
        resume_from_step: int = 0,
    ) -> None:
        """Refresh cache_dit context.

        When inter_request resumes from step N, the pipeline's denoise loop
        still iterates over ``num_inference_steps`` steps — inter_request
        simply skips the first N via the step hook. cache_dit operates inside
        the transformer and sees the original step indices, so it must be
        configured with the full ``num_inference_steps`` (not reduced).
        Reducing the step count here would cause cache_dit to re-run its
        warmup, losing the block-level acceleration for steps N..N+warmup.
        """
        self._cache_dit_backend.refresh(pipeline, num_inference_steps, verbose)

    @property
    def cache_dit_backend(self):
        """Access the internal cache_dit backend for summary/debugging."""
        return self._cache_dit_backend
