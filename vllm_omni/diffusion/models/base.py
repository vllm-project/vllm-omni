# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import ClassVar

import torch
import torch.nn as nn

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.models.interface import SupportsComponentDiscovery
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch


class DiffusionPipelineBase(nn.Module, SupportsComponentDiscovery, ABC):
    """Abstract base class for all diffusion pipelines.

    Provides a unified interface for diffusion pipeline implementations,
    including component discovery for cache backends and a method to
    resolve default sampling parameters before forward() execution.

    Subclasses must declare class variables:
        supports_request_batch: whether the pipeline handles batched requests
        default_num_inference_steps: the pipeline's default denoising steps
        _dit_modules: attribute paths to DiT transformer(s)
        _encoder_modules: attribute paths to text encoder(s)
        _vae_modules: attribute paths to VAE(s)
    """

    supports_request_batch: ClassVar[bool]
    default_num_inference_steps: ClassVar[int]

    _dit_modules: ClassVar[list[str]]
    _encoder_modules: ClassVar[list[str]]
    _vae_modules: ClassVar[list[str]]

    def __init__(self, *, od_config: OmniDiffusionConfig | None = None, **kwargs: object) -> None:
        super().__init__()
        if od_config is not None:
            self.od_config = od_config

    @abstractmethod
    def forward(self, req: DiffusionRequestBatch) -> DiffusionOutput | list[DiffusionOutput]:
        """Run the diffusion pipeline forward pass.

        Args:
            req: Batch of diffusion requests to process.

        Returns:
            Single or batched diffusion outputs, depending on supports_request_batch.
        """
        ...

    @abstractmethod
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load pretrained weights into the pipeline.

        Args:
            weights: Iterator of (name, tensor) tuples to load.

        Returns:
            Set of weight names that were loaded.
        """
        ...

    def resolve_default_num_inference_steps(self) -> int | None:
        """Return the pipeline's default num_inference_steps.

        Called by the model runner before cache refresh so the cache
        backend can configure itself without waiting for forward().

        Returns:
            The default number of inference steps, or None if not set.
        """
        return getattr(self, "default_num_inference_steps", None)
