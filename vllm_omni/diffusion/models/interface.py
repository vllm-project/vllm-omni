# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
from __future__ import annotations

from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Literal,
    Protocol,
    runtime_checkable,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    import torch

    from vllm_omni.diffusion.cache.cachedit import CacheDiTBackend
    from vllm_omni.diffusion.data import DiffusionOutput
    from vllm_omni.diffusion.interaction.types import ChunkMediaSpec
    from vllm_omni.diffusion.worker.input_batch import InputBatch
    from vllm_omni.diffusion.worker.utils import StepRequestState


@runtime_checkable
class SupportImageInput(Protocol):
    support_image_input: ClassVar[bool] = True
    color_format: ClassVar[str] = "RGB"  # Default color format


@dataclass(frozen=True)
class ReferenceVideoDecodeSpec:
    max_frames: int | None = None
    keep: Literal["first", "last"] = "first"


@runtime_checkable
class SupportAudioInput(Protocol):
    support_audio_input: ClassVar[bool] = True


@runtime_checkable
class SupportAudioOutput(Protocol):
    support_audio_output: ClassVar[bool] = True


@runtime_checkable
class SupportsStepExecution(Protocol):
    """State-driven step-level execution protocol for diffusion pipelines.

    Pipelines should split request-level ``forward()`` into:
    ``prepare_encode()`` (one-time request setup), ``denoise_step()``
    (one denoise forward), ``step_scheduler()`` (one scheduler update),
    and ``post_decode()`` (final decode).
    """

    supports_step_execution: ClassVar[bool] = True

    def prepare_encode(self, state: StepRequestState, **kwargs: Any) -> StepRequestState:
        """Prepare request-level inputs and return initialized state."""
        ...

    def denoise_step(
        self, input_batch: InputBatch, *, states: Sequence[StepRequestState] | None = None, **kwargs: Any
    ) -> torch.Tensor | None:
        """Run one denoise forward on the runner-assembled batch."""
        ...

    def step_scheduler(self, state: StepRequestState, noise_pred: torch.Tensor, **kwargs: Any) -> None:
        """Run one scheduler step."""
        ...

    def post_decode(self, state: StepRequestState, **kwargs: Any) -> DiffusionOutput:
        """Decode output after denoise loop or at a partial chunk boundary."""
        ...


@runtime_checkable
class SupportsComponentDiscovery(Protocol):
    """Declares which submodules serve as pipeline components.

    Used by the framework to locate DiT, encoder, and VAE modules for
    CPU offload, HSDP sharding, and other operations that need to know
    the pipeline's internal structure.

    All attribute names support dotted paths for nested submodules
    (e.g. ``"pipe.transformer"``).

    Attributes:
        _dit_modules: Denoising submodules (on GPU during diffusion).
        _encoder_modules: Encoder submodules (offloaded during diffusion).
        _vae_modules: VAE(s) (always on GPU).
        _resident_modules: Extra modules pinned on GPU during layerwise
            offloading.  Optional, defaults to ``[]``.
    """

    _dit_modules: ClassVar[list[str]]
    _encoder_modules: ClassVar[list[str]]
    _vae_modules: ClassVar[list[str]]
    _resident_modules: ClassVar[list[str]] = []


def supports_step_execution(pipeline: object) -> bool:
    """Return whether `pipeline` implements :class:`SupportsStepExecution`."""

    return isinstance(pipeline, SupportsStepExecution)


@runtime_checkable
class SupportsInteractionApply(Protocol):
    """Optional protocol for pipelines with unified mid-generation, chunk-boundary hooks."""

    def peek_chunk_media(self, state: StepRequestState) -> ChunkMediaSpec:
        """Return the media timeline represented by the upcoming/current chunk.

        Useful when interaction handler needs interpolation/integration on a frame-by-frame basis,
        or for backpressure/pacing.
        """
        ...

    def apply_interaction_at_chunk_boundary(self, state: StepRequestState) -> None:
        """Advance queued interactions before the next generation chunk."""
        ...

    def prepare_next_chunk(self, state: StepRequestState) -> None:
        """Set up pipeline state for the next chunk after interaction apply.

        Default implementations is a no-op; model-specific pipelines override
        when chunk transitions require latent/history bookkeeping.
        """
        ...


def supports_interaction_apply(pipeline: object) -> bool:
    """Return whether ``pipeline`` implements :class:`SupportsInteractionApply`."""

    return isinstance(pipeline, SupportsInteractionApply)


@runtime_checkable
class SupportsRequestScopedCacheDiT(Protocol):
    """Optional protocol for pipelines that own Cache-DiT hook transitions."""

    def adopt_cache_dit_backend(self, backend: CacheDiTBackend) -> None:
        """Assume ownership of an enabled Cache-DiT backend."""
        ...

    def is_cache_dit_enabled(self) -> bool:
        """Return whether this pipeline currently has Cache-DiT installed."""
        ...


def adopt_request_scoped_cache_dit(pipeline: object, backend: CacheDiTBackend) -> bool:
    """Transfer an enabled Cache-DiT backend to an opted-in pipeline."""

    if not isinstance(pipeline, SupportsRequestScopedCacheDiT):
        return False
    pipeline.adopt_cache_dit_backend(backend)
    return True


def is_request_scoped_cache_dit_enabled(pipeline: object) -> bool:
    """Read Cache-DiT state from a pipeline that owns its lifecycle."""

    return isinstance(pipeline, SupportsRequestScopedCacheDiT) and pipeline.is_cache_dit_enabled()
