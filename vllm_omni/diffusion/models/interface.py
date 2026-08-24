# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Literal,
    ParamSpec,
    Protocol,
    TypeVar,
    runtime_checkable,
)

_BORROWED_WEIGHT_CONSUMER_MARKER = "_consumes_borrowed_weight_tensors"
_P = ParamSpec("_P")
_R = TypeVar("_R")


def consumes_borrowed_weight_tensors(method: Callable[_P, _R]) -> Callable[_P, _R]:
    """Declare synchronous consumption of each ``load_weights`` tensor.

    The decorated method must finish every read or copy from the current
    checkpoint tensor, retain no aliases, and enqueue no asynchronous use
    before requesting the next iterator item.

    This is a method marker rather than a class capability so a subclass that
    overrides ``load_weights`` must explicitly re-declare the contract.
    """
    setattr(method, _BORROWED_WEIGHT_CONSUMER_MARKER, True)
    return method


def consumes_borrowed_weights_synchronously(model: object) -> bool:
    """Return whether the effective ``load_weights`` method declares the contract."""
    method = getattr(type(model), "load_weights", None)
    return bool(getattr(method, _BORROWED_WEIGHT_CONSUMER_MARKER, False))


if TYPE_CHECKING:
    from collections.abc import Sequence

    import torch

    from vllm_omni.diffusion.cache.cachedit import CacheDiTBackend
    from vllm_omni.diffusion.data import DiffusionOutput
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
class SupportsPromptUpdate(Protocol):
    """Optional protocol for pipelines that support midway prompt updates.

    Pipelines typically implement this via
    :class:`~vllm_omni.diffusion.prompt_update.PromptUpdateMixin`.
    """

    supports_prompt_update: ClassVar[bool] = True

    def prepare_prompt_update(
        self,
        state: StepRequestState,
        prompt: str,
        event_id: str,
        transition_chunks: int | None = None,
    ) -> None:
        """Encode and queue a prompt update on request-local state."""
        ...


def supports_prompt_update(pipeline: object) -> bool:
    """Return whether ``pipeline`` implements :class:`SupportsPromptUpdate`."""

    return isinstance(pipeline, SupportsPromptUpdate)


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
