# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Protocol,
    runtime_checkable,
)

if TYPE_CHECKING:
    import torch

    from vllm_omni.diffusion.data import DiffusionOutput
    from vllm_omni.diffusion.worker.utils import DiffusionRequestState


@runtime_checkable
class SupportImageInput(Protocol):
    support_image_input: ClassVar[bool] = True
    color_format: ClassVar[str] = "RGB"  # Default color format


@runtime_checkable
class SupportAudioInput(Protocol):
    support_audio_input: ClassVar[bool] = True


@runtime_checkable
class SupportCameraPosInput(Protocol):
    support_camera_pos_input: ClassVar[bool] = True


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

    def prepare_encode(self, state: DiffusionRequestState, **kwargs: Any) -> DiffusionRequestState:
        """Prepare request-level inputs and return initialized state."""

    def denoise_step(self, state: DiffusionRequestState, **kwargs: Any) -> torch.Tensor | None:
        """Run one denoise step."""

    def step_scheduler(self, state: DiffusionRequestState, noise_pred: torch.Tensor, **kwargs: Any) -> None:
        """Run one scheduler step."""

    def post_decode(self, state: DiffusionRequestState, **kwargs: Any) -> DiffusionOutput:
        """Decode output after denoise loop."""


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
class SupportsMicroStepExecution(Protocol):
    """Temporal-PP micro-step execution protocol.

    Per-micro-step hooks used by ``DiffusionModelRunner.execute_micro_step``:
    ``prepare_first_chunk`` runs chunk 0 to completion + seeds slots, ``prepare_chunks``
    rolls the ladder + admits a chunk, ``denoise_step`` + ``step_scheduler`` run one
    denoise, ``decode_chunks`` decodes the finished chunk, and
    ``set_pp_recv_dict_buffers`` / ``prefetch_tensors`` manage the PP comms.
    """

    supports_micro_step_execution: ClassVar[bool] = True

    def prepare_encode(self, state: DiffusionRequestState, **kwargs: Any) -> DiffusionRequestState:
        """Prepare request-level inputs and return initialized state."""

    def prepare_first_chunk(self, state: DiffusionRequestState, **kwargs: Any) -> DiffusionOutput | None:
        """Denoise chunk 0 to completion, seed all KV slots, and decode it."""

    def prepare_chunks(self, state: DiffusionRequestState, **kwargs: Any) -> None:
        """Roll the ladder one slot and admit the new chunk into ``state.latents``."""

    def denoise_step(self, state: DiffusionRequestState, **kwargs: Any) -> torch.Tensor | None:
        """Run one denoise step."""

    def step_scheduler(self, state: DiffusionRequestState, noise_pred: torch.Tensor, **kwargs: Any) -> None:
        """Run one scheduler step."""

    def decode_chunks(self, state: DiffusionRequestState, **kwargs: Any) -> DiffusionOutput | None:
        """Decode finished chunks; return the merged output once all are decoded."""

    def set_pp_recv_dict_buffers(self, state: DiffusionRequestState, **kwargs: Any) -> None:
        """Pre-register PP dict recv buffers and schema cache for this request."""

    def prefetch_tensors(self, state: DiffusionRequestState, **kwargs: Any) -> None:
        """Pre-post the next-step recv."""


def supports_micro_step_execution(pipeline: object) -> bool:
    """Return whether `pipeline` implements :class:`SupportsMicroStepExecution`."""

    return isinstance(pipeline, SupportsMicroStepExecution)
