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
class SupportsModuleOffload(Protocol):
    """Declares which submodules participate in sequential CPU offload.

    The offload system uses mutual exclusion: when one group runs,
    the other is moved to CPU.  Pipelines must declare both groups
    because only the pipeline knows its own architecture.

    ``_dit_modules``: attribute names of denoising submodules (kept
    on GPU during the diffusion loop).

    ``_encoder_modules``: attribute names of encoder/vision
    submodules (offloaded to CPU during the diffusion loop).

    ``_vae_modules``: attribute names of VAE(s) (always kept on GPU,
    not part of the mutual exclusion hooks).

    ``_resident_modules``: attribute names of small submodules that
    must stay on GPU during layer-wise offloading (e.g. embedders,
    connectors).  Optional — defaults to ``[]``.

    All attribute names support dotted paths (e.g.
    ``"bagel.time_embedder"``) for nested submodules.

    Example::

        class MyPipeline(nn.Module, SupportsModuleOffload):
            _dit_modules: ClassVar[list[str]] = ["transformer"]
            _encoder_modules: ClassVar[list[str]] = ["text_encoder", "vit"]
            _vae_modules: ClassVar[list[str]] = ["vae"]
            _resident_modules: ClassVar[list[str]] = ["time_embedder"]
    """

    _dit_modules: ClassVar[list[str]]
    _encoder_modules: ClassVar[list[str]]
    _vae_modules: ClassVar[list[str]]
    _resident_modules: ClassVar[list[str]] = []


def supports_step_execution(pipeline: object) -> bool:
    """Return whether `pipeline` implements :class:`SupportsStepExecution`."""

    return isinstance(pipeline, SupportsStepExecution)
