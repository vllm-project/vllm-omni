# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import (
    TYPE_CHECKING,
    ClassVar,
    Literal,
    Protocol,
    runtime_checkable,
)

if TYPE_CHECKING:
    import torch

    from vllm_omni.diffusion.data import DiffusionOutput
    from vllm_omni.diffusion.worker.input_batch import InputBatch
    from vllm_omni.diffusion.worker.utils import DiffusionRequestState


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


class StageBoundary(str, Enum):
    ENCODE_TO_DIT = "encode_to_dit"
    DIT_TO_DECODE = "dit_to_decode"


@dataclass
class StagePayload:
    request_id: str
    boundary: StageBoundary
    scalar_fields: dict[str, object]
    tensor_fields: dict[str, torch.Tensor]
    private_scalar_fields: dict[str, object]
    private_tensor_fields: dict[str, torch.Tensor]
    payload_version: int = 1


@runtime_checkable
class DiffusionV2Atoms(Protocol):
    """State-based diffusion atoms shared by request mode and step mode."""

    supports_step_execution: ClassVar[bool] = True

    def init_state(self, state: DiffusionRequestState) -> DiffusionRequestState:
        """Initialize pipeline-private fields on a newly created request state."""
        ...

    def check_inputs(self, state: DiffusionRequestState) -> DiffusionRequestState:
        """Validate request inputs before model work begins."""
        ...

    def encode(self, state: DiffusionRequestState) -> DiffusionRequestState:
        """Run text/input encoders and populate encoded prompt fields."""
        ...

    def prepare(self, state: DiffusionRequestState) -> DiffusionRequestState:
        """Prepare model-specific denoise state after encode."""
        ...

    def diffuse(self, state: DiffusionRequestState) -> DiffusionRequestState:
        """Run the full diffusion loop for request-mode/golden-path execution."""
        ...

    def decode(self, state: DiffusionRequestState) -> DiffusionRequestState:
        """Decode raw latent state into the model output representation."""
        ...

    def postprocess(self, state: DiffusionRequestState) -> DiffusionOutput:
        """Apply model-specific output post-processing and return final output."""
        ...

    def pack_stage_state(
        self,
        state: DiffusionRequestState,
        boundary: StageBoundary,
    ) -> StagePayload:
        """Pack state for a stage boundary without exposing model-private schema to the runner."""
        ...

    def unpack_stage_state(
        self,
        payload: StagePayload,
        state: DiffusionRequestState,
    ) -> DiffusionRequestState:
        """Apply a received stage payload to an existing request state."""
        ...

    def build_step_batch(
        self,
        states: list[DiffusionRequestState],
        *,
        cached_batch: InputBatch | None = None,
    ) -> InputBatch:
        """Build the runner-visible step batch for one scheduler tick."""
        ...

    def build_step_attention_metadata(
        self,
        input_batch: InputBatch,
    ) -> object | None:
        """Build optional forward-context attention metadata for the step batch."""
        ...

    def denoise_step(
        self,
        input_batch: InputBatch,
    ) -> torch.Tensor | None:
        """Run one DiT denoise step on the runner-assembled batch."""
        ...

    def step_scheduler(
        self,
        state: DiffusionRequestState,
        noise_pred: torch.Tensor,
    ) -> DiffusionRequestState:
        """Apply one scheduler step to a request-local state."""
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
    """Return whether `pipeline` implements the v2 step atom contract."""

    return getattr(pipeline, "supports_step_execution", False) is True and isinstance(
        pipeline,
        DiffusionV2Atoms,
    )
