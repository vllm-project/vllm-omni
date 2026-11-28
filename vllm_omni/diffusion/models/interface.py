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
    Protocol,
    TypeGuard,
    runtime_checkable,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    import torch

    from vllm_omni.diffusion.cache.cachedit import CacheDiTBackend
    from vllm_omni.diffusion.data import DiffusionOutput
    from vllm_omni.diffusion.interaction.types import ChunkMediaSpec
    from vllm_omni.diffusion.worker.input_batch import InputBatch
    from vllm_omni.diffusion.worker.utils import StepRequestState, StepWait


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


DecodedChunkConsumer = Callable[["torch.Tensor"], None]


@runtime_checkable
class SupportsChunkedVAEDecode(Protocol):
    """Optional capability for VAEs that stream decoded temporal chunks.

    Implementations must drain decoding before surfacing callback errors. In a
    distributed VAE, every rank invokes the method so collectives stay in
    lockstep, while ``on_chunk`` is called only on the rank that owns output.

    ``chunk_value_range`` is the closed interval the delivered floats occupy,
    which differs per checkpoint: a Wan VAE emits ``(-1.0, 1.0)`` while a
    MiniMax-H3 VAE reverts through its processor to ``(0.0, 1.0)``. A consumer
    cannot quantize chunks to pixels without it, so it is part of the
    capability rather than knowledge each consumer hard-codes per model.
    """

    chunk_value_range: ClassVar[tuple[float, float]]

    def decode_with_chunks(
        self,
        z: torch.Tensor,
        *,
        on_chunk: DecodedChunkConsumer,
    ) -> None:
        """Decode ``z`` and synchronously deliver committed chunks in order."""
        ...


def supports_chunked_vae_decode(vae: object) -> TypeGuard[SupportsChunkedVAEDecode]:
    """Return whether ``vae`` exposes the optional chunked decode capability."""

    return isinstance(vae, SupportsChunkedVAEDecode)


@runtime_checkable
class SupportsStepExecution(Protocol):
    """State-driven step-level execution protocol for diffusion pipelines.

    Pipelines should split request-level ``forward()`` into:
    ``prepare_encode()`` (one-time request setup), ``denoise_step()``
    (one denoise forward), ``step_scheduler()`` (one scheduler update),
    and ``post_decode()`` (final decode).

    A pipeline may additionally set the optional class attribute
    ``supports_chunk_step_grouping = True`` to declare that its request state
    may advance through every denoise step of its current chunk without a
    serving-scheduler cycle in between (a capability, not a policy: the runner
    that owns the scheduling decision chooses whether to group steps). It is
    optional so it does not become part of the runtime protocol check.

    Likewise, a pipeline may set the optional class attribute
    ``supports_mixfusion = True`` to declare that it can denoise a batch of
    mixed-resolution requests (ragged/packed latents), which lets the
    scheduler relax height/width out of the batch key when MixFusion is
    enabled. It is read via ``getattr`` and is likewise not part of the
    runtime protocol check.
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
class SupportsSkipLayerGuidance(Protocol):
    """Transformer with request-local identity skips for complete DiT blocks.

    Layer IDs are defined by the model. Its forward accepts ``skip_layers``
    without changing weights, hooks, or other requests' execution state.
    """

    supports_skip_layer_guidance: ClassVar[bool]

    def validate_skip_layers(self, skip_layers: tuple[int, ...]) -> None:
        """Reject layer IDs that cannot be skipped by this transformer."""
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


@runtime_checkable
class SupportsSamplingStep(Protocol):
    """Pipelines whose denoise_step also updates latents and step_index.

    denoise_step owns every model evaluation and commits one complete algorithm
    step per successful request. Return every request ID mapped to None on
    success or its exception on failure. A failed request must not advance.
    """

    supports_step_execution: ClassVar[bool] = True
    denoise_step_updates_state: ClassVar[bool] = True

    def prepare_encode(self, state: StepRequestState, **kwargs: Any) -> StepRequestState: ...

    def denoise_step(
        self, input_batch: InputBatch, *, states: Sequence[StepRequestState] | None = None
    ) -> dict[str, Exception | None]: ...

    def post_decode(self, state: StepRequestState, **kwargs: Any) -> DiffusionOutput: ...


def supports_step_execution(pipeline: object) -> bool:
    """Return whether the pipeline implements either supported step protocol."""

    return isinstance(pipeline, SupportsStepExecution) or (
        isinstance(pipeline, SupportsSamplingStep) and pipeline.denoise_step_updates_state
    )


@runtime_checkable
class SupportsStepWait(Protocol):
    """Optional capability: a step pipeline that can suspend and resume itself.

    A pipeline that returns a step boundary it does not want to cross yet -- H3
    holding at a request boundary until the client chooses the next prompt --
    declares it here instead of polling inside a step. The scheduler owns the
    deadline: ``step_wait`` only states whether to suspend and for how long, and
    the worker re-enters the pipeline through ``resume_step`` once the scheduler
    decided why (a control arrived, or the deadline expired).

    Optional on purpose: it is read through ``supports_step_wait``, so it does
    not become part of the required ``SupportsStepExecution`` contract.
    """

    def step_wait(self, state: StepRequestState) -> StepWait | None:
        """Return the wait this chunk boundary asks for, or ``None`` to continue."""
        ...

    def resume_step(self, state: StepRequestState, reason: Literal["interaction", "timeout"]) -> None:
        """Continue a suspended request with the scheduler's reason for waking it."""
        ...


# One-release compatibility: the AR-Diffusion runner originally spelled these
# hooks after its own package. Models use the neutral names above; the worker
# still accepts the old spelling so existing pipelines keep working.
_STEP_WAIT_ALIASES = {"step_wait": "ar_diffusion_step_wait", "resume_step": "ar_diffusion_resume_step"}


def step_wait_hook(pipeline: object) -> Callable[[StepRequestState], StepWait | None] | None:
    """Return the pipeline's wait hook, preferring the neutral name."""

    hook = getattr(pipeline, "step_wait", None)
    if callable(hook):
        return hook
    legacy = getattr(pipeline, _STEP_WAIT_ALIASES["step_wait"], None)
    return legacy if callable(legacy) else None


def step_resume_hook(pipeline: object) -> Callable[..., None] | None:
    """Return the pipeline's resume hook, preferring the neutral name.

    The hook takes ``(state, reason)``, where ``reason`` is either
    ``"interaction"`` or ``"timeout"``, as declared by
    :class:`SupportsStepWait`.
    """

    hook = getattr(pipeline, "resume_step", None)
    if callable(hook):
        return hook
    legacy = getattr(pipeline, _STEP_WAIT_ALIASES["resume_step"], None)
    return legacy if callable(legacy) else None


def supports_step_wait(pipeline: object) -> bool:
    """Return whether the pipeline declares both wait/resume hooks."""

    return step_wait_hook(pipeline) is not None and step_resume_hook(pipeline) is not None


# --- Generic role-based component loading for diffusion disaggregation ---------

# Which component groups (as declared by ``SupportsComponentDiscovery``) each
# diffusion stage role needs to load. ``encoder`` = text/image encoders,
# ``dit`` = denoising transformer(s), ``vae`` = VAE. A stage skips loading every
# group not listed for its role, which is what frees memory on disaggregated
# workers (e.g. an ENCODE stage drops the DiT + VAE). This mapping is
# model-agnostic: any pipeline that implements ``SupportsComponentDiscovery``
# becomes role-splittable without model-specific ``if`` branches.
_ROLE_COMPONENT_GROUPS: dict[str, frozenset[str]] = {
    "full": frozenset({"encoder", "dit", "vae"}),
    "encode": frozenset({"encoder"}),
    "denoise": frozenset({"dit"}),
    "denoise_decode": frozenset({"dit", "vae"}),
    "decode": frozenset({"vae"}),
}


def stage_component_groups(role: str) -> frozenset[str]:
    """Return the component groups (``encoder``/``dit``/``vae``) a role loads."""
    return _ROLE_COMPONENT_GROUPS.get(role, _ROLE_COMPONENT_GROUPS["full"])


def role_loads_component(role: str, group: Literal["encoder", "dit", "vae"]) -> bool:
    """Whether a diffusion stage ``role`` should load the given component group.

    Use inside a pipeline ``__init__`` to decide, model-agnostically, whether to
    instantiate the encoder / DiT / VAE for the current stage role::

        if role_loads_component(role, "dit"):
            self.transformer = ...
        if role_loads_component(role, "vae"):
            self.vae = ...
    """
    return group in stage_component_groups(role)


@runtime_checkable
class SupportsInteractionApply(Protocol):
    """Optional protocol for pipelines with unified mid-generation, chunk-boundary hooks."""

    def peek_chunk_media(self, state: StepRequestState) -> ChunkMediaSpec:
        """Return the media timeline and latent frame count for the upcoming/current chunk.

        Useful when interaction handler needs interpolation/integration on a frame-by-frame basis,
        and/or for backpressure/pacing.
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
