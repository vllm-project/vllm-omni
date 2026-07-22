# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Diffusion pipeline capability protocols and their feature-probe helpers.

Home of the ``runtime_checkable`` protocols the diffusion runtime uses to decide
what a loaded pipeline can do, plus the ``supports_*`` helpers that isinstance-probe
them:

* input/output modality markers (image/audio);
* :class:`DiffusionV2Atoms` — the PR #4948 state-based atom contract shared by
  request mode, step mode, and RFC #4590 disaggregated stages (``init_state`` /
  ``check_inputs`` / ``encode`` / ``prepare`` / ``diffuse`` / ``decode`` /
  ``postprocess`` plus the per-step atoms ``build_step_batch`` /
  ``denoise_step`` / ``step_scheduler`` and the ``pack_stage_state`` /
  ``unpack_stage_state`` payload hooks). ``supports_step_execution`` gates the
  step runner; the RFC #4590 ``required_components_for_stage`` extension and the
  ``supports_disaggregated_execution`` flag gate disaggregated stage execution.
  A whole-request pipeline (e.g. DreamZero's dual video+action DiT loop with
  model-owned AR-Diffusion KV) opts out of step mode and overrides ``diffuse``
  with its own loop while still satisfying the atom surface;
* :class:`SupportsComponentDiscovery` (submodule locations for offload/HSDP).

:class:`StagePayload` is the transport envelope handed from one diffusion stage
to the next; its transport safety (tensor sanitization, non-transportable-value
rejection) is delegated to the torch-free helpers in
:mod:`vllm_omni.diffusion.stage_payload`. Kept torch-free at module scope (torch
imports are TYPE_CHECKING-only) so capability checks stay importable without the
model runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Literal,
    Protocol,
    runtime_checkable,
)

# NOTE: the transport helpers live in the torch-free leaf
# ``vllm_omni.diffusion.stage_payload``. They are imported LAZILY inside the
# StagePayload methods (via ``_transport()``), NOT at module scope: a module-scope
# ``from vllm_omni.diffusion.stage_payload import ...`` is an absolute import that
# would trigger ``vllm_omni/__init__`` (which imports torch), breaking the
# torch-free by-path load the disaggregated test suite relies on. The schema
# version is duplicated as a module constant below and asserted equal to the leaf's
# in tests, so this module needs no import-time dependency on the leaf.
#: Bump in lockstep with ``stage_payload.DIFFUSION_STAGE_PAYLOAD_SCHEMA_VERSION``.
#: v2 (RFC #4590 Part A) adds DreamZero session-progress routing scalars.
STAGE_PAYLOAD_SCHEMA_VERSION = 2

if TYPE_CHECKING:
    import torch

    from vllm_omni.diffusion.data import DiffusionOutput
    from vllm_omni.diffusion.stage_roles import StageComponentSpec
    from vllm_omni.diffusion.worker.input_batch import InputBatch
    from vllm_omni.diffusion.worker.utils import DiffusionRequestState


def _transport():
    """Lazily resolve the torch-free transport helpers from the stage_payload leaf.

    Deferred to call time so importing this capability module never triggers the
    ``vllm_omni`` package ``__init__`` (torch). Checks ``sys.modules`` first so a
    by-path, torch-free load (the disaggregated tests, which register the leaf
    under its real dotted name) resolves without importing the package; falls
    back to a normal import at real runtime.
    """
    import sys

    mod = sys.modules.get("vllm_omni.diffusion.stage_payload")
    if mod is not None:
        return mod
    from vllm_omni.diffusion import stage_payload

    return stage_payload


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


# ---------------------------------------------------------------------------
# Legacy step-level execution protocol (retained for pipelines not yet migrated
# to the #4948 DiffusionV2Atoms surface: helios / hunyuan_image3 / qwen_image /
# diffusers_adapter). New/migrated pipelines implement DiffusionV2Atoms below.
# ---------------------------------------------------------------------------


@runtime_checkable
class SupportsStepExecution(Protocol):
    """State-driven step-level execution protocol for legacy diffusion pipelines.

    Pipelines that have not migrated to :class:`DiffusionV2Atoms` split
    request-level ``forward()`` into: ``prepare_encode()`` (one-time request
    setup), ``denoise_step()`` (one denoise forward), ``step_scheduler()`` (one
    scheduler update), and ``post_decode()`` (final decode).
    """

    supports_step_execution: ClassVar[bool] = True

    def prepare_encode(self, state: DiffusionRequestState, **kwargs: Any) -> DiffusionRequestState:
        """Prepare request-level inputs and return initialized state."""
        ...

    def denoise_step(self, input_batch: InputBatch, **kwargs: Any) -> torch.Tensor | None:
        """Run one denoise forward on the runner-assembled batch."""
        ...

    def step_scheduler(self, state: DiffusionRequestState, noise_pred: torch.Tensor, **kwargs: Any) -> None:
        """Run one scheduler step."""
        ...

    def post_decode(self, state: DiffusionRequestState, **kwargs: Any) -> DiffusionOutput:
        """Decode output after denoise loop or at a partial chunk boundary."""
        ...


# ---------------------------------------------------------------------------
# State-based diffusion atoms + transport envelope (PR #4948 / RFC #4590)
# ---------------------------------------------------------------------------


class StageBoundary(str, Enum):
    ENCODE_TO_DIT = "encode_to_dit"
    DIT_TO_DECODE = "dit_to_decode"


def boundary_from_roles(source_stage: str, target_stage: str) -> StageBoundary:
    """Map a disaggregated (source_stage, target_stage) transition to a boundary.

    ``encode -> denoise`` is :attr:`StageBoundary.ENCODE_TO_DIT`;
    ``denoise -> decode`` is :attr:`StageBoundary.DIT_TO_DECODE`. Torch-free;
    the role strings are the ones in :mod:`vllm_omni.diffusion.stage_roles`.
    """
    if source_stage == "encode" and target_stage == "denoise":
        return StageBoundary.ENCODE_TO_DIT
    if source_stage == "denoise" and target_stage == "decode":
        return StageBoundary.DIT_TO_DECODE
    raise _transport().StagePayloadError(f"No stage boundary for transition {source_stage!r}->{target_stage!r}.")


@dataclass
class StagePayload:
    """Transport envelope handed from one diffusion stage to the next.

    The #4948 four-dict envelope, carrying the *data* one stage produces for the
    next while a mutable runner-local ``DiffusionRequestState`` never crosses a
    process boundary. Public ``*_fields`` are runner-visible; ``private_*_fields``
    are model-private and only the owning pipeline interprets them. Both tensor
    dicts are sanitized to host memory at export; both scalar dicts are validated
    against the transportable allow-list. Transport safety is delegated to the
    torch-free helpers in :mod:`vllm_omni.diffusion.stage_payload`.
    """

    request_id: str
    boundary: StageBoundary
    scalar_fields: dict[str, object] = field(default_factory=dict)
    tensor_fields: dict[str, torch.Tensor] = field(default_factory=dict)
    private_scalar_fields: dict[str, object] = field(default_factory=dict)
    private_tensor_fields: dict[str, torch.Tensor] = field(default_factory=dict)
    payload_version: int = STAGE_PAYLOAD_SCHEMA_VERSION

    # -- construction --------------------------------------------------------

    @classmethod
    def create(
        cls,
        *,
        request_id: str,
        boundary: StageBoundary,
        scalar_fields: dict[str, object] | None = None,
        tensor_fields: dict[str, torch.Tensor] | None = None,
        private_scalar_fields: dict[str, object] | None = None,
        private_tensor_fields: dict[str, torch.Tensor] | None = None,
        sanitize: bool = True,
        validate: bool = True,
    ) -> StagePayload:
        """Build a payload, sanitizing both tensor dicts and validating.

        ``sanitize=True`` (default) routes every tensor through
        :func:`~vllm_omni.diffusion.stage_payload.sanitize_transport_tensor` so
        callers don't scatter ``.detach().cpu()`` through model code.
        ``validate=True`` runs the full transportability check before returning.
        """

        transport = _transport()

        def _prep(tensors: dict[str, torch.Tensor] | None, label: str) -> dict[str, torch.Tensor]:
            prepared: dict[str, torch.Tensor] = {}
            for name, tensor in (tensors or {}).items():
                if not transport._is_tensor(tensor):
                    raise transport.NonTransportableValueError(
                        f"{label}[{name!r}] is {type(tensor).__name__}, expected a torch.Tensor. "
                        "Small non-tensor values belong in a scalar dict."
                    )
                prepared[name] = transport.sanitize_transport_tensor(tensor) if sanitize else tensor
            return prepared

        payload = cls(
            request_id=request_id,
            boundary=boundary,
            scalar_fields=dict(scalar_fields or {}),
            tensor_fields=_prep(tensor_fields, "tensor_fields"),
            private_scalar_fields=dict(private_scalar_fields or {}),
            private_tensor_fields=_prep(private_tensor_fields, "private_tensor_fields"),
        )
        if validate:
            payload.validate()
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StagePayload:
        """Rehydrate a payload flattened to a plain dict by cross-process transport.

        Inter-stage IPC (msgpack) does not preserve dataclass identity, so a
        payload sent through ``custom_output`` arrives on the receiving stage as
        a plain ``dict``. The runner's ``_extract_incoming_payload`` and the
        generic transition processor reconstruct the real type via this method
        before validation. ``boundary`` round-trips as its string value.

        Raises :class:`StagePayloadError` (not ``KeyError`` / ``ValueError``) on a
        malformed or partial dict so a corrupted wire payload surfaces as the same
        typed transport error the rest of the stage path raises, instead of an
        opaque low-level exception.
        """
        error = _transport().StagePayloadError
        if not isinstance(data, dict):
            raise error(f"StagePayload.from_dict expected a dict, got {type(data).__name__}.")
        try:
            request_id = data["request_id"]
            raw_boundary = data["boundary"]
        except KeyError as exc:
            raise error(f"StagePayload dict is missing required key {exc.args[0]!r}.") from exc
        try:
            boundary = StageBoundary(raw_boundary)
        except ValueError as exc:
            raise error(f"StagePayload dict has unknown boundary {raw_boundary!r}.") from exc
        return cls(
            request_id=request_id,
            boundary=boundary,
            scalar_fields=dict(data.get("scalar_fields") or {}),
            tensor_fields=dict(data.get("tensor_fields") or {}),
            private_scalar_fields=dict(data.get("private_scalar_fields") or {}),
            private_tensor_fields=dict(data.get("private_tensor_fields") or {}),
            payload_version=data.get("payload_version", STAGE_PAYLOAD_SCHEMA_VERSION),
        )

    def to_dict(self) -> dict[str, Any]:
        """Flatten to a plain, transportable dict (``boundary`` as its str value)."""
        return {
            "request_id": self.request_id,
            "boundary": self.boundary.value,
            "scalar_fields": dict(self.scalar_fields),
            "tensor_fields": dict(self.tensor_fields),
            "private_scalar_fields": dict(self.private_scalar_fields),
            "private_tensor_fields": dict(self.private_tensor_fields),
            "payload_version": self.payload_version,
        }

    # -- validation ----------------------------------------------------------

    def validate(self) -> None:
        """Raise :class:`StagePayloadError` if the envelope is not transportable."""
        transport = _transport()
        error = transport.StagePayloadError
        if not isinstance(self.request_id, str) or not self.request_id:
            raise error("StagePayload.request_id must be a non-empty string.")
        if not isinstance(self.boundary, StageBoundary):
            raise error(f"StagePayload.boundary must be a StageBoundary, got {self.boundary!r}.")
        if self.payload_version != STAGE_PAYLOAD_SCHEMA_VERSION:
            raise error(
                f"Unsupported StagePayload payload_version {self.payload_version}; "
                f"this build understands version {STAGE_PAYLOAD_SCHEMA_VERSION}."
            )
        for label, tensors in (
            ("tensor_fields", self.tensor_fields),
            ("private_tensor_fields", self.private_tensor_fields),
        ):
            if not isinstance(tensors, dict):
                raise error(f"StagePayload.{label} must be a dict[str, Tensor].")
            for name, tensor in tensors.items():
                if not isinstance(name, str):
                    raise error(f"{label} key {name!r} must be a string.")
                if not transport._is_tensor(tensor):
                    raise transport.NonTransportableValueError(
                        f"{label}[{name!r}] is {type(tensor).__name__}, expected a torch.Tensor."
                    )
        for label, scalars in (
            ("scalar_fields", self.scalar_fields),
            ("private_scalar_fields", self.private_scalar_fields),
        ):
            if not isinstance(scalars, dict):
                raise error(f"StagePayload.{label} must be a dict.")
            for key, value in scalars.items():
                if not isinstance(key, str):
                    raise error(f"{label} key {key!r} must be a string.")
                transport.validate_transport_scalar(value, path=f"{label}[{key!r}]")

    def sanitize(self) -> StagePayload:
        """Return a copy with both tensor dicts moved to host memory."""
        return StagePayload.create(
            request_id=self.request_id,
            boundary=self.boundary,
            scalar_fields=self.scalar_fields,
            tensor_fields=self.tensor_fields,
            private_scalar_fields=self.private_scalar_fields,
            private_tensor_fields=self.private_tensor_fields,
            sanitize=True,
            validate=False,
        )

    # -- convenience ---------------------------------------------------------

    def require_tensors(self, *names: str, private: bool = True) -> None:
        """Raise if any of ``names`` is missing from the (private) tensor dict."""
        source = self.private_tensor_fields if private else self.tensor_fields
        missing = [n for n in names if n not in source]
        if missing:
            which = "private_tensor_fields" if private else "tensor_fields"
            raise _transport().StagePayloadError(
                f"Payload {self.boundary.value!r} for request {self.request_id!r} is missing "
                f"required {which}: {missing}. Present: {sorted(source)}."
            )

    def expect_boundary(self, boundary: StageBoundary) -> None:
        """Raise if the payload's boundary does not match ``boundary``."""
        if self.boundary != boundary:
            raise _transport().StagePayloadError(
                f"Payload for request {self.request_id!r} has boundary {self.boundary!r}, expected {boundary!r}."
            )

    def summary(self) -> str:
        """One-line, tensor-content-free description for debug logging."""

        def _desc(tensors: dict[str, torch.Tensor]) -> str:
            return ", ".join(
                f"{name}{tuple(t.shape)}:{str(t.dtype).replace('torch.', '')}" for name, t in tensors.items()
            )

        return (
            f"StagePayload(req={self.request_id} boundary={self.boundary.value} "
            f"v{self.payload_version} tensors=[{_desc(self.tensor_fields)}] "
            f"private_tensors=[{_desc(self.private_tensor_fields)}] "
            f"scalar_keys={sorted(self.scalar_fields)} private_scalar_keys={sorted(self.private_scalar_fields)})"
        )


@runtime_checkable
class DiffusionV2Atoms(Protocol):
    """State-based diffusion atoms shared by request mode and step mode (PR #4948).

    A pipeline splits request-level ``forward()`` into small state-in/state-out
    atoms that the runner drives:

    * whole-request / golden path: ``init_state`` -> ``check_inputs`` ->
      ``encode`` -> ``prepare`` -> ``diffuse`` -> ``decode`` -> ``postprocess``;
    * step / continuous-batching path: ``build_step_batch`` ->
      ``build_step_attention_metadata`` -> ``denoise_step`` -> ``step_scheduler``
      (the default ``diffuse`` in :class:`DiffusionV2AtomDefaultsMixin` composes
      these into the whole-request loop);
    * disaggregated stages (RFC #4590): the runner drives encode on one worker,
      ``diffuse`` on another, and ``decode`` / ``postprocess`` on a third, moving
      the typed :class:`StagePayload` produced by ``pack_stage_state`` across the
      process boundary and re-hydrating it with ``unpack_stage_state``;
      ``required_components_for_stage`` lets each stage build only the components
      its role owns.

    ``supports_step_execution`` gates whether the step runner may drive the model
    through the per-step atoms. A whole-request pipeline (e.g. DreamZero's dual
    video+action DiT loop with model-owned AR-Diffusion KV) sets it ``False`` and
    overrides ``diffuse`` with its own loop while still satisfying the atom
    surface for the disaggregated / golden path.
    """

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

    @classmethod
    def required_components_for_stage(cls, model_stage: str) -> StageComponentSpec:
        """Declare which components a given disaggregated stage role must build.

        RFC #4590 extension to the #4948 atom surface. Queried before module
        construction so a stage process loads only the components its role owns
        (see :class:`~vllm_omni.diffusion.stage_roles.StageComponentSpec`).
        Monolithic pipelines return the all-components spec.
        """
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
    """Return whether `pipeline` can be driven by the step runner.

    Requires the explicit ``supports_step_execution`` flag (so a whole-request
    pipeline like DreamZero can opt out even though it satisfies the atom
    surface) plus EITHER the #4948 :class:`DiffusionV2Atoms` contract (migrated
    pipelines) OR the legacy :class:`SupportsStepExecution` contract (pipelines
    not yet migrated: helios / hunyuan_image3 / qwen_image / diffusers_adapter).
    """

    if getattr(pipeline, "supports_step_execution", False) is not True:
        return False
    return isinstance(pipeline, DiffusionV2Atoms) or isinstance(pipeline, SupportsStepExecution)


def supports_disaggregated_execution(pipeline: object) -> bool:
    """Return whether `pipeline` can be driven as a disaggregated stage (RFC #4590).

    Uses an explicit ``supports_disaggregated_execution`` flag plus the full
    :class:`DiffusionV2Atoms` contract (the #4948 atom surface — the
    encode/diffuse/decode atoms, ``pack_stage_state`` / ``unpack_stage_state``
    payload hooks, plus the RFC #4590 ``required_components_for_stage``
    extension). ``isinstance`` against the runtime-checkable Protocol validates
    the hooks are present; the flag lets a step-only pipeline opt out even though
    it satisfies the same atom surface.
    """

    if not bool(getattr(pipeline, "supports_disaggregated_execution", False)):
        return False
    return isinstance(pipeline, DiffusionV2Atoms)
