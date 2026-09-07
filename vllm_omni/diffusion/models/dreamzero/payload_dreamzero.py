# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""DreamZero cross-stage payload schema.

DreamZero keeps a typed payload internally and degrades it to a plain nested
dict at the wire boundary. That dict is all a transport ever sees: no dataclass
identity, no scheduler, generator, module, device handle, live request state, KV
state or callable survives ``to_dict()``. A stage therefore cannot accidentally
depend on the producer's Python objects, and the same payload travels unchanged
over shared memory today and over any other transport later.

Layout::

    {
        "request_id": request_id,
        "boundary": "encode_to_dit" | "dit_to_decode",
        "payload_version": 1,
        "scalar_fields": {...},          # consumed by the receiving stage
        "tensor_fields": {...},
        "private_scalar_fields": {...},  # opaque passthrough for a later stage
        "private_tensor_fields": {...},
    }

The ``private_*`` groups exist so a middle stage can forward postprocess
metadata it does not itself consume (DreamZero's decode stage needs the
observation state and embodiment produced by encode) without the middle stage
having to know those field names.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import FunctionType, MethodType
from typing import Any

import torch

from vllm_omni.diffusion.models.dreamzero.utils import (
    DREAMZERO_BOUNDARY_DIT_TO_DECODE,
    DREAMZERO_BOUNDARY_ENCODE_TO_DIT,
    DREAMZERO_PAYLOAD_VERSION,
    DREAMZERO_STAGE_PAYLOAD_KEY,
)

_KNOWN_BOUNDARIES = frozenset(
    {
        DREAMZERO_BOUNDARY_ENCODE_TO_DIT,
        DREAMZERO_BOUNDARY_DIT_TO_DECODE,
    }
)

# Scalar types allowed on the wire. Anything else (module, scheduler, generator,
# callable, device, session object) is rejected by ``validate()`` rather than
# silently pickled into a transport buffer.
_ALLOWED_SCALARS = (bool, int, float, str, bytes, type(None))

_MISSING = object()


class DreamZeroPayloadError(ValueError):
    """A cross-stage payload is absent, malformed, or of the wrong boundary."""


class DreamZeroStaleRequestError(ValueError):
    """A payload is stale, duplicated, out of order, or from a fenced epoch.

    Raised by the committed-progress authority *before* any model or KV mutation,
    so a rejected request cannot corrupt a live AR-Diffusion session.
    """


def _reject_opaque(where: str, name: str, value: object) -> None:
    if isinstance(value, (torch.nn.Module, torch.device, torch.Generator, FunctionType, MethodType)):
        raise DreamZeroPayloadError(
            f"{where}[{name!r}] is a {type(value).__name__}; DreamZero stage payloads "
            "carry only plain scalars, sequences and tensors across the wire."
        )
    if callable(value):
        raise DreamZeroPayloadError(f"{where}[{name!r}] is callable; DreamZero stage payloads carry only data.")


def _validate_scalar(where: str, name: str, value: object) -> None:
    _reject_opaque(where, name, value)
    if isinstance(value, _ALLOWED_SCALARS):
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_scalar(where, f"{name}[{index}]", item)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise DreamZeroPayloadError(f"{where}[{name!r}] has a non-string key {key!r}.")
            _validate_scalar(where, f"{name}.{key}", item)
        return
    raise DreamZeroPayloadError(
        f"{where}[{name!r}] has unsupported type {type(value).__name__}; "
        "put tensors in the tensor field groups and keep scalar groups plain."
    )


def _validate_tensor(where: str, name: str, value: object) -> None:
    _reject_opaque(where, name, value)
    if not isinstance(value, torch.Tensor):
        raise DreamZeroPayloadError(
            f"{where}[{name!r}] has type {type(value).__name__}; tensor field groups accept tensors only."
        )


def _move(value: torch.Tensor, device: torch.device | str | None) -> torch.Tensor:
    if device is None:
        return value
    target = torch.device(device)
    return value if value.device == target else value.to(target)


@dataclass
class DreamZeroStagePayload:
    """Typed view of one DreamZero cross-stage payload."""

    request_id: str
    boundary: str
    payload_version: int = DREAMZERO_PAYLOAD_VERSION
    scalar_fields: dict[str, Any] = field(default_factory=dict)
    tensor_fields: dict[str, torch.Tensor] = field(default_factory=dict)
    private_scalar_fields: dict[str, Any] = field(default_factory=dict)
    private_tensor_fields: dict[str, torch.Tensor] = field(default_factory=dict)

    # -- wire format ----------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Degrade to the plain nested dict a transport is allowed to see."""
        self.validate()
        return {
            "request_id": self.request_id,
            "boundary": self.boundary,
            "payload_version": self.payload_version,
            "scalar_fields": dict(self.scalar_fields),
            "tensor_fields": dict(self.tensor_fields),
            "private_scalar_fields": dict(self.private_scalar_fields),
            "private_tensor_fields": dict(self.private_tensor_fields),
        }

    @classmethod
    def from_dict(cls, raw: object) -> DreamZeroStagePayload:
        """Rebuild a typed payload from a wire dict, validating as we go."""
        if raw is None:
            raise DreamZeroPayloadError("DreamZero stage payload is missing.")
        if isinstance(raw, DreamZeroStagePayload):
            # Monolithic / same-process execution never leaves Python; accept the
            # typed object so callers share one code path with the wire case.
            raw.validate()
            return raw
        if not isinstance(raw, dict):
            raise DreamZeroPayloadError(f"DreamZero stage payload must be a dict, got {type(raw).__name__}.")
        payload = cls(
            request_id=str(raw.get("request_id") or ""),
            boundary=str(raw.get("boundary") or ""),
            payload_version=int(raw.get("payload_version", -1)),
            scalar_fields=dict(raw.get("scalar_fields") or {}),
            tensor_fields=dict(raw.get("tensor_fields") or {}),
            private_scalar_fields=dict(raw.get("private_scalar_fields") or {}),
            private_tensor_fields=dict(raw.get("private_tensor_fields") or {}),
        )
        payload.validate()
        return payload

    # -- validation -----------------------------------------------------------

    def validate(
        self,
        *,
        request_id: str | None = None,
        boundary: str | None = None,
    ) -> None:
        """Check the envelope and the field groups.

        ``request_id`` / ``boundary`` let a consumer assert that the payload it
        received belongs to the request it is executing and was produced for the
        edge it sits on. Both checks run before any model or KV mutation.
        """
        if not self.request_id:
            raise DreamZeroPayloadError("DreamZero stage payload has no request_id.")
        if self.boundary not in _KNOWN_BOUNDARIES:
            raise DreamZeroPayloadError(
                f"DreamZero stage payload has unknown boundary {self.boundary!r}; "
                f"expected one of {sorted(_KNOWN_BOUNDARIES)}."
            )
        if self.payload_version != DREAMZERO_PAYLOAD_VERSION:
            raise DreamZeroPayloadError(
                f"DreamZero stage payload version {self.payload_version} is not supported "
                f"by this build (expected {DREAMZERO_PAYLOAD_VERSION}); the producing and "
                "consuming stages are running different versions."
            )
        for group_name, group in (
            ("scalar_fields", self.scalar_fields),
            ("private_scalar_fields", self.private_scalar_fields),
        ):
            for name, value in group.items():
                _validate_scalar(group_name, name, value)
        for group_name, group in (
            ("tensor_fields", self.tensor_fields),
            ("private_tensor_fields", self.private_tensor_fields),
        ):
            for name, value in group.items():
                _validate_tensor(group_name, name, value)

        if request_id is not None and self.request_id != request_id:
            raise DreamZeroPayloadError(
                f"DreamZero stage payload belongs to request {self.request_id!r} "
                f"but is being consumed by {request_id!r}."
            )
        if boundary is not None and self.boundary != boundary:
            raise DreamZeroPayloadError(
                f"DreamZero stage payload has boundary {self.boundary!r} but this stage consumes {boundary!r}."
            )

    # -- accessors ------------------------------------------------------------

    def scalar(self, name: str, default: Any = _MISSING) -> Any:
        if name in self.scalar_fields:
            return self.scalar_fields[name]
        if default is _MISSING:
            raise DreamZeroPayloadError(f"DreamZero {self.boundary!r} payload is missing scalar field {name!r}.")
        return default

    def tensor(self, name: str, default: Any = _MISSING) -> torch.Tensor | None:
        if name in self.tensor_fields:
            return self.tensor_fields[name]
        if default is _MISSING:
            raise DreamZeroPayloadError(f"DreamZero {self.boundary!r} payload is missing tensor field {name!r}.")
        return default

    def to_device(self, device: torch.device | str | None) -> DreamZeroStagePayload:
        """Return a copy with every tensor field on ``device``.

        Payload tensors cross a process boundary on CPU; the consuming stage
        pulls them onto its *own* current device rather than the producer's.
        """
        self.tensor_fields = {name: _move(value, device) for name, value in self.tensor_fields.items()}
        self.private_tensor_fields = {name: _move(value, device) for name, value in self.private_tensor_fields.items()}
        return self

    def as_custom_output(self) -> dict[str, Any]:
        """Wrap the wire dict under DreamZero's single transport key."""
        return {DREAMZERO_STAGE_PAYLOAD_KEY: self.to_dict()}


def get_incoming_stage_payload(prompt: object) -> DreamZeroStagePayload:
    """Read the upstream payload out of a consuming stage's prompt.

    The connector receive path writes into ``prompt["additional_information"]``;
    the orchestrator's inline fallback lands the same entry at the top level of
    the prompt dict. Look in both, in that order, so a stage behaves identically
    whether the payload arrived worker-to-worker or through the IPC hop.
    """
    if not isinstance(prompt, dict):
        raise DreamZeroPayloadError(
            f"DreamZero stage prompt must be a dict carrying {DREAMZERO_STAGE_PAYLOAD_KEY!r}, "
            f"got {type(prompt).__name__}."
        )
    additional = prompt.get("additional_information") or {}
    raw = additional.get(DREAMZERO_STAGE_PAYLOAD_KEY) if isinstance(additional, dict) else None
    if raw is None:
        raw = prompt.get(DREAMZERO_STAGE_PAYLOAD_KEY)
    if raw is None:
        raise DreamZeroPayloadError(
            f"DreamZero stage prompt carries no {DREAMZERO_STAGE_PAYLOAD_KEY!r}; the upstream "
            "stage produced no payload and no inline fallback was delivered."
        )
    return DreamZeroStagePayload.from_dict(raw)
