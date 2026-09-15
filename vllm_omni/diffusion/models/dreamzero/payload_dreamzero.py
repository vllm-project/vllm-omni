# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""DreamZero cross-stage payloads: plain nested dictionaries and tensors.

The envelope identifies request, boundary and schema version. Scalar/tensor
fields are consumed by the next stage; private fields pass through to later
postprocess. Live model, scheduler, generator and session objects stay local.
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

# Allowed wire scalars; validate() rejects opaque runtime objects.
_ALLOWED_SCALARS = (bool, int, float, str, bytes, type(None))

_MISSING = object()


class DreamZeroPayloadError(ValueError):
    """A cross-stage payload is absent, malformed, or of the wrong boundary."""


class DreamZeroStaleRequestError(ValueError):
    """Stale or out-of-order payload rejected before model or KV mutation."""


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

    def to_dict(self) -> dict[str, Any]:
        """Serialize the payload to its plain wire dictionary."""
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
            # Accept typed payloads for same-process handoffs.
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

    def validate(
        self,
        *,
        request_id: str | None = None,
        boundary: str | None = None,
    ) -> None:
        """Validate field groups and request/boundary identity before model or KV mutation."""
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
        """Copy tensor fields to the consuming stage's own device."""
        self.tensor_fields = {name: _move(value, device) for name, value in self.tensor_fields.items()}
        self.private_tensor_fields = {name: _move(value, device) for name, value in self.private_tensor_fields.items()}
        return self

    def as_custom_output(self) -> dict[str, Any]:
        """Wrap the wire dict under DreamZero's single transport key."""
        return {DREAMZERO_STAGE_PAYLOAD_KEY: self.to_dict()}


def get_incoming_stage_payload(prompt: object) -> DreamZeroStagePayload:
    """Read additional_information first, then the legacy top-level payload."""
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
