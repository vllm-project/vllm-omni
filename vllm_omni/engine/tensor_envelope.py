# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Versioned engine tensor handoff metadata with a host-tensor fallback."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, TypedDict

import torch

TENSOR_ENVELOPE_VERSION = 1
TENSOR_ENVELOPE_META_KEY = "tensor_envelopes"
INLINE_TENSOR_HANDLE = "inline_tensor"


class TensorHandle(TypedDict):
    kind: str
    payload_path: str


class TensorEnvelope(TypedDict, total=False):
    version: int
    request_id: str
    session_id: str
    epoch: int
    chunk_id: int
    shape: list[int]
    dtype: str
    device: str
    handle: TensorHandle


def build_inline_tensor_envelope(
    tensor: torch.Tensor,
    *,
    request_id: str,
    payload_path: str,
    session_id: str | None = None,
    epoch: int | None = None,
    chunk_id: int | None = None,
) -> TensorEnvelope:
    envelope: TensorEnvelope = {
        "version": TENSOR_ENVELOPE_VERSION,
        "request_id": str(request_id),
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "handle": {
            "kind": INLINE_TENSOR_HANDLE,
            "payload_path": payload_path,
        },
    }
    if session_id:
        envelope["session_id"] = str(session_id)
    if epoch is not None:
        envelope["epoch"] = int(epoch)
    if chunk_id is not None:
        envelope["chunk_id"] = int(chunk_id)
    return envelope


def install_tensor_envelope(
    buffer: dict[str, object],
    *,
    name: str,
    envelope: TensorEnvelope,
) -> None:
    meta = buffer.setdefault("meta", {})
    if not isinstance(meta, dict):
        raise TypeError("tensor envelope metadata requires a dict-valued meta field")
    envelopes = meta.setdefault(TENSOR_ENVELOPE_META_KEY, {})
    if not isinstance(envelopes, dict):
        raise TypeError("tensor_envelopes metadata must be a dict")
    envelopes[name] = envelope


def _request_id_from_buffer(buffer: Mapping[str, object]) -> str | None:
    value = buffer.get("global_request_id") or buffer.get("request_id")
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        value = value[0] if value else None
    return str(value) if value is not None else None


def validate_inline_tensor_envelope(
    buffer: Mapping[str, object],
    *,
    name: str,
    payload: object,
) -> object:
    """Validate a known inline envelope and return its tensor/list payload.

    An absent envelope is the legacy host-list fallback. Unknown handle kinds
    also preserve the payload so a future transport can materialize it before
    this boundary. Malformed metadata for the current inline kind fails
    explicitly instead of silently accepting a cross-request tensor.
    """
    meta = buffer.get("meta")
    envelopes = meta.get(TENSOR_ENVELOPE_META_KEY) if isinstance(meta, Mapping) else None
    envelope = envelopes.get(name) if isinstance(envelopes, Mapping) else None
    if not isinstance(envelope, Mapping):
        return payload

    if envelope.get("version") != TENSOR_ENVELOPE_VERSION:
        raise ValueError(f"unsupported tensor envelope version for {name}: {envelope.get('version')!r}")

    expected_request_id = _request_id_from_buffer(buffer)
    actual_request_id = envelope.get("request_id")
    if expected_request_id is not None and str(actual_request_id) != expected_request_id:
        raise ValueError(
            f"tensor envelope request mismatch for {name}: "
            f"expected={expected_request_id!r} actual={actual_request_id!r}"
        )
    duplex = buffer.get("duplex")
    if isinstance(duplex, Mapping):
        for envelope_key, duplex_key in (
            ("session_id", "session_id"),
            ("epoch", "epoch"),
            ("chunk_id", "turn_id"),
        ):
            expected = duplex.get(duplex_key)
            actual = envelope.get(envelope_key)
            if expected is not None and actual != expected:
                raise ValueError(
                    f"tensor envelope {envelope_key} mismatch for {name}: "
                    f"expected={expected!r} actual={actual!r}"
                )
    handle = envelope.get("handle")
    if not isinstance(handle, Mapping):
        raise ValueError(f"tensor envelope handle missing for {name}")
    if handle.get("payload_path") != name:
        raise ValueError(
            f"tensor envelope payload path mismatch for {name}: "
            f"{handle.get('payload_path')!r}"
        )
    if handle.get("kind") != INLINE_TENSOR_HANDLE:
        return payload
    if not isinstance(payload, torch.Tensor):
        # Host/list fallback remains accepted when the transport materialized
        # the tensor through the legacy representation.
        return payload
    if list(payload.shape) != list(envelope.get("shape", [])):
        raise ValueError(
            f"tensor envelope shape mismatch for {name}: "
            f"expected={envelope.get('shape')} actual={list(payload.shape)}"
        )
    if str(payload.dtype) != envelope.get("dtype"):
        raise ValueError(
            f"tensor envelope dtype mismatch for {name}: "
            f"expected={envelope.get('dtype')!r} actual={str(payload.dtype)!r}"
        )
    return payload
