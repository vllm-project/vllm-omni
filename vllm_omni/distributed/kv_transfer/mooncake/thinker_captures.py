# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Validated Mooncake Thinker captures and request-local row assembly.

Each stored block carries both capture tensors the Talker needs: the Thinker
input embedding and the hidden state of the configured layer. "HS" in the key
format and symbol names refers to that pair, not to hidden states alone.

This module has no dependency on Mooncake or vLLM. Store operations and request
attempt ownership belong to the connector; tensors here are CPU snapshots.
"""

from __future__ import annotations

import hashlib
import json
import struct
from collections.abc import Mapping
from dataclasses import dataclass

import torch
from safetensors import SafetensorError
from safetensors.torch import load as load_tensors
from safetensors.torch import save as save_tensors

_NAMES = frozenset({"embedding", "hidden"})
_DTYPES = {torch.float16: "F16", torch.bfloat16: "BF16", torch.float32: "F32"}
_FORMAT = "omni-mooncake-hs-v1"
_MAX_HEADER_BYTES = 16384
CaptureTensors = dict[str, torch.Tensor]


def prefix_digest(block_hash: bytes | bytearray | memoryview) -> str:
    """Use the KV connector's chained block hash, including its media identity."""
    if not isinstance(block_hash, (bytes, bytearray, memoryview)):
        raise TypeError("unsupported KV prefix hash type")
    # Native lookup RPCs expose memoryviews; scheduler metadata carries bytes.
    return hashlib.sha256(b"bytes:" + bytes(block_hash)).hexdigest()


def hs_key(namespace: str, block_hash: bytes | bytearray | memoryview) -> str:
    namespace_hash = hashlib.sha256(namespace.encode()).hexdigest()
    return f"omni-hs-v1:{namespace_hash}:{prefix_digest(block_hash)}"


@dataclass(frozen=True)
class HSBlockSpec:
    """Expected identity and layout supplied by the reader, not the object."""

    namespace: str
    prefix_hash: str
    start: int
    end: int
    hidden_layer: int
    hidden_size: int
    dtype: torch.dtype

    def __post_init__(self) -> None:
        if not self.namespace or not self.prefix_hash:
            raise ValueError("HS namespace and prefix hash must be nonempty")
        if self.start < 0 or self.end <= self.start or self.hidden_size <= 0 or self.hidden_layer <= 0:
            raise ValueError("invalid HS token range or capture layout")
        if self.dtype not in _DTYPES:
            raise ValueError("unsupported HS dtype")

    @property
    def shape(self) -> tuple[int, int]:
        return self.end - self.start, self.hidden_size

    @property
    def tensor_bytes(self) -> int:
        return 2 * self.shape[0] * self.shape[1] * torch.empty((), dtype=self.dtype).element_size()

    def metadata(self) -> dict[str, str]:
        return {
            "format": _FORMAT,
            "namespace": self.namespace,
            "prefix_hash": self.prefix_hash,
            "start": str(self.start),
            "end": str(self.end),
            "hidden_layer": str(self.hidden_layer),
            "hidden_size": str(self.hidden_size),
            "dtype": _DTYPES[self.dtype],
        }


def _validate_captures(tensors: Mapping[str, torch.Tensor]) -> int:
    if set(tensors) != _NAMES:
        raise ValueError("HS requires both embedding and hidden captures")
    embedding, hidden = tensors["embedding"], tensors["hidden"]
    if any(tensor.device.type != "cpu" or tensor.ndim != 2 for tensor in (embedding, hidden)):
        raise ValueError("HS captures must be rank-two CPU tensors")
    if embedding.shape != hidden.shape or embedding.dtype != hidden.dtype:
        raise ValueError("HS captures must have matching shapes and dtypes")
    if embedding.dtype not in _DTYPES or embedding.shape[1] <= 0:
        raise ValueError("unsupported HS capture layout")
    return embedding.shape[0]


def encode_hs(spec: HSBlockSpec, tensors: Mapping[str, torch.Tensor]) -> bytes:
    _validate_captures(tensors)
    if any(tuple(tensor.shape) != spec.shape or tensor.dtype != spec.dtype for tensor in tensors.values()):
        raise ValueError("HS captures do not match their declared layout")
    payload = save_tensors({name: tensor.contiguous() for name, tensor in tensors.items()}, metadata=spec.metadata())
    if struct.unpack_from("<Q", payload)[0] > _MAX_HEADER_BYTES:
        raise ValueError("HS metadata exceeds the header size limit")
    return payload


def decode_hs(payload: bytes, expected: HSBlockSpec) -> CaptureTensors:
    """Reject incompatible headers before safetensors allocates the tensors."""
    if len(payload) < 8 or len(payload) > 8 + _MAX_HEADER_BYTES + expected.tensor_bytes:
        raise ValueError("invalid HS object size")
    header_size = struct.unpack_from("<Q", payload)[0]
    if header_size > _MAX_HEADER_BYTES or len(payload) != 8 + header_size + expected.tensor_bytes:
        raise ValueError("truncated or oversized HS object")
    try:
        header = json.loads(payload[8 : 8 + header_size])
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid HS header") from exc
    if not isinstance(header, dict) or set(header) != _NAMES | {"__metadata__"}:
        raise ValueError("invalid HS capture names")
    if header["__metadata__"] != expected.metadata():
        raise ValueError("incompatible HS identity or token range")
    for name in _NAMES:
        tensor = header[name]
        if (
            not isinstance(tensor, dict)
            or tensor.get("shape") != list(expected.shape)
            or tensor.get("dtype") != _DTYPES[expected.dtype]
        ):
            raise ValueError("incompatible HS tensor layout")
    try:
        return load_tensors(payload)
    except SafetensorError as exc:
        raise ValueError("invalid HS tensor payload") from exc


class RequestHS:
    """Assemble one request attempt and stage complete prompt blocks for saving.

    ``append`` returns only this step's rows, prepending a restored prefix on
    the first step. Saving clips at the prompt end; decode rows remain in the
    returned output. A full save queue disables further saves for this attempt
    without changing its output. The connector must serialize calls and discard
    this object when an attempt is cancelled, preempted, or fails.
    """

    def __init__(self, prompt_tokens: int, block_size: int, max_bytes: int):
        if prompt_tokens <= 0 or block_size <= 0 or max_bytes <= 0:
            raise ValueError("invalid HS request limits")
        self.prompt_tokens = prompt_tokens
        self.block_size = block_size
        self.max_bytes = max_bytes
        self.save_enabled = True
        self._restored: CaptureTensors | None = None
        self._next_token: int | None = None
        self._save_cursor = 0
        self._partial: CaptureTensors | None = None
        self._partial_rows = 0
        self._blocks: list[tuple[int, int, CaptureTensors]] = []
        self._layout: tuple[int, torch.dtype] | None = None

    @property
    def buffered_bytes(self) -> int:
        groups = [group for group in (self._restored, self._partial) if group is not None]
        groups.extend(group for _, _, group in self._blocks)
        return sum(tensor.numel() * tensor.element_size() for group in groups for tensor in group.values())

    def _check_layout(self, tensors: Mapping[str, torch.Tensor]) -> int:
        rows = _validate_captures(tensors)
        layout = (tensors["embedding"].shape[1], tensors["embedding"].dtype)
        if self._layout is not None and layout != self._layout:
            raise ValueError("HS capture layout changed during a request")
        self._layout = layout
        return rows

    def restore(self, tensors: Mapping[str, torch.Tensor]) -> None:
        if self._next_token is not None or self._restored is not None:
            raise ValueError("HS prefix already installed or consumed")
        rows = self._check_layout(tensors)
        if rows <= 0 or rows > self.prompt_tokens or rows % self.block_size:
            raise ValueError("restored HS must cover complete prompt blocks")
        size = sum(tensor.numel() * tensor.element_size() for tensor in tensors.values())
        if size > self.max_bytes:
            raise ValueError("restored HS exceeds request memory limit")
        self._restored = dict(tensors)
        self._save_cursor = rows

    def append(self, start: int, tensors: Mapping[str, torch.Tensor]) -> CaptureTensors:
        rows = self._check_layout(tensors)
        if start < 0 or rows == 0:
            raise ValueError("HS step must contain a nonnegative, nonempty token range")
        if self._next_token is not None:
            if start != self._next_token:
                raise ValueError("HS steps have a gap or duplicate token positions")
            merged = dict(tensors)
        else:
            restored_rows = 0 if self._restored is None else self._restored["embedding"].shape[0]
            if start > restored_rows:
                raise ValueError("computed HS starts after the restored prefix")
            if start and self._restored is not None:
                # The scheduler can replay the final prompt token even after a
                # full load. Only restored rows before the actual start belong
                # in the output; the newly computed row replaces the overlap.
                merged = {name: torch.cat((self._restored[name][:start], tensors[name])) for name in _NAMES}
            else:
                merged = dict(tensors)
            self._restored = None
        self._next_token = start + rows
        self._stage_blocks(start, tensors)
        return merged

    def _stage_blocks(self, start: int, tensors: Mapping[str, torch.Tensor]) -> None:
        if not self.save_enabled:
            return
        end = min(start + tensors["embedding"].shape[0], self.prompt_tokens)
        # Restored blocks already exist in the store; only stage new rows.
        begin = max(start, self._save_cursor)
        if end <= begin:
            return
        if begin != self._save_cursor + self._partial_rows:
            raise ValueError("HS save rows have a gap")
        bytes_per_row = 2 * tensors["embedding"].shape[1] * tensors["embedding"].element_size()
        while begin < end:
            # Do not allocate a fragment that can never form a prompt block.
            if self._save_cursor + self.block_size > self.prompt_tokens:
                return
            if self._partial is None:
                if self.buffered_bytes + self.block_size * bytes_per_row > self.max_bytes:
                    self.save_enabled = False
                    return
                # The allocation is the eventual saved block. Filling it over
                # several steps never creates another concatenation buffer.
                self._partial = {
                    name: torch.empty((self.block_size, tensor.shape[1]), dtype=tensor.dtype)
                    for name, tensor in tensors.items()
                }
            count = min(end - begin, self.block_size - self._partial_rows)
            for name, tensor in tensors.items():
                self._partial[name][self._partial_rows : self._partial_rows + count].copy_(
                    tensor[begin - start : begin - start + count]
                )
            self._partial_rows += count
            begin += count
            if self._partial_rows == self.block_size:
                self._blocks.append((self._save_cursor, self._save_cursor + self.block_size, self._partial))
                self._save_cursor += self.block_size
                self._partial = None
                self._partial_rows = 0

    def pop_blocks(self) -> list[tuple[int, int, CaptureTensors]]:
        blocks, self._blocks = self._blocks, []
        return blocks
