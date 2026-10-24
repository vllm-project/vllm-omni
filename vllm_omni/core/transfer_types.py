# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Typed contracts for cross-stage execution.

The vLLM-Omni control plane historically forwarded payloads between stages using
ad-hoc dictionaries whose keys (``chunk_id``, ``hidden_states``, ``embed``, ...)
carried implicit semantics.  Issue #2336 identified two data-plane gaps that
block the control-plane decoupling work (Replica-as-first-class-object, unified
runtime socket ownership):

* No shared transfer payload contract — each model hand-rolls its own payload
  scattering ``.detach().cpu()`` calls.
* Implicit chunk semantics — phase (prefill / decode / preempt), sequence
  index, and final-vs-intermediate are inferred from a single ``chunk_id``
  counter baked into string keys rather than carried explicitly.

This module adds the missing typing foundation so future refactors can consume
well-defined objects instead of probing ``inspect.signature`` or string keys.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Literal

import torch

# ── Device policy literals ──────────────────────────────────────────────
#
# Where should the receiver place ``TransferPayload.tensors`` before
# forwarding to the actual model runner?
#
# * ``keep``       — leave tensors on their current device (the sender
#                    already placed them where the consumer expects).
# * ``cpu``        — move / detach tensors to CPU (default for cross-device
#                    transfers so the receiver can move them at its leisure).
# * ``pin_memory`` — same as ``cpu`` but pinned for faster H2D DMA when the
#                    consumer is on GPU.

DevicePolicy = Literal["keep", "cpu", "pin_memory"]
_DEFAULT_DEVICE_POLICY: DevicePolicy = "cpu"


# ── ChunkPhase ─────────────────────────────────────────────────────────
#
# Logical execution phase of a chunk.  Chunk id zero was historically used
# as a stand-in for "prefill", everything else as "decode", with preemption
# and finalization implied by neighbouring boolean flags.  ``ChunkPhase``
# makes each of those states an explicit enum value so downstream code can
# branch on a single field.


class ChunkPhase(str, enum.Enum):
    """Explicit phase labels replacing implicit ``chunk_id`` inference."""

    PREFILL = "prefill"
    """Initial prompt encoding chunk (chunk_id == 0 historically)."""

    DECODE = "decode"
    """Speculative / incremental decode chunk (chunk_id > 0 historically)."""

    PREEMPT = "preempt"
    """Synthetic chunk emitted after a preemption; carries re-prefill data."""

    FINAL = "final"
    """Terminal chunk carrying only end-of-sequence metadata."""


@dataclass(frozen=True)
class ChunkMeta:
    """Explicit metadata for chunked multi-stage execution.

    Fields
    ------
    phase:
        Logical execution phase.  Inferred from ``chunk_index`` when the
        caller only has an integer counter (``chunk_index == 0`` → PREFILL,
        ``chunk_index > 0`` → DECODE), but producers that know about
        preemption / termination should set the phase directly.
    sequence_id:
        Index of the sequence within its batch.  Used to correlate chunks
        that belong to the same concurrent sequence.
    chunk_index:
        Monotonic counter per sequence.  Starts at 0 for PREFILL and
        increments by 1 for every subsequent DECODE chunk.  PREEMPT resets
        the counter to 0 for the resumed prefill.
    is_final:
        ``True`` iff this chunk is the terminal chunk of the sequence.  A
        FINAL phase always has ``is_final=True``; other phases may also
        be marked final when the sequence ends early.
    total_chunks:
        Optional total chunk count if the producer knows it in advance
        (rare — most streaming producers only know ``is_final``).
    """

    phase: ChunkPhase
    sequence_id: int = 0
    chunk_index: int = 0
    is_final: bool = False
    total_chunks: int | None = None

    # ── Factory helpers (keep back-compat with integer-based callers) ──

    @classmethod
    def from_chunk_index(
        cls,
        chunk_index: int,
        *,
        sequence_id: int = 0,
        is_final: bool = False,
        total_chunks: int | None = None,
    ) -> ChunkMeta:
        """Infer ``ChunkPhase`` from a legacy integer chunk counter.

        ``chunk_index == 0`` → PREFILL, anything else → DECODE.  Callers
        that produce PREEMPT / FINAL chunks should construct ``ChunkMeta``
        directly instead of going through this helper.
        """
        phase = ChunkPhase.PREFILL if chunk_index == 0 else ChunkPhase.DECODE
        # A bare chunk_index counter cannot express "preempt"; we only mark
        # PREEMPT when the caller asks for it explicitly.
        return cls(
            phase=phase,
            sequence_id=sequence_id,
            chunk_index=chunk_index,
            is_final=is_final,
            total_chunks=total_chunks,
        )

    @classmethod
    def prefill(
        cls,
        *,
        sequence_id: int = 0,
        is_final: bool = False,
        total_chunks: int | None = None,
    ) -> ChunkMeta:
        """Shorthand for the initial prefill chunk."""
        return cls(
            phase=ChunkPhase.PREFILL,
            sequence_id=sequence_id,
            chunk_index=0,
            is_final=is_final,
            total_chunks=total_chunks,
        )

    @classmethod
    def decode(
        cls,
        chunk_index: int,
        *,
        sequence_id: int = 0,
        is_final: bool = False,
    ) -> ChunkMeta:
        """Shorthand for a decode chunk (``chunk_index`` must be > 0)."""
        assert chunk_index > 0, (
            "decode ChunkMeta requires chunk_index > 0; use ChunkMeta.prefill() for chunk_index == 0"
        )
        return cls(
            phase=ChunkPhase.DECODE,
            sequence_id=sequence_id,
            chunk_index=chunk_index,
            is_final=is_final,
            total_chunks=None,
        )

    @classmethod
    def preempt(
        cls,
        *,
        sequence_id: int = 0,
    ) -> ChunkMeta:
        """Shorthand for a preemption-triggered re-prefill chunk."""
        return cls(
            phase=ChunkPhase.PREEMPT,
            sequence_id=sequence_id,
            chunk_index=0,
            is_final=False,
            total_chunks=None,
        )

    @classmethod
    def final(
        cls,
        *,
        sequence_id: int = 0,
        chunk_index: int = 0,
    ) -> ChunkMeta:
        """Shorthand for a terminal (end-of-sequence) chunk."""
        return cls(
            phase=ChunkPhase.FINAL,
            sequence_id=sequence_id,
            chunk_index=chunk_index,
            is_final=True,
            total_chunks=chunk_index + 1,
        )

    def is_prefill(self) -> bool:
        """True iff this chunk is (or was inferred as) a prefill chunk."""
        return self.phase in (ChunkPhase.PREFILL, ChunkPhase.PREEMPT)

    def is_decode(self) -> bool:
        """True iff this chunk is (or was inferred as) a decode chunk."""
        return self.phase == ChunkPhase.DECODE

    def is_preempt(self) -> bool:
        """True iff this chunk is a preemption-triggered re-prefill."""
        return self.phase == ChunkPhase.PREEMPT

    def is_terminal(self) -> bool:
        """True iff the sequence ended on or before this chunk."""
        return self.is_final or self.phase == ChunkPhase.FINAL


# ── TransferPayload ─────────────────────────────────────────────────────
#
# Typed container for anything that crosses a stage boundary.  Replaces the
# hand-rolled ``OmniPayload`` / ``custom_process_input_func`` dictionary
# contracts with a single, explicitly-typed envelope.


@dataclass
class TransferPayload:
    """Typed payload for cross-stage forwarding.

    Fields
    ------
    tensors:
        Named tensors the downstream stage needs.  Keys are semantic names
        (``"hidden_states"``, ``"prefill_embedding"``, ...) — downstream
        consumers look them up by name rather than probing for specific
        dictionaries.  ``None`` entries are allowed for optional tensors
        and will be dropped by :meth:`as_dict`.
    device_policy:
        Where tensors should live when the receiver starts processing them.
        Defaults to ``"cpu"`` so the receiver can move them at its leisure
        (matches the historic ``.detach().cpu()`` convention).
    metadata:
        Model-specific opaque info the orchestrator does not interpret
        (speaker id, language, shapes, config blobs, ...).  Consumers treat
        this as a black box unless they know the producer.
    source_stage:
        Logical stage id or label that produced this payload.  ``None`` for
        externally-injected inputs.
    target_stage:
        Logical stage id or label that should consume this payload.  The
        orchestrator uses this to sanity-check its routing decisions.
    chunk:
        Explicit chunk semantics when this payload is part of a chunked
        multi-stage execution.  ``None`` for single-shot / non-chunked
        transfers.
    request_id:
        Optional request identifier for tracing.  Carried through as a
        convenience even though the orchestrator already knows the id.
    """

    tensors: dict[str, torch.Tensor | None] = field(default_factory=dict)
    device_policy: DevicePolicy = _DEFAULT_DEVICE_POLICY
    metadata: dict[str, Any] = field(default_factory=dict)
    source_stage: Any | None = None
    target_stage: Any | None = None
    chunk: ChunkMeta | None = None
    request_id: str | None = None

    # ── Construction helpers ──────────────────────────────────────────

    @classmethod
    def _flatten_nested_dict(
        cls,
        raw: dict[str, Any],
        *,
        prefix: str = "",
    ) -> tuple[dict[str, torch.Tensor | None], dict[str, Any]]:
        """Recursively flatten a nested dict into (dotted_keyed_tensors, remaining_nested_meta).

        Any ``torch.Tensor`` anywhere in the nesting ends up in the tensors
        dict with a dotted key (``"hidden_states.layers.0"``).  Non-tensor
        leaves are stored back into the remaining_nested_meta dict so that
        :meth:`as_dict` can rebuild the original schema.
        """
        tensors: dict[str, torch.Tensor | None] = {}
        meta: dict[str, Any] = {}
        for k, v in raw.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, torch.Tensor):
                tensors[key] = v
            elif isinstance(v, dict):
                sub_tensors, sub_meta = cls._flatten_nested_dict(v, prefix=key)
                tensors.update(sub_tensors)
                if sub_meta:
                    meta[k] = sub_meta
            else:
                meta[k] = v
        return tensors, meta

    @classmethod
    def from_dict(
        cls,
        raw: dict[str, Any],
        *,
        source_stage: Any | None = None,
        target_stage: Any | None = None,
        device_policy: DevicePolicy | None = None,
        chunk: ChunkMeta | None = None,
        request_id: str | None = None,
    ) -> TransferPayload:
        """Wrap an ad-hoc ``OmniPayload``-style dict as a ``TransferPayload``.

        The input dict is treated as a set of *named* fields; values that
        are ``torch.Tensor`` instances land in ``self.tensors``, everything
        else lands in ``self.metadata``.  Nested dicts are recursively
        scanned — every tensor, no matter how deeply nested, ends up in
        ``self.tensors`` under a dotted key (``"embed.prefill"``,
        ``"hidden_states.layers.0"``).  Non-tensor leaves are preserved
        as nested metadata so :meth:`as_dict` can rebuild the original
        schema for legacy consumers.
        """
        tensors, metadata = cls._flatten_nested_dict(raw)
        return cls(
            tensors=tensors,
            device_policy=device_policy or _DEFAULT_DEVICE_POLICY,
            metadata=metadata,
            source_stage=source_stage,
            target_stage=target_stage,
            chunk=chunk,
            request_id=request_id,
        )

    @classmethod
    def from_omni_payload(
        cls,
        omni_payload: Any,
        *,
        source_stage: Any | None = None,
        target_stage: Any | None = None,
        device_policy: DevicePolicy | None = None,
        chunk: ChunkMeta | None = None,
        request_id: str | None = None,
    ) -> TransferPayload:
        """Wrap an ``OmniPayload`` TypedDict as a ``TransferPayload``.

        Accepts either an ``OmniPayload`` dict or an ``OmniPayloadStruct``.
        Uses :func:`from_dict` under the hood so any nested structure is
        flattened into dotted keys.
        """
        from vllm_omni.data_entry_keys import to_dict  # local import to avoid cycle

        raw = to_dict(omni_payload) if hasattr(omni_payload, "__struct_fields__") else dict(omni_payload)
        return cls.from_dict(
            raw,
            source_stage=source_stage,
            target_stage=target_stage,
            device_policy=device_policy,
            chunk=chunk,
            request_id=request_id,
        )

    # ── Access helpers ────────────────────────────────────────────────

    def get_tensor(self, name: str, default: torch.Tensor | None = None) -> torch.Tensor | None:
        """Return a tensor by semantic name, falling back to ``default``."""
        t = self.tensors.get(name, default)
        if t is None:
            return default
        return t

    def set_tensor(self, name: str, tensor: torch.Tensor | None) -> None:
        """Insert or replace a tensor.  ``None`` drops the entry."""
        if tensor is None:
            self.tensors.pop(name, None)
        else:
            self.tensors[name] = tensor

    def enforce_device_policy(self) -> TransferPayload:
        """Apply ``self.device_policy`` to every tensor in ``self.tensors``.

        Returns ``self`` for chaining.  Idempotent — calling it twice is a
        no-op.
        """
        policy = self.device_policy
        if policy == "keep":
            return self
        if policy == "cpu":
            self.tensors = {
                k: (v.detach().cpu() if isinstance(v, torch.Tensor) and v.device.type != "cpu" else v)
                for k, v in self.tensors.items()
            }
        elif policy == "pin_memory":
            self.tensors = {
                k: (
                    v.detach().cpu().pin_memory()
                    if isinstance(v, torch.Tensor) and v.device.type != "cpu"
                    else (v.pin_memory() if isinstance(v, torch.Tensor) and not v.is_pinned() else v)
                )
                for k, v in self.tensors.items()
            }
        return self

    # ── Serialisation helpers ────────────────────────────────────────

    def as_dict(self) -> dict[str, Any]:
        """Rebuild a nested dict representation (strips ``None`` values).

        Consumers that still speak the legacy ``OmniPayload``-style
        dictionary contract can call this to downgrade before processing.
        Dotted-keyed entries in ``self.tensors`` (``"embed.prefill"``) are
        folded back into their original nesting (``{"embed": {"prefill": t}}``)
        so legacy consumers that read ``payload["embed"]["prefill"]`` keep
        working.
        """
        # Start from the non-tensor metadata — it already has the nested shape.
        out: dict[str, Any] = {k: v for k, v in self.metadata.items() if v is not None}
        # Fold each (non-None) tensor back into nested buckets.
        for dotted_key, t in self.tensors.items():
            if t is None:
                continue
            parts: list[str | int] = [int(p) if p.isdigit() else p for p in dotted_key.split(".")]
            cursor: Any = out
            for part in parts[:-1]:
                if part not in cursor or not isinstance(cursor[part], dict):
                    cursor[part] = {}
                cursor = cursor[part]
            cursor[parts[-1]] = t
        return out

    # ── Introspection ─────────────────────────────────────────────────

    def describe(self) -> str:
        """Human-readable summary for debug logs."""
        tensor_shapes = {
            k: tuple(v.shape) if isinstance(v, torch.Tensor) else None for k, v in self.tensors.items() if v is not None
        }
        chunk_str = f"chunk={self.chunk}" if self.chunk is not None else "chunk=None"
        return (
            f"TransferPayload(req={self.request_id}, "
            f"{self.source_stage}->{self.target_stage}, "
            f"policy={self.device_policy}, "
            f"tensors={tensor_shapes}, "
            f"{chunk_str})"
        )


# ── Re-exports for convenience ─────────────────────────────────────────

__all__ = [
    "ChunkMeta",
    "ChunkPhase",
    "DevicePolicy",
    "TransferPayload",
]
