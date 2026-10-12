# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Bounded Lychee decision slabs leased until the runner's D2H fence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any

import torch
from vllm.logger import init_logger

from vllm_omni.model_executor.output_snapshot import (
    OutputCopyLifetimeError,
    PackedOutputSnapshot,
    pack_output_snapshot,
)

logger = init_logger(__name__)

LYCHEE_OUTPUT_COLUMNS = (
    "lychee_text_token_ids",
    "lychee_speech_token_ids",
    "lychee_control_token_ids",
    "lychee_mode_after",
    "lychee_tail_padding_until",
    "lychee_next_text_token_ids",
    "lychee_next_speech_token_ids",
    "lychee_tick",
    "lychee_execution_epoch",
    "lychee_model_position",
    "lychee_audio_window_seq",
)


@dataclass
class _OutputSlot:
    storage: torch.Tensor
    indices: torch.Tensor
    generation: int = 0
    snapshot: PackedOutputSnapshot | None = None
    producer_stream: Any = None
    completion: Any = None
    pending: bool = False


class LycheeOutputRing:
    """Owner-thread metadata; request cancel never frees a batch slab."""

    def __init__(self, *, max_num_reqs: int, capacity: int, device: torch.device) -> None:
        if max_num_reqs < 1 or capacity < 1:
            raise ValueError("Lychee output ring requires positive bounded capacity")
        self.max_num_reqs = max_num_reqs
        self.device = torch.device(device)
        self.slots = [
            _OutputSlot(
                torch.empty(len(LYCHEE_OUTPUT_COLUMNS) * max_num_reqs, dtype=torch.int32, device=self.device),
                torch.empty(max_num_reqs, dtype=torch.long, device=self.device),
            )
            for _ in range(capacity)
        ]
        self.cursor = 0

    def pack(self, outputs: Mapping[str, torch.Tensor], final_rows: Sequence[int]) -> PackedOutputSnapshot:
        rows = tuple(int(row) for row in final_rows)
        if not rows or len(rows) > self.max_num_reqs:
            raise ValueError("Lychee decision batch exceeds the output ring capacity")
        unknown = set(outputs).difference(LYCHEE_OUTPUT_COLUMNS)
        if unknown or not outputs:
            raise ValueError(f"Unsupported Lychee decision columns: {sorted(unknown)}")
        for value in outputs.values():
            if (
                not isinstance(value, torch.Tensor)
                or value.ndim != 1
                or value.device != self.device
                or min(rows) < 0
                or max(rows) >= value.numel()
            ):
                raise ValueError("Lychee decision columns do not match captured model rows")
        slot_index = self.cursor
        slot = self.slots[slot_index]
        if slot.pending:
            raise RuntimeError("Lychee output ring has an unbound copy lease")
        stream = torch.cuda.current_stream(self.device) if self.device.type == "cuda" else None
        if stream is not None and slot.completion is not None:
            stream.wait_event(slot.completion)
        slot.pending = True
        slot.generation += 1
        slot.producer_stream = stream
        self.cursor = (slot_index + 1) % len(self.slots)
        # Index storage and decision storage share the same consumer fence.
        indices = slot.indices[: len(rows)]
        indices.copy_(torch.tensor(rows, dtype=torch.long, device="cpu"), non_blocking=True)
        payload = {}
        for field_index, key in enumerate(key for key in LYCHEE_OUTPUT_COLUMNS if key in outputs):
            column = slot.storage.narrow(0, field_index * len(rows), len(rows))
            source = outputs[key]
            if source.dtype != torch.int32:
                source = source.to(dtype=torch.int32)
            torch.index_select(source, 0, indices, out=column)
            payload[key] = column
        snapshot = pack_output_snapshot(payload, {}, max_buckets=0, reuse_existing_storage=True)
        if snapshot is None:
            raise RuntimeError("Lychee output slab could not use the packed copy contract")
        slot.snapshot = snapshot
        if stream is not None:
            try:
                snapshot.record_producer_event(stream)
            except Exception as exc:
                raise OutputCopyLifetimeError("Lychee packed decision readiness could not be fenced") from exc
        snapshot.set_copy_completion_callback(partial(self._bind_copy_event, slot_index, slot.generation))
        return snapshot

    def _bind_copy_event(self, slot_index: int, generation: int, event: Any) -> None:
        # Called once by the runner owner after queuing D2H. Do not synchronize,
        # throw, or consult live request/row state in this callback.
        slot = self.slots[slot_index]
        if not slot.pending or slot.generation != generation:
            logger.error("Ignoring stale Lychee output lease completion slot=%s generation=%s", slot_index, generation)
            return
        slot.completion = event
        slot.pending = False
        slot.snapshot = None
        slot.producer_stream = None

    def abort_unbound(self) -> None:
        """Fence preconstructor writes; preserve started or bound D2H leases."""
        for slot in self.slots:
            if not slot.pending or (slot.snapshot is not None and slot.snapshot.copy_started):
                continue
            if slot.producer_stream is not None:
                try:
                    event = torch.cuda.Event()
                    event.record(slot.producer_stream)
                except Exception as exc:
                    # A missing lifetime fence is an engine fault. Keep the
                    # slot unavailable instead of recycling unsafe storage.
                    raise OutputCopyLifetimeError("Lychee pre-copy slab writes could not be fenced") from exc
                slot.completion = event
            slot.pending = False
            slot.snapshot = None
            slot.producer_stream = None


__all__ = ["LYCHEE_OUTPUT_COLUMNS", "LycheeOutputRing"]
