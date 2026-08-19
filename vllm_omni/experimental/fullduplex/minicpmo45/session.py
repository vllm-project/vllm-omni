from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from vllm_omni.experimental.fullduplex.minicpmo45.input import (
    MiniCPMO45PcmAppendBuffer,
)


@dataclass(slots=True)
class MiniCPMO45ServingSessionState:
    """Mutable serving state owned by one MiniCPM duplex session."""

    audio_buffer: MiniCPMO45PcmAppendBuffer = field(default_factory=MiniCPMO45PcmAppendBuffer)
    input_since_commit: bool = False
    speech_since_commit: bool = False
    committed_audio_payload: dict[str, object] | None = None
    committed_audio_operation_id: str | None = None
    committed_audio_reserved_bytes: int = 0
    deferred_response_create: bool = False
    deferred_precreate_response: bool = False
    data_plane_task: asyncio.Task[None] | None = None
    data_plane_restart_requested: bool = False
    continuation_owner_id: str | None = None
    continuation_units: int = 0
    pending_silence_task: asyncio.Task[bool] | None = None
    pending_silence_owner_id: str | None = None
    silence_continuation_scheduler: Callable[..., Awaitable[bool]] | None = None
    accepted_input_epoch: int = -1
    accepted_input_seq: int = 0
    accepted_input_seqs: set[int] = field(default_factory=set)
    accepted_input_turns: dict[int, int] = field(default_factory=dict)
    final_input_epoch: int = -1
    final_input_seq: int = 0
    input_acceptances_inflight: int = 0
    final_input_acceptances_inflight: int = 0
    deferred_processed_inputs: dict[tuple[int, int], tuple[str, str | None]] = field(default_factory=dict)

    def begin_input_acceptance(self) -> None:
        self.input_acceptances_inflight += 1

    def cancel_input_acceptance(self) -> None:
        self.input_acceptances_inflight = max(0, self.input_acceptances_inflight - 1)

    def begin_final_input_acceptance(self) -> None:
        self.final_input_acceptances_inflight += 1

    def finish_final_input_acceptance(self) -> None:
        self.final_input_acceptances_inflight = max(0, self.final_input_acceptances_inflight - 1)

    def final_input_acceptance_pending(self) -> bool:
        return self.final_input_acceptances_inflight > 0

    def record_accepted_input(
        self,
        *,
        epoch: int,
        seq: int,
        model_turn_id: int | None = None,
    ) -> tuple[str, str | None] | None:
        if epoch != self.accepted_input_epoch:
            self.accepted_input_epoch = epoch
            self.accepted_input_seq = 0
            self.accepted_input_seqs.clear()
            self.accepted_input_turns.clear()
            self.deferred_processed_inputs = {
                key: value for key, value in self.deferred_processed_inputs.items() if key[0] == epoch
            }
        if seq > self.accepted_input_seq:
            self.accepted_input_seq = seq
        deferred = self.deferred_processed_inputs.pop((epoch, seq), None)
        if deferred is None:
            self.accepted_input_seqs.add(seq)
            if model_turn_id is not None:
                self.accepted_input_turns[seq] = int(model_turn_id)
        self.input_acceptances_inflight = max(0, self.input_acceptances_inflight - 1)
        return deferred

    def accepted_input_watermark(self, *, epoch: int) -> int | None:
        if epoch != self.accepted_input_epoch or self.accepted_input_seq <= 0:
            return None
        return self.accepted_input_seq

    def mark_input_processed(self, *, epoch: int, seq: int) -> bool:
        if seq <= 0 or epoch != self.accepted_input_epoch or seq not in self.accepted_input_seqs:
            return False
        # Removal deduplicates output and bounds state to the pending backlog.
        self.accepted_input_seqs.remove(seq)
        self.accepted_input_turns.pop(seq, None)
        return True

    def defer_input_processed(
        self,
        *,
        epoch: int,
        seq: int,
        outcome: str,
        response_id: str | None,
    ) -> bool:
        if (
            self.input_acceptances_inflight <= 0
            or epoch < self.accepted_input_epoch
            or (epoch == self.accepted_input_epoch and seq <= self.accepted_input_seq)
        ):
            return False
        self.deferred_processed_inputs.setdefault((epoch, seq), (outcome, response_id))
        return True

    def pending_input_identity(self, *, epoch: int) -> tuple[int, int] | None:
        if epoch != self.accepted_input_epoch:
            return None
        pending = [
            (seq, turn_id) for seq, turn_id in self.accepted_input_turns.items() if seq in self.accepted_input_seqs
        ]
        return max(pending) if pending else None

    def promote_pending_input_turn(self, *, epoch: int, seq: int, model_turn_id: int) -> None:
        if epoch == self.accepted_input_epoch and seq in self.accepted_input_seqs:
            current = self.accepted_input_turns.get(seq)
            if current is not None and current < model_turn_id:
                self.accepted_input_turns[seq] = model_turn_id

    def record_final_input(self, *, epoch: int, seq: int) -> None:
        if seq > 0:
            self.final_input_epoch = epoch
            self.final_input_seq = seq

    def final_input_identity(self, *, epoch: int) -> int | None:
        if epoch != self.final_input_epoch or self.final_input_seq <= 0:
            return None
        return self.final_input_seq

    def resolve_final_input(self, *, epoch: int, seq: int) -> None:
        if epoch == self.final_input_epoch and seq == self.final_input_seq:
            self.final_input_epoch = -1
            self.final_input_seq = 0

    def retain_committed_audio(
        self,
        payload: dict[str, object],
        *,
        operation_id: str | None,
        reserved_bytes: int = 0,
    ) -> None:
        self.committed_audio_payload = payload
        self.committed_audio_operation_id = operation_id
        self.committed_audio_reserved_bytes += max(0, int(reserved_bytes))

    def clear_committed_audio(self) -> int:
        reserved_bytes = self.committed_audio_reserved_bytes
        self.committed_audio_payload = None
        self.committed_audio_operation_id = None
        self.committed_audio_reserved_bytes = 0
        self.deferred_response_create = False
        self.deferred_precreate_response = False
        return reserved_bytes

    def clear_continuation(self) -> None:
        self.continuation_owner_id = None
        self.continuation_units = 0
        self.pending_silence_task = None
        self.pending_silence_owner_id = None
