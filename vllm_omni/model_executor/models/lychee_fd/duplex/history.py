# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Committed channel and PCM evidence used to rebuild native paged KV."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field

from .codec import aligned_values


@dataclass(slots=True)
class LycheeSessionHistory:
    prefix: list[int]
    text_pad: int = 158358
    speech_pad: int = 158359
    sleep: int = 158357
    speech_eos: int = 151694
    text: list[int] = field(default_factory=list)
    speech: list[int | None] = field(default_factory=list)
    control: list[int | None] = field(default_factory=list)
    ticks: list[int] = field(default_factory=list)
    raw_text: list[int | None] = field(default_factory=list)
    raw_speech: list[int | None] = field(default_factory=list)
    raw_control: list[int | None] = field(default_factory=list)
    audio_windows: list[dict[str, object]] = field(default_factory=list)
    request_ids: set[str] = field(default_factory=set)
    bound_request_id: str | None = None
    bound_execution_epoch: int | None = None
    frontier_tick: int = 0
    execution_epoch: int = 0
    force_listen_at_frontier: bool = False
    last_audio_operation: tuple[int, int] | None = None
    pending_eos_rebuild_tick: int | None = None

    def __post_init__(self):
        self.text = self.prefix + [self.text_pad]
        # Captured reference fusion pads BEFORE singleton side streams:
        # system rows are zero and tick zero owns speech/control/audio inputs.
        self.speech = [None] * len(self.prefix) + [self.speech_pad]
        self.control = [None] * len(self.prefix) + [self.sleep]
        self.ticks = [-1] * len(self.prefix) + [0]
        self.raw_text = [None] * len(self.text)
        self.raw_speech = [None] * len(self.text)
        self.raw_control = [None] * len(self.text)

    def has_resident_binding(self, *, request_id: str, execution_epoch: int) -> bool:
        return (
            request_id in self.request_ids
            and self.bound_request_id == request_id
            and self.bound_execution_epoch == execution_epoch
        )

    def bind_request(self, *, request_id: str, execution_epoch: int) -> None:
        # Only accepted appends establish residency. The data plane removes
        # retired ids from request_ids; these two scalar markers stay bounded.
        self.request_ids.add(request_id)
        self.bound_request_id = request_id
        self.bound_execution_epoch = execution_epoch

    def append_audio(self, payload: Mapping[str, object], *, epoch: int, seq: int) -> dict[str, object]:
        operation = (epoch, seq)
        if operation == self.last_audio_operation:
            return deepcopy(self.audio_windows[-1]["payload"])
        copied = deepcopy(dict(payload))
        ledger = copied.get("lychee_audio_ledger")
        if not isinstance(ledger, Mapping):
            raise ValueError("Lychee rebuild requires an audio ledger")
        start = len(self.audio_windows) * 10
        ledger = dict(ledger)
        ledger.update(
            consumable_tick_start=start, consumable_tick_end=start + 10, audio_window_seq=len(self.audio_windows) + 1
        )
        copied["lychee_audio_ledger"] = ledger
        self.audio_windows.append({"seq": len(self.audio_windows) + 1, "start_tick": start, "payload": copied})
        self.last_audio_operation = operation
        return copied

    def discard_pending_audio(self) -> None:
        # Output frontier N has committed input/KV only through tick N-1.
        # Keep original PCM for those rows; future rows use AUDIO_PAD on replay.
        cutoff = self.frontier_tick - 1
        for window in self.audio_windows:
            start = int(window["start_tick"])
            if start + 9 <= cutoff:
                continue
            previous = window.get("discard_after_tick", start + 9)
            window["discard_after_tick"] = min(int(previous), max(start - 1, cutoff))

    def snapshot(self, *, execution_epoch: int) -> dict[str, object]:
        return {
            "version": 1,
            "text_input_ids": list(self.text),
            "speech_input_ids": list(self.speech),
            "control_input_ids": list(self.control),
            "logical_ticks": list(self.ticks),
            "raw_text_output_ids": list(self.raw_text),
            "raw_speech_output_ids": list(self.raw_speech),
            "raw_control_output_ids": list(self.raw_control),
            "audio_windows": deepcopy(self.audio_windows),
            "execution_epoch": execution_epoch,
            "force_listen_at_frontier": self.force_listen_at_frontier,
        }

    def record_outputs(self, payload: Mapping[str, object]) -> None:
        ticks = aligned_values(payload, "lychee_tick")
        if not ticks:
            return
        texts = aligned_values(payload, "lychee_text_token_ids", length=len(ticks))
        speeches = aligned_values(payload, "lychee_speech_token_ids", length=len(ticks))
        controls = aligned_values(payload, "lychee_control_token_ids", length=len(ticks))
        epochs = aligned_values(payload, "lychee_execution_epoch", length=len(ticks))
        next_texts = aligned_values(payload, "lychee_next_text_token_ids") or texts
        next_speeches = aligned_values(payload, "lychee_next_speech_token_ids") or speeches
        next_controls = aligned_values(payload, "lychee_next_control_token_ids") or controls
        if any(len(values) != len(ticks) for values in (next_texts, next_speeches, next_controls)):
            raise ValueError("Lychee next-input histories do not align to output ticks")
        for index, (tick, text, speech, control, epoch) in enumerate(zip(ticks, texts, speeches, controls, epochs)):
            if tick < 0 or tick <= self.frontier_tick:
                continue
            if tick != self.frontier_tick + 1:
                raise ValueError(f"Lychee committed history gap: {self.frontier_tick} -> {tick}")
            self.text.append(next_texts[index])
            self.speech.append(next_speeches[index])
            self.control.append(next_controls[index])
            self.raw_text.append(text)
            self.raw_speech.append(speech)
            self.raw_control.append(control)
            self.ticks.append(tick)
            self.frontier_tick = tick
            self.execution_epoch = epoch
            if speech == self.speech_eos:
                # The released AR request ends at speech EOS. Rebuild its KV
                # at the next input boundary while the waveform owner drains.
                self.pending_eos_rebuild_tick = tick
