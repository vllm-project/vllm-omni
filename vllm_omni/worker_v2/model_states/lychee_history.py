# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Explicit model-position evidence for native Lychee prefill and rebuilding."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any


@dataclass(frozen=True)
class LycheePromptHistory:
    text: tuple[int, ...]
    speech: tuple[int | None, ...]
    control: tuple[int | None, ...]
    logical_ticks: tuple[int, ...]
    audio_windows: tuple[dict[str, Any], ...]
    execution_epoch: int
    raw_text: tuple[int | None, ...]
    raw_speech: tuple[int | None, ...]
    raw_control: tuple[int | None, ...]
    force_listen_at_frontier: bool

    @classmethod
    def from_payload(cls, payload: Any, *, window_ticks: int, vocab_size: int | None = None) -> LycheePromptHistory:
        if not isinstance(payload, dict) or type(payload.get("version")) is not int or payload.get("version") != 1:
            raise ValueError("Lychee history requires explicit version=1 model-position evidence")

        def channel(name: str, nullable: bool = False) -> tuple:
            values = payload.get(name)
            if not isinstance(values, (list, tuple)) or not values:
                raise ValueError(f"Lychee history {name} must be a nonempty token list")
            if any((value is not None or not nullable) and type(value) is not int for value in values):
                raise ValueError(f"Lychee history {name} contains an invalid token")
            if name != "logical_ticks" and any(value is not None and value < 0 for value in values):
                raise ValueError(f"Lychee history {name} contains a negative token")
            if (
                name != "logical_ticks"
                and vocab_size is not None
                and any(value is not None and value >= vocab_size for value in values)
            ):
                raise ValueError(f"Lychee history {name} contains an out-of-vocabulary token")
            return tuple(values)

        text = channel("text_input_ids")
        speech = channel("speech_input_ids", True)
        control = channel("control_input_ids", True)
        ticks = channel("logical_ticks")
        if any(len(values) != len(text) for values in (speech, control, ticks)):
            raise ValueError("Lychee history must align all three channels and logical ticks to model positions")
        prefix = len(ticks) - sum(tick >= 0 for tick in ticks)
        if ticks[:prefix] != (-1,) * prefix or ticks[prefix:] != tuple(range(len(ticks) - prefix)):
            raise ValueError("Lychee history logical ticks must be prefix -1 followed by contiguous ticks from zero")
        if not ticks or ticks[-1] < 0:
            raise ValueError("Lychee history must contain a streaming frontier")
        for position, tick in enumerate(ticks):
            if tick > 0 and (speech[position] is None or control[position] is None):
                raise ValueError("Lychee replay is missing committed speech/control evidence")
        epoch = payload.get("execution_epoch")
        if type(epoch) is not int or epoch < 0:
            raise ValueError("Lychee history execution_epoch must be a nonnegative integer")
        windows = payload.get("audio_windows")
        if not isinstance(windows, list) or not windows:
            raise ValueError("Lychee replay requires the original audio windows")
        seen_sequences: set[int] = set()
        starts: set[int] = set()
        for window in windows:
            if not isinstance(window, dict):
                raise ValueError("Invalid Lychee replay audio window")
            seq, start = window.get("seq"), window.get("start_tick")
            if type(seq) is not int or seq <= 0 or seq in seen_sequences:
                raise ValueError("Lychee replay audio sequence must be positive and unique")
            if type(start) is not int or start < 0 or start % window_ticks:
                raise ValueError("Lychee replay audio starts must align to complete windows")
            cutoff = window.get("discard_after_tick")
            if "discard_after_tick" in window and (
                type(cutoff) is not int or not start - 1 <= cutoff < start + window_ticks
            ):
                raise ValueError("Lychee audio discard_after_tick must bound this window's absolute input ticks")
            if "payload" not in window:
                raise ValueError("Lychee replay audio window lacks its original PCM payload")
            if start in starts:
                raise ValueError("Lychee replay audio windows overlap")
            seen_sequences.add(seq)
            starts.add(start)
        for tick in ticks[prefix:]:
            if tick // window_ticks * window_ticks not in starts:
                raise ValueError(f"Lychee replay audio has a gap or overlap at logical tick {tick}")
        raw_channels = []
        for name, fallback in (
            ("raw_text_output_ids", text),
            ("raw_speech_output_ids", speech),
            ("raw_control_output_ids", control),
        ):
            values = channel(name, True) if name in payload else fallback
            if len(values) != len(text) or any(
                values[position] is None for position, tick in enumerate(ticks) if tick > 0
            ):
                raise ValueError("Lychee raw sampled history must align with every committed logical tick")
            raw_channels.append(values)
        force_listen = payload.get("force_listen_at_frontier", False)
        if type(force_listen) is not bool:
            raise ValueError("Lychee force_listen_at_frontier must be a bool")
        return cls(
            text,
            speech,
            control,
            ticks,
            tuple(windows),
            epoch,
            raw_channels[0],
            raw_channels[1],
            raw_channels[2],
            force_listen,
        )

    @property
    def prefix_len(self) -> int:
        return self.logical_ticks.index(0)

    @cached_property
    def audio_by_start(self) -> dict[int, dict[str, Any]]:
        return {window["start_tick"]: window for window in self.audio_windows}


@dataclass(frozen=True)
class LycheeResidentHistory:
    """A retained binding needs only its prefix and two adjacent PCM windows."""

    prefix_len: int
    execution_epoch: int
    session_epoch: int
    op_seq: int
    audio_window_seq: int
    audio_by_start: dict[int, dict[str, Any]]

    @classmethod
    def from_bootstrap(cls, history: LycheePromptHistory, *, op_seq: int, session_epoch: int) -> LycheeResidentHistory:
        if (
            type(op_seq) is not int
            or op_seq <= 0
            or type(session_epoch) is not int
            or session_epoch != history.execution_epoch
        ):
            raise ValueError("Lychee full bootstrap has an invalid owner epoch or operation sequence")
        latest = max(history.audio_windows, key=lambda window: window["seq"])
        windows = {
            window["start_tick"]: window for window in history.audio_windows if window["seq"] >= latest["seq"] - 1
        }
        return cls(history.prefix_len, history.execution_epoch, session_epoch, op_seq, latest["seq"], windows)

    def append_packet(self, duplex: dict[str, Any], *, req_id: str, window_ticks: int) -> LycheeResidentHistory:
        header = duplex.get("lychee_audio_delta")
        if not isinstance(header, dict) or header.get("kind") != "resident_append":
            raise ValueError("Lychee audio delta requires kind=resident_append; full rebuild required")
        expected = {
            "version": 1,
            "session_epoch": self.session_epoch,
            "execution_epoch": self.execution_epoch,
            "op_seq": self.op_seq + 1,
            "previous_audio_window_seq": self.audio_window_seq,
            "audio_window_seq": self.audio_window_seq + 1,
            "start_tick": self.audio_window_seq * window_ticks,
            "window_ticks": window_ticks,
        }
        if header.get("request_id") != req_id:
            raise ValueError("Lychee audio delta owner mismatch; full rebuild required")
        if any(type(header.get(key)) is not int or header[key] != value for key, value in expected.items()):
            raise ValueError("Lychee audio delta epoch/sequence/window mismatch; full rebuild required")
        if (
            type(duplex.get("seq")) is not int
            or type(duplex.get("epoch")) is not int
            or duplex["seq"] != header["op_seq"]
            or duplex["epoch"] != self.session_epoch
        ):
            raise ValueError("Lychee audio delta does not match its request envelope; full rebuild required")
        payload = duplex.get("payload")
        ledger = payload.get("lychee_audio_ledger") if isinstance(payload, dict) else None
        if not isinstance(ledger, dict) or any(
            type(ledger.get(key)) is not int or ledger[key] != value
            for key, value in {
                "audio_window_seq": header["audio_window_seq"],
                "consumable_tick_start": header["start_tick"],
                "consumable_tick_end": header["start_tick"] + window_ticks,
            }.items()
        ):
            raise ValueError("Lychee audio delta lacks matching absolute PCM ledger evidence; full rebuild required")
        window = dict(seq=header["audio_window_seq"], start_tick=header["start_tick"], payload=payload)
        previous_start = header["start_tick"] - window_ticks
        if previous_start not in self.audio_by_start:
            raise ValueError("Lychee retained audio frontier is missing; full rebuild required")
        windows = {previous_start: self.audio_by_start[previous_start], header["start_tick"]: window}
        return type(self)(
            self.prefix_len,
            self.execution_epoch,
            self.session_epoch,
            header["op_seq"],
            header["audio_window_seq"],
            windows,
        )
