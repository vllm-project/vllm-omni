# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from vllm_omni.diffusion.models.interface import ProxyStreamSpec


@dataclass(frozen=True)
class ProxyPacket:
    input_seq: int
    frame_start: int
    capture_ms: tuple[float, ...]
    eos: bool
    rgb: bytes
    num_frames: int

    @property
    def frame_count(self) -> int:
        return self.num_frames

    @classmethod
    def parse(cls, header: Mapping, rgb: object, *, spec: ProxyStreamSpec) -> "ProxyPacket":
        seq, start, eos = header.get("input_seq"), header.get("frame_start"), header.get("eos")
        if type(seq) is not int or seq < 0 or type(start) is not int or start < 0 or type(eos) is not bool:
            raise ValueError("Proxy input_seq and frame_start must be nonnegative integers and eos must be boolean")
        count = spec.tail_frames if eos else spec.packet_frames
        if type(header.get("num_frames")) is not int or header["num_frames"] != count:
            raise ValueError(f"Proxy packet requires {count} frames")
        timestamps = header.get("capture_ms")
        if (
            not isinstance(timestamps, (list, tuple))
            or len(timestamps) != count
            or any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in timestamps)
            or any(right <= left for left, right in zip(timestamps, timestamps[1:]))
        ):
            raise ValueError("Proxy capture_ms must contain one increasing timestamp per frame")
        if not isinstance(rgb, bytes) or len(rgb) != count * spec.width * spec.height * 3:
            raise ValueError("Proxy RGB24 packet byte length does not match its canvas and frame count")
        return cls(seq, start, tuple(timestamps), eos, rgb, count)

    def event(self) -> dict:
        return {
            "input_seq": self.input_seq,
            "frame_start": self.frame_start,
            "num_frames": self.frame_count,
            "capture_ms": self.capture_ms,
            "eos": self.eos,
            "rgb": self.rgb,
        }

    def as_frames(self, size: tuple[int, int]) -> np.ndarray:
        return np.frombuffer(self.rgb, dtype=np.uint8).reshape(self.frame_count, *size[::-1], 3)


class ProxySequence:
    def __init__(self, spec: ProxyStreamSpec) -> None:
        self.spec = spec
        self.next_seq = 0
        self.accepted = 0
        self.consumed = 0
        self.last_capture_ms = -1.0
        self.ended = False

    @property
    def credit(self) -> int:
        return self.spec.credit_frames - (self.accepted - self.consumed)

    def validate(self, packet: ProxyPacket) -> None:
        if self.ended or packet.input_seq != self.next_seq or packet.frame_start != self.accepted:
            raise ValueError("Proxy packet is out of sequence or follows EOS")
        if packet.eos and self.accepted < self.spec.bootstrap_packets * self.spec.packet_frames:
            raise ValueError(f"Proxy EOS requires at least {self.spec.bootstrap_packets} preceding full packets")
        if packet.capture_ms[0] <= self.last_capture_ms:
            raise ValueError("Proxy capture_ms must increase across packets")
        if packet.frame_count > self.credit:
            raise ValueError("Proxy queue is full. Wait for input credit before sending another packet.")

    def append(self, packet: ProxyPacket) -> None:
        self.validate(packet)
        self.next_seq += 1
        self.accepted += packet.frame_count
        self.last_capture_ms = packet.capture_ms[-1]
        self.ended = packet.eos

    def consume(self, frames: int) -> bool:
        if type(frames) is not int or not self.consumed <= frames <= self.accepted:
            raise RuntimeError("Invalid proxy consumption count")
        changed = self.consumed != frames
        self.consumed = frames
        return changed
