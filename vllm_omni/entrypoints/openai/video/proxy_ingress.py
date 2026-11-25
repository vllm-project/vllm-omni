# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import asyncio
from typing import Any, cast

from vllm_omni.diffusion.interaction.proxy import ProxyPacket, ProxySequence
from vllm_omni.diffusion.models.interface import ProxyStreamSpec
from vllm_omni.inputs.data import OmniInteractionEvent, OmniInteractionPrompt


class ProxyIngress:
    def __init__(self, spec: ProxyStreamSpec) -> None:
        self.spec = spec
        self.sequence = ProxySequence(spec)
        self.initial: list[dict[str, Any]] = []
        self.bootstrap_ready = asyncio.Event()
        self.engine_ready = asyncio.Event()

    @property
    def size(self) -> tuple[int, int]:
        return self.spec.size

    def start_metadata(self) -> dict[str, int]:
        return {
            "width": self.spec.width,
            "height": self.spec.height,
            "fps": self.spec.fps,
            "packet_frames": self.spec.packet_frames,
            "tail_frames": self.spec.tail_frames,
            "credit_frames": self.spec.credit_frames,
        }

    def validate_start(
        self, *, num_frames: int | None, fps: int | float | None, streaming_buffer_seconds: float | None
    ) -> str | None:
        if streaming_buffer_seconds is not None or num_frames is not None:
            return "Proxy sessions use source-driven pacing and finish at EOS"
        if fps not in (None, self.spec.fps):
            return f"Proxy sessions use {self.spec.fps} FPS"
        return None

    def receive(self, header: dict[str, Any], rgb: bytes) -> ProxyPacket | None:
        packet = ProxyPacket.parse(header, rgb, spec=self.spec)
        self.sequence.append(packet)
        if packet.input_seq < self.spec.bootstrap_packets:
            self.initial.append(packet.event())
            if len(self.initial) == self.spec.bootstrap_packets:
                self.bootstrap_ready.set()
            return None
        return packet

    def bootstrap_media(self) -> dict[str, list[dict[str, Any]]]:
        packets = self.initial
        self.initial = []
        return {self.spec.bootstrap_key: packets}

    def interaction_prompt(self, packet: ProxyPacket) -> OmniInteractionPrompt:
        return {
            "event_id": f"proxy-{packet.input_seq}",
            "event": cast(OmniInteractionEvent, {"multi_modal_data": {self.spec.modality: packet.event()}}),
        }

    def credit_message(self) -> dict[str, Any]:
        return {
            "type": "session.proxy.credit",
            "next_input_seq": self.sequence.next_seq,
            "accepted_frames": self.sequence.accepted,
            "consumed_frames": self.sequence.consumed,
            "credit_frames": self.sequence.credit,
        }

    async def wait_for_bootstrap(self, stop: asyncio.Event, timeout: float) -> None:
        ready = asyncio.create_task(self.bootstrap_ready.wait())
        stopped = asyncio.create_task(stop.wait())
        try:
            await asyncio.wait({ready, stopped}, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
            if stop.is_set():
                raise RuntimeError("Proxy session stopped during bootstrap")
            if not self.bootstrap_ready.is_set():
                raise TimeoutError(f"Timed out waiting for {self.spec.bootstrap_packets} proxy packets")
        finally:
            for task in (ready, stopped):
                task.cancel()
            await asyncio.gather(ready, stopped, return_exceptions=True)
