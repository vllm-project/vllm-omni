# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import asyncio

from vllm_omni.diffusion.models.invoker_render.proxy import ProxyPacket, ProxySequence


class ProxyIngress:
    def __init__(self, size: tuple[int, int]) -> None:
        self.size = size
        self.sequence = ProxySequence()
        self.initial: list[dict] = []
        self.bootstrap_ready = asyncio.Event()
        self.engine_ready = asyncio.Event()

    def receive(self, header: dict, rgb: bytes) -> ProxyPacket | None:
        packet = ProxyPacket.parse(header, rgb, size=self.size)
        self.sequence.append(packet)
        if packet.input_seq < 2:
            self.initial.append(packet.event())
            if len(self.initial) == 2:
                self.bootstrap_ready.set()
            return None
        return packet

    def credit_message(self) -> dict:
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
                raise TimeoutError("Timed out waiting for two proxy packets")
        finally:
            for task in (ready, stopped):
                task.cancel()
            await asyncio.gather(ready, stopped, return_exceptions=True)
