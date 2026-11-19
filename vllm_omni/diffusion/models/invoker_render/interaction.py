# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from collections import deque
from collections.abc import Mapping

from vllm_omni.diffusion.interaction.modality_handlers.base import InteractionHandler
from vllm_omni.diffusion.interaction.types import InteractionChunkMetadata, InteractionPayload
from vllm_omni.diffusion.worker.utils import StepRequestState

from .proxy import PROXY_MODALITY, ProxyPacket, ProxySequence


class ProxyQueue:
    def __init__(self, size: tuple[int, int]) -> None:
        self.size = size
        self.sequence = ProxySequence()
        self.pending: deque[ProxyPacket] = deque()

    def packet(self, payload: InteractionPayload) -> ProxyPacket:
        if not isinstance(payload, Mapping):
            raise ValueError("Invoker-Render proxy packet must be an object")
        return ProxyPacket.parse(payload, payload.get("rgb"), size=self.size)

    def append(self, packet: ProxyPacket) -> None:
        self.sequence.append(packet)
        self.pending.append(packet)

    def take(self) -> ProxyPacket:
        packet = self.pending.popleft()
        self.sequence.consume(packet.frame_start + packet.frame_count)
        return packet


class InvokerProxyInteraction(InteractionHandler):
    modality = PROXY_MODALITY

    def validate_payload(
        self, state: StepRequestState, *, event_id: str, payload: InteractionPayload, transition_chunks: int | None
    ) -> None:
        if transition_chunks not in (None, 0):
            raise ValueError("Invoker-Render proxy packets cannot be interpolated")
        queue: ProxyQueue = state.extra[PROXY_MODALITY]
        queue.sequence.validate(queue.packet(payload))

    def enqueue(
        self,
        state: StepRequestState,
        *,
        event_id: str,
        received_at: float,
        payload: InteractionPayload,
        transition_chunks: int | None,
    ) -> None:
        queue: ProxyQueue = state.extra[PROXY_MODALITY]
        queue.append(queue.packet(payload))

    def apply_at_chunk_boundary(
        self,
        state: StepRequestState,
        *,
        boundary_at: float,
        chunk_index: int | None = None,
        num_media_frames: int | None = None,
        fps: float | None = None,
        num_latent_frames: int | None = None,
    ) -> InteractionChunkMetadata | None:
        # Proxy packets are queued in source-frame order and consumed when the model needs more input.
        return None
