# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from vllm_omni.diffusion.interaction.proxy import ProxyPacket, ProxySequence
from vllm_omni.diffusion.models.interface import ProxyStreamSpec

from .assets import FPS, PACKET_FRAMES, TAIL_FRAMES

PROXY_MODALITY = "invoker_proxy"
BOOTSTRAP_KEY = "invoker_render_bootstrap"
BOOTSTRAP_PACKETS = 2
PROXY_FRAME_LIMIT = 3 * PACKET_FRAMES + TAIL_FRAMES


def invoker_render_proxy_spec(size: tuple[int, int]) -> ProxyStreamSpec:
    return ProxyStreamSpec(
        width=size[0],
        height=size[1],
        fps=FPS,
        packet_frames=PACKET_FRAMES,
        tail_frames=TAIL_FRAMES,
        credit_frames=PROXY_FRAME_LIMIT,
        bootstrap_packets=BOOTSTRAP_PACKETS,
        modality=PROXY_MODALITY,
        bootstrap_key=BOOTSTRAP_KEY,
    )


__all__ = [
    "BOOTSTRAP_KEY",
    "BOOTSTRAP_PACKETS",
    "PROXY_FRAME_LIMIT",
    "PROXY_MODALITY",
    "ProxyPacket",
    "ProxySequence",
    "invoker_render_proxy_spec",
]
