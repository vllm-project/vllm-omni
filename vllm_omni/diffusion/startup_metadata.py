# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Retain optional facts from the existing, upstream-owned READY handshake.

Socket.shadow borrows the same ZeroMQ socket: no extra connection, receive, or
RPC is introduced. The upstream startup loop still owns sequencing, polling,
process-failure handling and state transitions. The original socket context
owns closure; a shadow socket must not be explicitly closed.
"""

from typing import Any

import msgspec
import zmq

from vllm_omni.request_capabilities import RuntimeServingMetadata


class DiffusionStartupSocket(zmq.Socket):
    serving_metadata: RuntimeServingMetadata = RuntimeServingMetadata()

    def recv_multipart(self, flags: int = 0, copy: bool = True, track: bool = False) -> list[Any]:
        frames = super().recv_multipart(flags=flags, copy=copy, track=track)
        if len(frames) == 2:
            message = msgspec.msgpack.decode(bytes(frames[1]))
            if isinstance(message, dict) and message.get("status") == "READY":
                value = message.get("serving_metadata")
                self.serving_metadata = (
                    RuntimeServingMetadata.model_validate(value) if value is not None else RuntimeServingMetadata()
                )
        return frames
