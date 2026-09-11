# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from typing import Protocol

from fastapi import WebSocket

from vllm_omni.entrypoints.realtime.session import RealtimeSessionProtocol


class RealtimeModelAdapter(Protocol):
    """Execute a session using the shared wire codec.

    The adapter owns admission, input delivery, model execution, cancellation,
    and cleanup. It binds the codec's sender and uses it to decode input and
    project output. The codec does not own a second execution state machine.

    The existing duplex handler is the first structural implementation. Its
    model-selected ServingRuntimeAdapter and native append path are unchanged;
    a turn-based model need not implement that native runtime contract.
    """

    async def handle_session(
        self,
        websocket: WebSocket,
        *,
        realtime_protocol: RealtimeSessionProtocol,
    ) -> None: ...
