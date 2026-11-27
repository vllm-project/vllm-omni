# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""The OpenAI Realtime error envelope.

A Realtime server reports a rejected client event as one ``error`` event whose
``error`` object carries ``type`` / ``code`` / ``message`` (plus the client's
``event_id`` and an optional ``param``). ``type`` is not free-form: it is one
of OpenAI's four buckets, derived from our internal code by
:data:`REALTIME_ERROR_TYPES_BY_CODE`.

OpenAI standardises the three ``type`` classes but not the *codes*, so the code
vocabulary is a consumer's own. vLLM-Omni's lives in
``vllm_omni.protocol.duplex.errors`` (Tier 3 in
``docs/serving/realtime_duplex_api.md``); what stays here is the exception type
every consumer raises, so the envelope is rendered identically whoever built it.
"""

from __future__ import annotations

__all__ = ["RealtimeProtocolError"]


class RealtimeProtocolError(ValueError):
    """A client payload could not be decoded into a valid Realtime intent.

    ``code`` is the internal error code that :data:`REALTIME_ERROR_TYPES_BY_CODE`
    maps to an OpenAI ``error.type``; ``event_id`` is the *client* event id the
    error answers. ``vllm_omni.engine.duplex.commands.DuplexCommandError`` is
    the duplex specialization, so a consumer catching either sees the same
    three attributes.
    """

    def __init__(self, message: str, *, code: str = "bad_event", event_id: str | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.event_id = event_id
