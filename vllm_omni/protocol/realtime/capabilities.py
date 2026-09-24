# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""What a Realtime consumer can serve, and the session check that reads it.

``session.update`` is the one client event whose acceptance depends on who is
behind the socket. The audio formats a server can decode, and whether it can do
server-side turn detection at all, are properties of the *consumer* --- the
duplex engine, or some other runtime --- not of the protocol. The codec still
has to do the checking, because it owns the error envelope and the order the
checks run in.

:class:`RealtimeProtocolCapabilities` is how a consumer states those answers.
It is the post-#7413 form of the ``supports(turn_detection, formats)`` half of
RFC #6592's proposed ``RealtimeModelAdapter``. The other three halves of that
proposal --- building a prompt, starting a response, cancelling one --- are not
here and must not come here: for a duplex model they are already owned by
``DuplexModelPlugin`` together with ``DuplexSessionRunner``, and a second
interface over the same responsibility is exactly the fork this layer exists to
prevent.

The duplex binding of this object is
``vllm_omni.engine.duplex.mailbox.DUPLEX_REALTIME_CAPABILITIES``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from vllm_omni.protocol.realtime.formats import (
    REALTIME_INPUT_AUDIO_FORMATS,
    REALTIME_OUTPUT_AUDIO_FORMATS,
    validate_realtime_session_audio_formats,
)

__all__ = [
    "RealtimeProtocolCapabilities",
    "RealtimeSessionRejection",
    "validate_session_payload",
]

#: A consumer's answer to "can you do this turn detection?": an error message, or None.
TurnDetectionValidator = Callable[[Mapping[str, object]], str | None]


@dataclass(frozen=True, slots=True)
class RealtimeProtocolCapabilities:
    """One consumer's answer to what a ``session`` object may ask for.

    The defaults are permissive, not restrictive: every format the codec can
    decode, and **no turn-detection validation at all**. A consumer that leaves
    ``validate_turn_detection`` at ``None`` therefore accepts whatever
    ``turn_detection`` the session object carries --- including values it cannot
    actually serve, such as ``semantic_vad``. A consumer that supports only some
    turn-detection modes (or none) must supply its own validator; see
    ``vllm_omni.engine.duplex.mailbox.DUPLEX_REALTIME_CAPABILITIES``.
    """

    input_audio_formats: frozenset[str] = field(default_factory=lambda: frozenset(REALTIME_INPUT_AUDIO_FORMATS))
    output_audio_formats: frozenset[str] = field(default_factory=lambda: frozenset(REALTIME_OUTPUT_AUDIO_FORMATS))
    #: Consumer-specific ``turn_detection`` validation. ``None`` means the
    #: consumer takes whatever the session object says without a check of its own.
    validate_turn_detection: TurnDetectionValidator | None = None


@dataclass(frozen=True, slots=True)
class RealtimeSessionRejection:
    """Why a session object was refused: the internal error code and the message.

    ``param`` names the offending session field where there is one, for the
    ``error.param`` slot of the OpenAI error envelope.
    """

    code: str
    message: str
    param: str | None = None


def validate_session_payload(
    session_payload: Mapping[str, object],
    *,
    capabilities: RealtimeProtocolCapabilities,
) -> RealtimeSessionRejection | None:
    """Check a ``session`` object against one consumer; ``None`` when it is acceptable.

    Audio formats are checked before turn detection, so a session object that
    is wrong in both ways reports the format problem --- keep that order, it is
    what clients already see.
    """
    format_error = validate_realtime_session_audio_formats(
        session_payload,
        input_audio_formats=capabilities.input_audio_formats,
        output_audio_formats=capabilities.output_audio_formats,
    )
    if format_error is not None:
        return RealtimeSessionRejection(code="unsupported_audio_format", message=format_error)
    validate_turn_detection = capabilities.validate_turn_detection
    if validate_turn_detection is not None:
        turn_detection_error = validate_turn_detection(session_payload)
        if turn_detection_error is not None:
            return RealtimeSessionRejection(
                code="unsupported_turn_detection", message=turn_detection_error, param="turn_detection"
            )
    return None
