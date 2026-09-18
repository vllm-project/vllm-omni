# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""The client-event vocabulary a full-duplex session accepts.

Seven of the seventeen are events OpenAI's Realtime API has no equivalent for,
because they only make sense when the model and the user can talk at the same
time: the user cuts in (``barge_in``), the client reports how much audio it
played (``playback.ack``), a turn transition is signalled locally
(``turn.signal``), input is abandoned without committing (``input.cancel``),
text is pushed into a live turn (``input.text.append``), and the session is
kept alive or closed explicitly (``session.heartbeat`` / ``session.close``).
Two more are OpenAI commands carrying duplex-only fields (Tier 2).

The remaining eight are OpenAI's, unchanged, and re-exported from
``vllm_omni.protocol.realtime.commands`` rather than redeclared --- they are
the same objects. They are re-exported so this module is the *whole*
vocabulary and a duplex consumer never has to import the Tier 1 package
directly; keeping them declared there means a GA-only consumer is still not
handed events its clients never send.

None of these carries the engine's mailbox rendering: ``payload()`` and its
channel live in ``vllm_omni.engine.duplex.commands``, because for four of the
seventeen the runner's channel is not the client event type at all.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import ClassVar

from vllm_omni.protocol.realtime import commands as realtime_commands
from vllm_omni.protocol.realtime.commands import (
    CancelResponse,
    ClearInput,
    ClearOutputAudio,
    CreateItem,
    CreateResponse,
    DeleteItem,
    RealtimeCommand,
    TruncateItem,
    UpdateSession,
)

# ---- Tier 2: OpenAI names carrying vLLM-Omni extensions ----


@dataclass(frozen=True, slots=True, kw_only=True)
class AppendAudio(realtime_commands.AppendAudio):
    """Tier 2: OpenAI's ``input_audio_buffer.append`` plus the duplex hints.

    OpenAI's append is just base64 audio. The duplex lane also lets a client
    declare this chunk's own format and rate, whether it believes the chunk is
    speech, camera frames captured alongside it, and its timing --- all
    additive, all ignored by a stock client.
    """

    #: Model-neutral hints carried through from the wire (rms, vad, transcript hints ...).
    format: str = "pcm16"
    sample_rate_hz: int | None = None
    is_speech: bool | None = None
    video_frames: tuple[str, ...] = ()
    duration_ms: int | None = None
    audio_end_ms: int | None = None
    hints: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True, kw_only=True)
class Commit(realtime_commands.Commit):
    """Tier 2: OpenAI's ``input_audio_buffer.commit`` plus duplex turn control.

    OpenAI's commit carries nothing and always produces an item. Here a commit
    may be non-final, may decline to start a response, and may declare itself
    silence --- the semantic divergence documented in
    ``docs/serving/realtime_duplex_api.md`` ("Commit != response").
    """

    #: Realtime conversation item created for this commit (wire correlation only).
    #: ``None`` means "no explicit request": the runner decides on commit (auto-response
    #: sessions answer on their own); ``True`` / ``False`` force it.
    create_response: bool | None = None
    is_speech: bool | None = None
    realtime_item_id: str | None = None


# ---- Tier 3: vLLM-Omni only ----


@dataclass(frozen=True, slots=True, kw_only=True)
class AppendText(RealtimeCommand):
    wire_type: ClassVar[str] = "input.text.append"
    text: str


@dataclass(frozen=True, slots=True, kw_only=True)
class CancelInput(RealtimeCommand):
    wire_type: ClassVar[str] = "input.cancel"


@dataclass(frozen=True, slots=True, kw_only=True)
class BargeIn(RealtimeCommand):
    wire_type: ClassVar[str] = "barge_in"


@dataclass(frozen=True, slots=True, kw_only=True)
class SignalTurn(RealtimeCommand):
    """Generic ``turn.signal`` (local turn transitions such as ``user_started``)."""

    wire_type: ClassVar[str] = "turn.signal"
    event: str
    signal_payload: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True, kw_only=True)
class AckPlayback(RealtimeCommand):
    wire_type: ClassVar[str] = "playback.ack"
    played_ms: int
    committed_ms: int | None = None
    response_id: str | None = None
    item_id: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class Heartbeat(RealtimeCommand):
    wire_type: ClassVar[str] = "session.heartbeat"


@dataclass(frozen=True, slots=True, kw_only=True)
class CloseSession(RealtimeCommand):
    """Graceful close requested through the command stream (``session.close``)."""

    wire_type: ClassVar[str] = "session.close"
    reason: str = "client_close"


#: The complete command vocabulary a duplex client may send: the eight Tier 1
#: classes re-exported unchanged, the two Tier 2 classes defined above, and the
#: seven Tier 3 ones. A duplex consumer imports from here and never reaches
#: past this module into ``vllm_omni.protocol.realtime``, so a command that
#: later grows a duplex extension changes only this file.
__all__ = [
    # Tier 1 --- re-exported from ``vllm_omni.protocol.realtime.commands``.
    "CancelResponse",
    "ClearInput",
    "ClearOutputAudio",
    "CreateItem",
    "CreateResponse",
    "DeleteItem",
    "RealtimeCommand",
    "TruncateItem",
    "UpdateSession",
    # Tier 2 --- an OpenAI command plus duplex-only fields.
    "AppendAudio",
    "Commit",
    # Tier 3 --- vLLM-Omni only.
    "AckPlayback",
    "AppendText",
    "BargeIn",
    "CancelInput",
    "CloseSession",
    "Heartbeat",
    "SignalTurn",
]
