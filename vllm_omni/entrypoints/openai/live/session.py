# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Per-session state and the conversation history owned by the API server."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from openai.types.live import SessionResource

from vllm_omni.entrypoints.openai.live.audio import LiveAudioFormat
from vllm_omni.entrypoints.openai.live.protocol import new_id


@dataclass
class SystemItem:
    """Developer context: ``instructions`` and ``input`` developer messages."""

    text: str


@dataclass
class UserTextItem:
    text: str


@dataclass
class AssistantItem:
    """Assistant history supplied as text in ``input``."""

    text: str


@dataclass
class NativeUnitItem:
    """One unit of input the model consumed, and what it produced for it.

    ``audio`` is the unit's input at the model's input rate; ``data`` holds
    processor-specific inputs (e.g. streaming mel features); ``token_ids`` is
    the stage-0 output for the unit.
    """

    audio: np.ndarray
    start_ms: int
    end_ms: int
    token_ids: list[int] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)


HistoryItem = SystemItem | UserTextItem | AssistantItem | NativeUnitItem


@dataclass
class LiveSessionState:
    model: str
    audio_format: LiveAudioFormat
    expires_at: int
    voice: Any = None  # as supplied by the client, echoed in the resource
    instructions: str | None = None
    input: list[dict[str, Any]] | None = None
    delegation: dict[str, Any] | None = None
    id: str = field(default_factory=lambda: new_id("sess"))
    started_monotonic: float = field(default_factory=time.monotonic)

    # Timeline and usage.
    session_ms: float = 0.0
    input_audio_ms: float = 0.0
    output_audio_ms: float = 0.0
    muted: bool = False

    # Conversation history as rendered into the model.
    history: list[HistoryItem] = field(default_factory=list)

    @property
    def now_ms(self) -> int:
        return int(self.session_ms)

    @property
    def usage_seconds(self) -> float:
        return round((self.input_audio_ms + self.output_audio_ms) / 1000.0, 3)

    @property
    def units(self) -> list[NativeUnitItem]:
        return [item for item in self.history if isinstance(item, NativeUnitItem)]

    def to_resource(self) -> SessionResource:
        audio: dict[str, Any] = {"format": self.audio_format.to_dict()}
        if self.voice is not None:
            audio["output"] = {"voice": self.voice}
        return SessionResource(
            id=self.id,
            expires_at=self.expires_at,
            model=self.model,
            status="active",
            audio=audio,
            client=None,
            delegation=self.delegation,
            input=self.input,
            instructions=self.instructions,
            store=False,
        )
