# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Stage1-local SessionHistory for duplex AURA (import from asr2aura / aura2tts)."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock

_DEFAULT_MAX_TURNS = 16
_STORE: dict[str, SessionHistory] = {}
_LOCK = Lock()


@dataclass
class SessionHistory:
    """Per-session chat turns held in the Stage1 process."""

    session_id: str
    max_turns: int = _DEFAULT_MAX_TURNS
    messages: list[dict[str, str]] = field(default_factory=list)
    pending_user: str | None = None

    def render_prefix(self) -> str:
        rendered: list[str] = []
        for message in self.messages:
            role = message.get("role")
            content = message.get("content")
            if role not in {"user", "assistant"} or not isinstance(content, str) or not content:
                continue
            rendered.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
        return "".join(rendered)

    def begin_user_turn(self, transcript: str) -> None:
        self.pending_user = transcript.strip()

    def commit_turn(self, assistant_text: str) -> None:
        user = self.pending_user
        self.pending_user = None
        if user:
            self.messages.append({"role": "user", "content": user})
        text = assistant_text.strip()
        if text:
            self.messages.append({"role": "assistant", "content": text})
        self.prune()

    def prune(self) -> None:
        max_messages = max(2, int(self.max_turns) * 2)
        if len(self.messages) > max_messages:
            self.messages = self.messages[-max_messages:]


def get_or_create_session_history(session_id: str, *, max_turns: int = _DEFAULT_MAX_TURNS) -> SessionHistory:
    with _LOCK:
        history = _STORE.get(session_id)
        if history is None:
            history = SessionHistory(session_id=session_id, max_turns=max_turns)
            _STORE[session_id] = history
        return history


def drop_session_history(session_id: str) -> None:
    with _LOCK:
        _STORE.pop(session_id, None)


__all__ = [
    "SessionHistory",
    "drop_session_history",
    "get_or_create_session_history",
]
