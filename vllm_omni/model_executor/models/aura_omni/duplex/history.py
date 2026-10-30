# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Stage1-local SessionHistory for duplex AURA (import from asr2aura / aura2tts)."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock

# Native gateway (AURA_demo-main/gateway/actor.py): keep up to 45 video
# rounds, then strip the video payload from the oldest 30. Text stays.
# Empty user + <|silent|> after the strip is dropped as a pair.
# Total conversation rounds are effectively unbounded (Native default 999).
_MAX_VIDEO_ROUNDS = 45
_VIDEO_ROUNDS_TO_REMOVE = 30
_MAX_CONVERSATION_ROUNDS = 999
_SILENT = "<|silent|>"
_STORE: dict[str, SessionHistory] = {}
_LOCK = Lock()


@dataclass
class SessionHistory:
    """Per-session chat turns held in the Stage1 process."""

    session_id: str
    max_video_rounds: int = _MAX_VIDEO_ROUNDS
    video_rounds_to_remove: int = _VIDEO_ROUNDS_TO_REMOVE
    max_conversation_rounds: int = _MAX_CONVERSATION_ROUNDS
    messages: list[dict[str, object]] = field(default_factory=list)
    pending_user: str | None = None
    pending_video: object | None = None

    def render_prefix(self) -> str:
        rendered: list[str] = []
        for message in self.messages:
            role = message.get("role")
            content = message.get("content")
            if role not in {"user", "assistant"} or not isinstance(content, str):
                continue
            body = content
            if role == "user" and message.get("video") is not None:
                body = "<|vision_start|><|video_pad|><|vision_end|>" + content
            if not body:
                continue
            rendered.append(f"<|im_start|>{role}\n{body}<|im_end|>\n")
        return "".join(rendered)

    def retained_videos(self) -> list[object]:
        """Committed clips that still have pixels, in prompt order."""
        return [
            message["video"]
            for message in self.messages
            if message.get("role") == "user" and message.get("video") is not None
        ]

    def begin_user_turn(self, transcript: str, video: object | None = None) -> None:
        self.pending_user = transcript.strip()
        self.pending_video = video

    def commit_turn(self, assistant_text: str) -> None:
        user = self.pending_user or ""
        video = self.pending_video
        self.pending_user = None
        self.pending_video = None
        text = (assistant_text or "").strip()
        if not user and video is None:
            if not text or text == _SILENT:
                return
            self.messages.append({"role": "assistant", "content": text})
            self.prune()
            return
        user_message: dict[str, object] = {"role": "user", "content": user}
        if video is not None:
            user_message["video"] = video
        self.messages.append(user_message)
        self.messages.append({"role": "assistant", "content": text or _SILENT})
        self.prune()

    def prune(self) -> None:
        video_indices = [
            index
            for index, message in enumerate(self.messages)
            if message.get("role") == "user" and message.get("video") is not None
        ]
        if len(video_indices) > self.max_video_rounds:
            drop: set[int] = set()
            for index in video_indices[: self.video_rounds_to_remove]:
                user_message = self.messages[index]
                user_message["video"] = None
                text = user_message.get("content")
                if isinstance(text, str) and text.strip():
                    continue
                nxt = index + 1
                if nxt >= len(self.messages) or self.messages[nxt].get("role") != "assistant":
                    continue
                assistant = self.messages[nxt].get("content")
                if isinstance(assistant, str) and assistant.strip() in {_SILENT, ""}:
                    drop.add(index)
                    drop.add(nxt)
            if drop:
                self.messages = [message for index, message in enumerate(self.messages) if index not in drop]
        user_count = sum(1 for message in self.messages if message.get("role") == "user")
        overflow = user_count - self.max_conversation_rounds
        if overflow > 0:
            self.messages = self.messages[overflow * 2 :]


def get_or_create_session_history(session_id: str) -> SessionHistory:
    with _LOCK:
        history = _STORE.get(session_id)
        if history is None:
            history = SessionHistory(session_id=session_id)
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
