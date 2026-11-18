# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Stage1-local SessionHistory for duplex AURA (import from asr2aura / aura2tts)."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock

import regex as re

# Native gateway trims after the new user message is in context and before
# inference (AURA_demo-main/gateway/actor.py _trim_video_context). Crossing 45
# video messages drops the oldest 30, so the prompt holds 16, not 46.
# Empty user + <|silent|> after the strip is dropped as a pair.
# Message-count trim runs only in that same step (Native default 999 rounds).
_MAX_VIDEO_ROUNDS = 45
_VIDEO_ROUNDS_TO_REMOVE = 30
_MAX_CONVERSATION_ROUNDS = 999
_SILENT = "<|silent|>"
_RESPOND = "<|respond|>"
# AURA_v2 was trained with ms-swift's template, whose _add_non_thinking_prefix()
# writes this empty think block before EVERY assistant turn. The native gateway
# reproduces it with two_token_support/chat_template_aura.jinja; without it each
# stored reply is 4 tokens short of training and the model repeats itself.
_EMPTY_THINK = "<think>\n\n</think>\n\n"
# Native actor._CLEAN_PATTERN. Stage1 keeps its stop string
# (include_stop_str_in_output), and render_prefix adds the turn's own
# <|im_end|>, so an uncleaned reply would close twice.
_CLEAN_PATTERN = re.compile(
    r"<\|im_start\|>|<\|im_end\|>|<\|endoftext\|>|"
    r"<\|tool_call\|>|<\|tool_result\|>|<\|tool_call_end\|>|"
    r"<\|tool_response\|>|<\|tool_use\|>|<\|tool_result_end\|>|"
    r"<think>|</think>|<\|silent\|>"
)
_STORE: dict[str, SessionHistory] = {}


def _clean_assistant_text(text: str) -> str:
    """Stored assistant text as native keeps it: cleaned, one control prefix, empty when silent."""
    cleaned = _CLEAN_PATTERN.sub("", text or "").strip()
    respond = cleaned.startswith(_RESPOND)
    while cleaned.startswith(_RESPOND):
        cleaned = cleaned[len(_RESPOND) :].lstrip()
    return f"{_RESPOND}{cleaned}" if respond else cleaned


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

    def render_prefix(self, *, think_prefix: bool = False) -> str:
        """ChatML prefix for the next Stage1 turn.

        Same as v1: stored user/assistant text is written back as-is, and a
        user turn that still has pixels gets one video placeholder. v2 spoken
        turns keep their leading ``<|respond|>`` in that stored text.
        ``think_prefix`` (AURA_v2 two-token) puts the empty think block before
        every assistant turn, as the training template and the native
        ``chat_template_aura.jinja`` do. A
        ``tool_response`` entry (``append_tool_response``) renders as the
        template's own tool-result shape: a ``user``-role turn wrapping
        ``<tool_response>``, with no pairing user message of its own.
        Consecutive results share that one user turn, as in the template.
        """
        rendered: list[str] = []
        tool_results: list[str] = []
        for message in self.messages:
            role = message.get("role")
            content = message.get("content")
            if role == "tool_response":
                if isinstance(content, str) and content:
                    tool_results.append(f"\n<tool_response>\n{content}\n</tool_response>")
                continue
            if role not in {"user", "assistant"} or not isinstance(content, str):
                continue
            if tool_results:
                rendered.append("<|im_start|>user" + "".join(tool_results) + "<|im_end|>\n")
                tool_results.clear()
            body = content
            if role == "user" and message.get("video") is not None:
                body = "<|vision_start|><|video_pad|><|vision_end|>" + content
            if think_prefix and role == "assistant" and body and not body.startswith("<think>"):
                body = _EMPTY_THINK + body
            if not body:
                continue
            rendered.append(f"<|im_start|>{role}\n{body}<|im_end|>\n")
        if tool_results:
            rendered.append("<|im_start|>user" + "".join(tool_results) + "<|im_end|>\n")
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
        # The pending clip is the newest video. Trim before render_prefix so
        # this turn's prompt never contains the 46th clip.
        self.prune()

    def append_tool_response(self, output: str) -> None:
        """Splice a tool result in after the assistant turn that called it.

        Not paired through ``pending_user``: the follow-up Stage1 turn that
        reads this back has no new user utterance of its own
        (``asr2aura`` skips ``begin_user_turn`` for that turn).
        """
        self.messages.append({"role": "tool_response", "content": output or ""})

    def commit_turn(self, assistant_text: str) -> None:
        user = self.pending_user or ""
        video = self.pending_video
        self.pending_user = None
        self.pending_video = None
        text = _clean_assistant_text(assistant_text)
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
        """Drop oldest silent videos on overflow, and always cap message pairs.

        Video pixels follow the gateway: once stored clips plus the pending clip
        pass 45, the oldest 30 lose pixels and empty silent pairs are removed.
        The 999-round message cap applies on every prune, including text-only
        sessions that never overflow the video budget.
        """
        video_indices = [
            index
            for index, message in enumerate(self.messages)
            if message.get("role") == "user" and message.get("video") is not None
        ]
        pending_extra = 1 if self.pending_video is not None else 0
        if len(video_indices) + pending_extra > self.max_video_rounds:
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
        # No system row is stored here, so the cap is the conversation pairs
        # themselves. Native keeps ctx[0] because that row is the system prompt.
        max_messages = self.max_conversation_rounds * 2
        remove = (len(self.messages) - max_messages) // 2 * 2
        if remove > 0:
            self.messages = self.messages[remove:]


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
