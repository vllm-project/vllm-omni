# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""OpenAI-compatible local judge client."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pybase64 as base64
import requests


def _build_openai_url(base_url: str, api_path: str) -> str:
    base = base_url.rstrip("/")
    normalized_path = api_path if api_path.startswith("/") else f"/{api_path}"
    if base.endswith(normalized_path):
        return base
    if base.endswith("/v1"):
        return f"{base}{normalized_path}"
    return f"{base}/v1{normalized_path}"


def _choice_modality(choice: dict[str, Any]) -> str:
    """Read the modality tag from a choice; ``""`` when the server omits it."""
    modality = choice.get("modality")
    if modality is None:
        message = choice.get("message")
        if isinstance(message, dict):
            modality = message.get("modality")
    return str(modality) if modality else ""


def _message_text(choice: dict[str, Any]) -> str:
    """Extract the plain text of ``choice.message.content``.

    String content is returned unchanged; list content (multimodal parts) is
    flattened by concatenating the ``text`` field of each text part.
    """
    message = choice.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("type", "text") == "text"
        )
    return "" if content is None else str(content)


def select_judge_text(payload: dict[str, Any]) -> str:
    """Pick the judge answer from an OpenAI-compatible response.

    Three-level strategy (design §5.6a) so a Qwen3-Omni multi-modality judge
    that also emits audio choices still yields the text answer:

    1. the first choice explicitly tagged ``modality == "text"`` with text;
    2. otherwise the first choice whose message text is non-empty;
    3. otherwise ``choices[0]`` (preserves the single-choice behaviour).
    """
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("judge response contains no choices")
    for choice in choices:
        if isinstance(choice, dict) and _choice_modality(choice) == "text" and _message_text(choice).strip():
            return _message_text(choice)
    for choice in choices:
        if isinstance(choice, dict) and _message_text(choice).strip():
            return _message_text(choice)
    first = choices[0]
    return _message_text(first) if isinstance(first, dict) else str(first)


class DuplexJudge:
    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: str = "EMPTY",
        timeout: int = 600,
        modalities: Sequence[str] | None = None,
    ) -> None:
        self.base_url, self.model, self.api_key, self.timeout = base_url, model, api_key, timeout
        # ``None`` keeps the request body byte-identical to the pre-Qwen3-Omni
        # client; only an explicit list asks the server for those modalities.
        self.modalities = tuple(modalities) if modalities else None

    def chat(self, content: Any, *, system: str | None = None, max_tokens: int = 1200) -> str:
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": content}]
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": max_tokens,
        }
        if self.modalities:
            body["modalities"] = list(self.modalities)
        response = requests.post(
            _build_openai_url(self.base_url, "/chat/completions"),
            json=body,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            timeout=self.timeout,
        )
        if not response.ok:
            raise requests.HTTPError(
                f"{response.status_code} response from judge: {response.text[:500]}", response=response
            )
        return select_judge_text(response.json())

    def temporal(self, prompt: str, frames: list[bytes]) -> str:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        content.extend(
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(frame).decode()}}
            for frame in frames
        )
        return self.chat(content, system="You are a careful evaluator for real-time multimodal systems.")

    def content(
        self,
        prompt: str,
        video: str | Path | None = None,
        frames: list[bytes] | None = None,
        *,
        mode: str = "video_url",
    ) -> str:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        if mode == "video_url" and video is not None:
            content.append({"type": "video_url", "video_url": {"url": "file://" + str(Path(video).resolve())}})
        else:
            content.extend(
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(frame).decode()},
                }
                for frame in (frames or [])
            )
        return self.chat(content)
