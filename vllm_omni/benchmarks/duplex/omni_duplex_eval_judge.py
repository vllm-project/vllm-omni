# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""OpenAI-compatible local judge client."""

from __future__ import annotations

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


class DuplexJudge:
    def __init__(self, base_url: str, model: str, *, api_key: str = "EMPTY", timeout: int = 600) -> None:
        self.base_url, self.model, self.api_key, self.timeout = base_url, model, api_key, timeout

    def chat(self, content: Any, *, system: str | None = None, max_tokens: int = 1200) -> str:
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": content}]
        response = requests.post(
            _build_openai_url(self.base_url, "/chat/completions"),
            json={
                "model": self.model,
                "messages": messages,
                "temperature": 0.1,
                "max_tokens": max_tokens,
            },
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            timeout=self.timeout,
        )
        if not response.ok:
            raise requests.HTTPError(
                f"{response.status_code} response from judge: {response.text[:500]}", response=response
            )
        return str(response.json()["choices"][0]["message"]["content"])

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
