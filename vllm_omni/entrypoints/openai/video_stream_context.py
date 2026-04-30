# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Small helpers shared by streaming video handlers."""

from __future__ import annotations

from typing import Any


def text_only_message(message: dict[str, Any]) -> dict[str, Any]:
    """Return a history message with multimodal content stripped out."""
    role = message.get("role", "user")
    content = message.get("content", "")

    if isinstance(content, str):
        return {"role": role, "content": content}

    if not isinstance(content, list):
        return {"role": role, "content": ""}

    text_parts: list[str] = []
    for part in content:
        if isinstance(part, dict) and part.get("type") == "text":
            text = part.get("text")
            if isinstance(text, str):
                text_parts.append(text)

    return {"role": role, "content": "".join(text_parts)}
