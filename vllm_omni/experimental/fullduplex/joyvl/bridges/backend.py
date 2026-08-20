# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from typing import Any, Protocol

from openai import AsyncOpenAI


class ModelBackend(Protocol):
    async def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        extra_body: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any] | None]: ...

    async def aclose(self) -> None: ...


class OpenAIBackend:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "EMPTY",
        timeout: float = 300.0,
    ) -> None:
        self.model = model
        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=timeout)

    async def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        extra_body: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any] | None]:
        response = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            extra_body=extra_body or {},
        )
        if not response.choices:
            return "", None
        text = response.choices[0].message.content or ""
        usage = response.usage.model_dump() if getattr(response, "usage", None) else None
        return text, usage

    async def aclose(self) -> None:
        await self._client.close()
