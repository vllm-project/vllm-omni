# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Protocol

import httpx
from openai import AsyncOpenAI

from vllm_omni.experimental.fullduplex.joyvl.decision.prompts import DELEGATION_SOLVER_PROMPT


def _cancel_task(tasks: dict[str, asyncio.Task], task_id: str) -> None:
    task = tasks.pop(task_id, None)
    if task is not None and not task.done():
        task.cancel()


async def _aclose_tasks(tasks: dict[str, asyncio.Task]) -> None:
    pending = list(tasks.values())
    for task in pending:
        if not task.done():
            task.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)
    tasks.clear()


@dataclass
class DelegationResult:
    task_id: str
    status: str
    digest: str = ""
    media: str = ""

    @property
    def is_ready(self) -> bool:
        return self.status == "ready"


class DelegationBridge(Protocol):
    async def submit(self, question: str, note: str, frames: list[tuple[str, str]]) -> str: ...

    async def poll(self, task_id: str) -> DelegationResult: ...

    def cancel(self, task_id: str) -> None:
        """Drop a task and cancel its background work if still running.

        Called when a session is reset/evicted so finished-but-unpolled tasks do not
        accumulate on a long-lived shared bridge.
        """
        ...

    async def aclose(self) -> None:
        """Cancel+await all pending tasks and close any held client (server shutdown)."""
        ...


class StubDelegationBridge:
    def __init__(self, ready_after_ticks: int = 2) -> None:
        self._ready_after = max(1, ready_after_ticks)
        self._tasks: dict[str, dict[str, Any]] = {}
        self._counter = 0

    async def submit(self, question: str, note: str, frames: list[tuple[str, str]]) -> str:
        self._counter += 1
        task_id = f"deleg-{self._counter}"
        self._tasks[task_id] = {"question": question, "polls": 0}
        return task_id

    async def poll(self, task_id: str) -> DelegationResult:
        task = self._tasks.get(task_id)
        if task is None:
            return DelegationResult(task_id, "error", "unknown task")
        task["polls"] += 1
        if task["polls"] < self._ready_after:
            return DelegationResult(task_id, "pending")
        question = task["question"]
        return DelegationResult(
            task_id,
            "ready",
            digest=f"(background result for: {question}) — stub digest; wire a real brain here.",
        )

    def cancel(self, task_id: str) -> None:
        self._tasks.pop(task_id, None)

    async def aclose(self) -> None:
        self._tasks.clear()


class OpenAIDelegationBridge:
    """Hand a delegated question to a stronger background model over an OpenAI-compatible API.

    Non-blocking by design: ``submit`` launches the request as a background task and returns
    immediately; ``poll`` only checks whether that task has finished, so the per-tick interaction
    loop never stalls waiting on the slower background brain.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "EMPTY",
        *,
        max_frames: int = 8,
        max_tokens: int = 512,
        temperature: float = 0.2,
        timeout: float = 120.0,
        system_prompt: str = DELEGATION_SOLVER_PROMPT,
    ) -> None:
        self.model = model
        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self._max_frames = max_frames
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._system_prompt = system_prompt
        self._tasks: dict[str, asyncio.Task] = {}
        self._counter = 0

    async def submit(self, question: str, note: str, frames: list[tuple[str, str]]) -> str:
        self._counter += 1
        task_id = f"deleg-{self._counter}"
        self._tasks[task_id] = asyncio.create_task(self._solve(question, note, frames))
        return task_id

    async def poll(self, task_id: str) -> DelegationResult:
        task = self._tasks.get(task_id)
        if task is None:
            return DelegationResult(task_id, "error", "unknown task")
        if not task.done():
            return DelegationResult(task_id, "pending")
        self._tasks.pop(task_id, None)
        try:
            digest = task.result()
        except Exception as exc:  # surface any background failure as an error result, never crash the loop
            return DelegationResult(task_id, "error", str(exc))
        return DelegationResult(task_id, "ready", digest=digest)

    async def _solve(self, question: str, note: str, frames: list[tuple[str, str]]) -> str:
        content: list[dict[str, Any]] = []
        for time_range, data_url in frames[-self._max_frames :]:
            content.append({"type": "text", "text": f"<{time_range}>"})
            content.append({"type": "image_url", "image_url": {"url": data_url}})
        content.append({"type": "text", "text": question or note or "Answer the user's question."})
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": content},
        ]
        return await self._complete(messages)

    async def _complete(self, messages: list[dict[str, Any]]) -> str:
        response = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
        )
        if not response.choices:
            return ""
        return (response.choices[0].message.content or "").strip()

    def cancel(self, task_id: str) -> None:
        _cancel_task(self._tasks, task_id)

    async def aclose(self) -> None:
        await _aclose_tasks(self._tasks)
        await self._client.close()


class ImageGenDelegationBridge:
    """Delegate an image-generation request to a text-to-image model over the
    OpenAI-compatible ``/v1/images/generations`` endpoint (e.g. vLLM-Omni Qwen-Image).

    Non-blocking: ``submit`` launches generation as a background task and returns at once;
    ``poll`` only checks completion. The generated image is returned as a ``data:`` URL in
    ``DelegationResult.media`` (surfaced to the client), while ``digest`` stays a short text
    placeholder so the base64 image never enters the model's context.
    """

    def __init__(
        self,
        base_url: str,
        *,
        size: str = "1024x1024",
        timeout: float = 300.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._size = size
        self._timeout = timeout
        self._tasks: dict[str, asyncio.Task] = {}
        self._counter = 0

    async def submit(self, question: str, note: str, frames: list[tuple[str, str]]) -> str:
        self._counter += 1
        task_id = f"deleg-{self._counter}"
        self._tasks[task_id] = asyncio.create_task(self._generate(question or note))
        return task_id

    async def poll(self, task_id: str) -> DelegationResult:
        task = self._tasks.get(task_id)
        if task is None:
            return DelegationResult(task_id, "error", "unknown task")
        if not task.done():
            return DelegationResult(task_id, "pending")
        self._tasks.pop(task_id, None)
        try:
            prompt, data_url = task.result()
        except Exception as exc:  # surface any background failure as an error result, never crash the loop
            return DelegationResult(task_id, "error", str(exc))
        return DelegationResult(task_id, "ready", digest=f"(generated an image for: {prompt})", media=data_url)

    async def _generate(self, prompt: str) -> tuple[str, str]:
        data = await self._call_images({"prompt": prompt, "size": self._size})
        items = data.get("data") or []
        b64 = items[0].get("b64_json") if items and isinstance(items[0], dict) else None
        if not b64:
            raise RuntimeError("image endpoint returned no image data")
        return prompt, f"data:image/png;base64,{b64}"

    async def _call_images(self, payload: dict[str, Any]) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(f"{self._base_url}/images/generations", json=payload)
            resp.raise_for_status()
            return resp.json()

    def cancel(self, task_id: str) -> None:
        _cancel_task(self._tasks, task_id)

    async def aclose(self) -> None:
        await _aclose_tasks(self._tasks)


def _extract_image_url(data: dict[str, Any]) -> str | None:
    choices = data.get("choices") or []
    if not choices:
        return None
    content = (choices[0].get("message") or {}).get("content")
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("image_url"):
                return (part["image_url"] or {}).get("url")
    if isinstance(content, str) and content.startswith("data:image/"):
        return content
    return None


class ImageEditDelegationBridge:
    """Delegate "restyle / cartoonify what you currently see" to an image-editing model
    (e.g. vLLM-Omni Qwen-Image-Edit) over the OpenAI-compatible ``/v1/chat/completions``
    endpoint, which takes an input image + instruction and returns the edited image.

    Conditions on the latest frame passed to ``submit`` — so it preserves the real
    composition instead of re-imagining it. Non-blocking submit/poll; the edited image
    rides in ``DelegationResult.media`` while ``digest`` stays a short text placeholder.
    """

    _DEFAULT_INSTRUCTION = (
        "Convert this image into a clean cartoon / anime illustration style. "
        "Preserve the composition, subjects, poses, and overall layout."
    )

    def __init__(self, base_url: str, model: str, *, timeout: float = 300.0, instruction: str | None = None) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout
        self._instruction = instruction or self._DEFAULT_INSTRUCTION
        self._tasks: dict[str, asyncio.Task] = {}
        self._counter = 0

    async def submit(self, question: str, note: str, frames: list[tuple[str, str]]) -> str:
        self._counter += 1
        task_id = f"deleg-{self._counter}"
        frame = frames[-1][1] if frames else None
        self._tasks[task_id] = asyncio.create_task(self._edit(question or note, frame))
        return task_id

    async def poll(self, task_id: str) -> DelegationResult:
        task = self._tasks.get(task_id)
        if task is None:
            return DelegationResult(task_id, "error", "unknown task")
        if not task.done():
            return DelegationResult(task_id, "pending")
        self._tasks.pop(task_id, None)
        try:
            label, data_url = task.result()
        except Exception as exc:  # surface any background failure as an error result, never crash the loop
            return DelegationResult(task_id, "error", str(exc))
        return DelegationResult(task_id, "ready", digest=f"(restyled the current view: {label})", media=data_url)

    async def _edit(self, request_text: str, frame: str | None) -> tuple[str, str]:
        if not frame:
            raise RuntimeError("no frame available to restyle")
        instruction = request_text or self._instruction
        data = await self._call_chat(frame, instruction)
        url = _extract_image_url(data)
        if not url:
            raise RuntimeError("image-edit endpoint returned no image")
        return (request_text or "cartoon"), url

    async def _call_chat(self, frame: str, instruction: str) -> dict[str, Any]:
        body = {
            "model": self._model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": frame}},
                        {"type": "text", "text": instruction},
                    ],
                }
            ],
        }
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(f"{self._base_url}/chat/completions", json=body)
            resp.raise_for_status()
            return resp.json()

    def cancel(self, task_id: str) -> None:
        _cancel_task(self._tasks, task_id)

    async def aclose(self) -> None:
        await _aclose_tasks(self._tasks)


_EDIT_STYLE_KW = ("卡通", "漫画", "动漫", "cartoon", "anime", "风格", "restyle", "变成", "画成")
_EDIT_SCENE_KW = ("看到", "画面", "当前", "现在", "this", "what you see", "场景", "镜头")
_IMAGE_KW = ("画", "生成", "draw", "generate", "picture", "image", "图", "render")


class RoutingDelegationBridge:
    """Dispatch each delegation to the right backend by inspecting the request text.

    - "restyle / cartoonify what you currently see" -> image-edit (conditions on the frame)
    - "draw / generate an image of X"               -> text-to-image
    - everything else (hard questions)              -> the chat / VL background brain

    Wraps sub-bridges that each implement the DelegationBridge protocol; any may be None.
    Routing is heuristic and demo-grade, not a trained classifier.
    """

    def __init__(self, *, chat: Any = None, image: Any = None, edit: Any = None) -> None:
        self._chat = chat
        self._image = image
        self._edit = edit
        self._route: dict[str, tuple[Any, str]] = {}
        self._counter = 0

    def _pick(self, question: str, note: str) -> Any:
        text = f"{question} {note}".lower()
        if self._edit and any(k in text for k in _EDIT_STYLE_KW) and any(k in text for k in _EDIT_SCENE_KW):
            return self._edit
        if self._image and any(k in text for k in _IMAGE_KW):
            return self._image
        return self._chat or self._image or self._edit

    async def submit(self, question: str, note: str, frames: list[tuple[str, str]]) -> str:
        bridge = self._pick(question, note)
        if bridge is None:
            raise RuntimeError("no delegation backend configured")
        self._counter += 1
        key = f"route-{self._counter}"
        self._route[key] = (bridge, await bridge.submit(question, note, frames))
        return key

    async def poll(self, task_id: str) -> DelegationResult:
        entry = self._route.get(task_id)
        if entry is None:
            return DelegationResult(task_id, "error", "unknown task")
        bridge, inner_id = entry
        res = await bridge.poll(inner_id)
        if res.status in ("ready", "error"):
            self._route.pop(task_id, None)
        return DelegationResult(task_id, res.status, res.digest, res.media)

    def cancel(self, task_id: str) -> None:
        entry = self._route.pop(task_id, None)
        if entry is not None:
            bridge, inner_id = entry
            bridge.cancel(inner_id)

    async def aclose(self) -> None:
        self._route.clear()
        for bridge in (self._chat, self._image, self._edit):
            if bridge is not None:
                await bridge.aclose()
