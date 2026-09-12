# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Adapted from SGLang (https://github.com/sgl-project/sglang/pull/32848).

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Protocol, TypeVar

import regex as re
import torch.distributed as dist
from PIL import Image

from vllm_omni.diffusion.models.lingbot_video.request_utils import LingBotGenerationMode
from vllm_omni.diffusion.models.lingbot_video.rewriter_prompts import (
    IMAGE_STEP1_EXPAND,
    IMAGE_STEP2_MAP,
    VIDEO_DURATION_EN,
    VIDEO_DURATION_ZH,
    VIDEO_STEP1_EXPAND,
    VIDEO_STEP2_MAP,
)

_CJK = re.compile(r"[　-〿㐀-䶿一-鿿＀-￯]")
_FENCED_JSON = re.compile(r"```(?:json)?\s*(\{.*\})\s*```", re.DOTALL)
_T = TypeVar("_T")


class RewriterBackend(Protocol):
    def generate(self, text: str, image: Image.Image | None, use_lora: bool) -> str: ...


def build_expand_prompt(mode: str, prompt: str, duration: int) -> str:
    if mode == LingBotGenerationMode.T2I.value:
        return f"{IMAGE_STEP1_EXPAND}\n\nUser image prompt:\n{prompt}"
    template = VIDEO_DURATION_ZH if _CJK.search(prompt) else VIDEO_DURATION_EN
    duration_line = template.format(duration=duration)
    return f"{VIDEO_STEP1_EXPAND}\n\n{prompt}\n\n{duration_line}"


def build_map_prompt(mode: str, detailed: str, duration: int) -> str:
    if mode == LingBotGenerationMode.T2I.value:
        return f"{IMAGE_STEP2_MAP}\n\nDETAILED CAPTION:\n{detailed}"
    return (
        f"{VIDEO_STEP2_MAP}\n\nVideo Duration: {duration} seconds\n\n"
        f"DETAILED CAPTION:\n{detailed}\n\nOutput the JSON now."
    )


def parse_caption(raw: str) -> dict | None:
    """Parse a mapped caption while tolerating code fences and stray prose."""

    text = (raw or "").strip()
    fenced = _FENCED_JSON.search(text)
    if fenced:
        text = fenced.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        caption = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return caption if isinstance(caption, dict) else None


def needs_rewrite(prompt: str) -> bool:
    """Structured captions are already in distribution and pass through."""

    return not prompt.lstrip().startswith("{")


def _run_on_primary_rank(callback: Callable[[], _T]) -> _T:
    """Run an external rewriter once and broadcast its result to all ranks."""

    if not dist.is_available() or not dist.is_initialized():
        return callback()

    payload: tuple[bool, object] | None = None
    if dist.get_rank() == 0:
        try:
            payload = (True, callback())
        except Exception as exc:  # noqa: BLE001 - synchronize the failure to peers
            payload = (False, (type(exc).__name__, str(exc)))
    objects = [payload]
    dist.broadcast_object_list(objects, src=0)
    success, result = objects[0]
    if not success:
        error_type, message = result
        raise RuntimeError(f"LingBot prompt rewriter failed on rank 0 ({error_type}): {message}")
    return result  # type: ignore[return-value]


class LingBotVideoRewriter:
    """Turn free text into the structured caption expected by LingBot-Video.

    The first turn expands the request with the base VLM. The second turn maps
    that expansion to JSON with the official rewriter adapter enabled.
    """

    def __init__(self, backend: RewriterBackend, *, auto_negative: bool = False):
        self.backend = backend
        self.auto_negative = auto_negative

    def rewrite_request(
        self,
        *,
        prompt: str,
        negative_prompt: str,
        mode: LingBotGenerationMode,
        num_frames: int,
        fps: int,
        input_image: Image.Image | None,
    ) -> tuple[str, str]:
        if not needs_rewrite(prompt) and not self.auto_negative:
            return prompt, negative_prompt

        mode_value = mode.value
        image = input_image if mode is LingBotGenerationMode.TI2V else None
        duration = max(1, round(int(num_frames) / max(int(fps), 1)))

        def rewrite() -> tuple[str, str]:
            rewritten = self._rewrite_prompt(prompt, mode_value, duration, image)
            rewritten_negative = negative_prompt
            if self.auto_negative:
                from vllm_omni.diffusion.models.lingbot_video.auto_negative import (
                    customize_negative_prompt,
                )

                rewritten_negative = customize_negative_prompt(
                    backend=self.backend,
                    caption=rewritten,
                    mode=mode_value,
                    negative_prompt=negative_prompt,
                    image=image,
                )
            return rewritten, rewritten_negative

        return _run_on_primary_rank(rewrite)

    def _rewrite_prompt(
        self,
        prompt: str,
        mode: str,
        duration: int,
        image: Image.Image | None,
    ) -> str:
        if not needs_rewrite(prompt):
            return prompt
        detailed = self.backend.generate(
            build_expand_prompt(mode, prompt, duration),
            image,
            use_lora=False,
        )
        raw = self.backend.generate(
            build_map_prompt(mode, detailed, duration),
            image,
            use_lora=True,
        )
        caption = parse_caption(raw)
        if caption is None:
            raise RuntimeError(
                "Prompt rewriting produced no parseable structured caption. "
                "Pass a structured JSON caption directly, or check the rewriter backend."
            )
        return json.dumps(caption, ensure_ascii=False, separators=(",", ":"))
