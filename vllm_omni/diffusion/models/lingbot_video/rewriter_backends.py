# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Adapted from SGLang (https://github.com/sgl-project/sglang/pull/32848).

from __future__ import annotations

import io
import math
from dataclasses import dataclass
from typing import Any

import httpx
import pybase64 as base64
from PIL import Image
from vllm.logger import init_logger

from vllm_omni.diffusion.models.lingbot_video.rewriter import LingBotVideoRewriter

logger = init_logger(__name__)


def _chat_completions_url(base_url: str) -> str:
    url = base_url.rstrip("/")
    if url.endswith("/v1/chat/completions"):
        return url
    if url.endswith("/v1"):
        return f"{url}/chat/completions"
    return f"{url}/v1/chat/completions"


def _data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{encoded}"


class HTTPRewriterBackend:
    """Call OpenAI-compatible endpoints, selecting a model per turn."""

    def __init__(
        self,
        *,
        url: str,
        map_url: str,
        expand_model: str,
        map_model: str,
        timeout: float,
    ):
        self.url = _chat_completions_url(url)
        self.map_url = _chat_completions_url(map_url)
        self.expand_model = expand_model
        self.map_model = map_model
        self.timeout = timeout

    def generate(self, text: str, image: Image.Image | None, use_lora: bool) -> str:
        content: list[dict[str, Any]] = [{"type": "text", "text": text}]
        if image is not None:
            content.insert(0, {"type": "image_url", "image_url": {"url": _data_url(image)}})
        response = httpx.post(
            self.map_url if use_lora else self.url,
            json={
                "model": self.map_model if use_lora else self.expand_model,
                "messages": [{"role": "user", "content": content}],
                "temperature": 0.0,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        try:
            generated = response.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise RuntimeError("LingBot rewriter endpoint returned an invalid chat completion payload.") from exc
        if not isinstance(generated, str):
            raise RuntimeError("LingBot rewriter endpoint returned non-text chat content.")
        return generated.strip()


class TransformersRewriterBackend:
    """Lazily load the base VLM and official PEFT rewriter adapter."""

    def __init__(
        self,
        *,
        model_path: str,
        adapter_path: str,
        device_map: Any,
        max_new_tokens: int,
    ):
        self.model_path = model_path
        self.adapter_path = adapter_path
        self.device_map = device_map
        self.max_new_tokens = max_new_tokens
        self.processor = None
        self.model = None

    def _load(self) -> None:
        try:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:
            raise ImportError(
                "The in-process LingBot rewriter requires `peft` and a Transformers build "
                "with AutoModelForImageTextToText support."
            ) from exc

        logger.info(
            "Loading LingBot prompt rewriter model %s with adapter %s",
            self.model_path,
            self.adapter_path,
        )
        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        base = AutoModelForImageTextToText.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map=self.device_map,
            trust_remote_code=True,
        )
        self.model = PeftModel.from_pretrained(base, self.adapter_path).eval()

    def generate(self, text: str, image: Image.Image | None, use_lora: bool) -> str:
        import contextlib

        import torch

        if self.model is None:
            self._load()
        content = [{"type": "image", "image": image}] if image is not None else []
        content.append({"type": "text", "text": text})
        chat = self.processor.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.processor(
            text=[chat],
            images=[image] if image is not None else None,
            return_tensors="pt",
        ).to(self.model.device)
        adapter = contextlib.nullcontext() if use_lora else self.model.disable_adapter()
        with torch.no_grad(), adapter:
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        generated = out[:, inputs["input_ids"].shape[1] :]
        return self.processor.batch_decode(generated, skip_special_tokens=True)[0].strip()


@dataclass(frozen=True)
class LingBotRewriterSettings:
    url: str | None
    map_url: str | None
    expand_model: str
    map_model: str
    timeout: float
    model_path: str | None
    adapter_path: str | None
    device_map: Any
    max_new_tokens: int
    auto_negative: bool

    @classmethod
    def from_model_config(cls, model_config: dict[str, Any]) -> LingBotRewriterSettings:
        def optional_string(name: str) -> str | None:
            value = model_config.get(name)
            if value is None:
                return None
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"LingBot `{name}` must be a non-empty string when set.")
            return value

        url = optional_string("rewriter_url")
        map_url = optional_string("rewriter_map_url")
        model_path = optional_string("rewriter_model_path")
        adapter_path = optional_string("rewriter_adapter_path")
        expand_model = optional_string("rewriter_expand_model") or "lingbot-rewriter-base"
        map_model = optional_string("rewriter_map_model") or "lingbot-rewriter-lora"

        timeout_value = model_config.get("rewriter_timeout", 300.0)
        try:
            timeout = float(timeout_value)
        except (TypeError, ValueError) as exc:
            raise ValueError("LingBot `rewriter_timeout` must be a positive number.") from exc
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("LingBot `rewriter_timeout` must be positive.")
        max_new_tokens_value = model_config.get("rewriter_max_new_tokens", 6144)
        try:
            max_new_tokens = int(max_new_tokens_value)
            numeric_max_new_tokens = float(max_new_tokens_value)
        except (TypeError, ValueError) as exc:
            raise ValueError("LingBot `rewriter_max_new_tokens` must be a positive integer.") from exc
        if (
            isinstance(max_new_tokens_value, bool)
            or not math.isfinite(numeric_max_new_tokens)
            or numeric_max_new_tokens != max_new_tokens
            or max_new_tokens <= 0
        ):
            raise ValueError("LingBot `rewriter_max_new_tokens` must be a positive integer.")
        auto_negative = model_config.get("rewriter_auto_negative", False)
        if not isinstance(auto_negative, bool):
            raise ValueError("LingBot `rewriter_auto_negative` must be a boolean.")

        if url is not None and model_path is not None:
            raise ValueError("LingBot `rewriter_url` and `rewriter_model_path` are mutually exclusive.")
        if map_url is not None and url is None:
            raise ValueError("LingBot `rewriter_map_url` requires `rewriter_url`.")
        if adapter_path is not None and model_path is None:
            raise ValueError("LingBot `rewriter_adapter_path` requires `rewriter_model_path`.")
        if model_path is not None and adapter_path is None:
            raise ValueError(
                "LingBot `rewriter_model_path` requires `rewriter_adapter_path`; "
                "the mapping turn must enable the official adapter."
            )
        if auto_negative and url is None and model_path is None:
            raise ValueError("LingBot `rewriter_auto_negative` requires a configured rewriter backend.")

        return cls(
            url=url,
            map_url=map_url,
            expand_model=expand_model,
            map_model=map_model,
            timeout=timeout,
            model_path=model_path,
            adapter_path=adapter_path,
            device_map=model_config.get("rewriter_device_map", "auto"),
            max_new_tokens=max_new_tokens,
            auto_negative=auto_negative,
        )


def build_lingbot_rewriter(model_config: dict[str, Any]) -> LingBotVideoRewriter | None:
    settings = LingBotRewriterSettings.from_model_config(model_config)
    if settings.url is not None:
        backend = HTTPRewriterBackend(
            url=settings.url,
            map_url=settings.map_url or settings.url,
            expand_model=settings.expand_model,
            map_model=settings.map_model,
            timeout=settings.timeout,
        )
    elif settings.model_path is not None:
        assert settings.adapter_path is not None
        backend = TransformersRewriterBackend(
            model_path=settings.model_path,
            adapter_path=settings.adapter_path,
            device_map=settings.device_map,
            max_new_tokens=settings.max_new_tokens,
        )
    else:
        return None
    return LingBotVideoRewriter(backend, auto_negative=settings.auto_negative)
