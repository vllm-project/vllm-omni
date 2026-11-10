# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Contract tests for the AR budget cap and its chat-path reachability."""

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest
from vllm import SamplingParams

import vllm_omni.entrypoints.openai.serving_chat as serving_chat_module
from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat
from vllm_omni.model_extras.registry import get_ar_max_tokens_builder

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _chat_with_builder(monkeypatch, builder):
    chat = object.__new__(OmniOpenAIServingChat)
    monkeypatch.setattr(chat, "_resolve_multistage_model_class_name", lambda engine: "GlmImagePipeline")
    monkeypatch.setattr(serving_chat_module, "get_ar_max_tokens_builder", lambda name: builder)
    return chat


def test_cap_passes_sizeless_request_to_builder(monkeypatch):
    """A size-less request reaches the builder unresolved (None, None).

    Fails on 5784c3f0/19abcf9a, where the cap substituted 1024 itself, and on
    earlier heads, where the cap returned before reaching the builder. The
    builder owns the default resolution (GLM-Image: 1281 for t2i).
    """
    captured: dict[str, Any] = {}

    def builder(height, width, is_i2i=False):
        captured["args"] = (height, width, is_i2i)
        return height + width if height is not None else 1281

    chat = _chat_with_builder(monkeypatch, builder)
    sp = SimpleNamespace(max_tokens=4353)
    chat._cap_ar_stage_max_tokens(None, [sp], 0, None, None)
    assert captured["args"] == (None, None, False)
    # Only checks the write-back: the builder's return value lands on the
    # AR stage budget. The real 1281/1025 values are pinned by the registry
    # test below.
    assert sp.max_tokens == 1281


def test_cap_keeps_configured_ceiling_without_builder(monkeypatch):
    """Models without a builder keep their configured ceiling."""
    chat = _chat_with_builder(monkeypatch, None)
    sp = SimpleNamespace(max_tokens=4353)
    chat._cap_ar_stage_max_tokens(None, [sp], 0, 768, 768)
    assert sp.max_tokens == 4353


def test_registry_builder_resolves_sizeless_to_glm_default():
    """The registered GLM-Image builder resolves an absent size itself."""
    builder = get_ar_max_tokens_builder("GlmImagePipeline")
    assert builder(None, None, is_i2i=False) == 1281
    assert builder(None, None, is_i2i=True) == 1025


# ---------------------------------------------------------------------------
# Chat-path reachability: a size-less /v1/chat/completions request must
# reach the AR budget cap. Fails on 5784c3f0, where comprehension_idx was
# gated on the request size and the cap stopped at ar_idx is None.
# ---------------------------------------------------------------------------


def _stage_configs():
    return [
        SimpleNamespace(stage_id=0, stage_type="llm", is_comprehension=True, model_arch="GlmImagePipeline"),
        SimpleNamespace(stage_id=1, stage_type="diffusion", is_comprehension=False, model_arch="GlmImagePipeline"),
    ]


class _EngineStub:
    def __init__(self, captured: dict[str, Any]):
        self.stage_configs = _stage_configs()
        self.output_modalities = ["image"]
        self.errored = False
        self.default_sampling_params_list = [SamplingParams(max_tokens=4353), SamplingParams()]
        self._captured = captured

    def resolve_entry_stage_id(self, modalities):
        return 0

    def get_stage_renderer(self, stage_id):
        return SimpleNamespace(get_tokenizer=lambda: None)

    async def get_tokenizer(self):
        return None

    def generate(self, **kwargs):
        self._captured["params"] = kwargs["sampling_params_list"]

        async def _fail():
            raise ValueError("sentinel")
            yield  # pragma: no cover

        return _fail()


def _build_chat(monkeypatch, engine):
    chat = object.__new__(OmniOpenAIServingChat)
    chat._diffusion_mode = False
    chat.engine_client = engine
    chat.models = SimpleNamespace(model_name=lambda lora_request: "glm-image")
    chat.renderer = SimpleNamespace(get_tokenizer=lambda: None)
    chat.online_renderer = SimpleNamespace(validate_chat_template=lambda **kwargs: None)
    chat.parser_cls = None
    chat.use_harmony = False
    chat.chat_template = None
    chat.chat_template_content_format = "openai"
    chat.trust_request_chat_template = False
    chat.enable_prompt_tokens_details = False
    chat.request_logger = None
    monkeypatch.setattr(chat, "_input_modalities", lambda messages: ["image"])
    monkeypatch.setattr(chat, "_check_model", _async_none)
    monkeypatch.setattr(chat, "_maybe_get_adapters", lambda request, supports_default_mm_loras: None)
    monkeypatch.setattr(chat, "_effective_chat_template_kwargs", lambda request: {})
    monkeypatch.setattr(chat, "_preprocess_chat", _async_preprocess)
    monkeypatch.setattr(
        chat,
        "_normalize_diffusion_request_args",
        lambda request: ({}, {}),
    )
    monkeypatch.setattr(
        chat,
        "_extract_diffusion_prompt_and_images_from_messages",
        lambda messages: ("a cat", []),
    )
    monkeypatch.setattr(chat, "_resolve_height_width_from_extra_body", lambda extra_body: (None, None))
    monkeypatch.setattr(
        chat, "_build_sampling_params_list_from_request", lambda request: list(engine.default_sampling_params_list)
    )
    monkeypatch.setattr(chat, "_apply_text_chat_ar_task_mode", lambda params, request: None)
    monkeypatch.setattr(chat, "_get_diffusion_extra_body_params", lambda: frozenset())
    monkeypatch.setattr(chat, "create_error_response", lambda e: e)
    return chat


async def _async_none(*args, **kwargs):
    return None


async def _async_preprocess(*args, **kwargs):
    return (None, [])


def test_sizeless_chat_request_caps_ar_stage(monkeypatch):
    """Driving the real chat handler: the AR stage keeps its grid budget.

    The engine stub's generate() captures the sampling params the handler
    builds and then aborts with a sentinel ValueError, which the handler
    converts into an error response. On 5784c3f0 this test fails: the
    handler gated comprehension_idx on the request size, the cap stopped at
    ar_idx is None, and the AR stage kept the 4353 ceiling.
    """
    captured: dict[str, Any] = {}
    engine = _EngineStub(captured)
    chat = _build_chat(monkeypatch, engine)
    request = SimpleNamespace(
        request_id="req-1",
        messages=[{"role": "user", "content": "a cat"}],
        modalities=["image"],
        stream=False,
        tool_choice=None,
        tools=None,
        add_generation_prompt=True,
        continue_final_message=False,
        add_special_tokens=False,
        chat_template=None,
        chat_template_kwargs={},
        watermarking=None,
    )

    result = asyncio.run(chat._create_chat_completion(request, None))

    assert isinstance(result, ValueError)  # the sentinel error response
    ar_params = captured["params"][0]
    assert ar_params.max_tokens == 1281
