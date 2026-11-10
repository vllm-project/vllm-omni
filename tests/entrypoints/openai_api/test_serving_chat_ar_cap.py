# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest

import vllm_omni.entrypoints.openai.serving_chat as serving_chat_module
from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat
from vllm_omni.model_executor.models.glm_image.prompt import DEFAULT_TARGET_SIZE

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _chat_with_builder(monkeypatch, builder):
    chat = object.__new__(OmniOpenAIServingChat)
    monkeypatch.setattr(chat, "_resolve_multistage_model_class_name", lambda engine: "GlmImagePipeline")
    monkeypatch.setattr(serving_chat_module, "get_ar_max_tokens_builder", lambda name: builder)
    return chat


def test_cap_resolves_sizeless_request_to_processor_default(monkeypatch):
    """A size-less request is capped at the processor's default grid budget.

    Fails before the size-less cap default existed: the cap returned early
    and left the deploy-YAML ceiling (4353) untouched.
    """
    captured = {}

    def builder(height, width, is_i2i=False):
        captured["args"] = (height, width, is_i2i)
        return height + width

    chat = _chat_with_builder(monkeypatch, builder)
    sp = SimpleNamespace(max_tokens=4353)
    chat._cap_ar_stage_max_tokens(None, [sp], 0, None, None)
    assert captured["args"] == (DEFAULT_TARGET_SIZE, DEFAULT_TARGET_SIZE, False)
    assert sp.max_tokens == DEFAULT_TARGET_SIZE * 2


def test_cap_keeps_configured_ceiling_without_builder(monkeypatch):
    """Models without a builder keep their configured ceiling."""
    chat = _chat_with_builder(monkeypatch, None)
    sp = SimpleNamespace(max_tokens=4353)
    chat._cap_ar_stage_max_tokens(None, [sp], 0, 768, 768)
    assert sp.max_tokens == 4353


def test_cap_uses_explicit_size_when_given(monkeypatch):
    """An explicit target size still reaches the builder unchanged."""

    def builder(height, width, is_i2i=False):
        return height * 10 + width + is_i2i

    chat = _chat_with_builder(monkeypatch, builder)
    sp = SimpleNamespace(max_tokens=4353)
    chat._cap_ar_stage_max_tokens(None, [sp], 0, 512, 768, is_i2i=True)
    assert sp.max_tokens == 512 * 10 + 768 + 1
