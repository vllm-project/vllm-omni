# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The two-stage BAGEL pipeline must render chat requests like the reference inferencer.

``interleave_inference`` builds the context segment by segment: an image is
``<|vision_start|>`` + ViT tokens + ``<|vision_end|>``, a text is
``<|im_start|>`` + text + ``<|im_end|>``, and text generation starts from a
fresh ``<|im_start|>``. There are no role names and no system prompt, so the
tokenizer's Qwen chat template must not be used for this model.
"""

from __future__ import annotations

import re

import pytest
from jinja2.sandbox import ImmutableSandboxedEnvironment

from vllm_omni.entrypoints.openai.chat_template import load_pipeline_chat_template
from vllm_omni.model_executor.models.bagel.pipeline import (
    BAGEL_CHAT_TEMPLATE,
    BAGEL_PIPELINE,
    BAGEL_SINGLE_STAGE_PIPELINE,
    BAGEL_THINK_PIPELINE,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

BOS = "<|im_start|>"
EOS = "<|im_end|>"
IMAGE = "<|vision_start|><|image_pad|><|vision_end|>"
SPECIAL_TOKENS = (BOS, EOS, "<|vision_start|>", "<|vision_end|>", "<|image_pad|>")
QUESTION = "What is the capital of France?"
DESCRIBE = "Describe this image in detail."


def _render(messages: list[dict], add_generation_prompt: bool = True) -> str:
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    return env.from_string(BAGEL_CHAT_TEMPLATE).render(messages=messages, add_generation_prompt=add_generation_prompt)


def _tokens(text: str) -> list[str]:
    pattern = "(" + "|".join(re.escape(token) for token in SPECIAL_TOKENS) + ")"
    tokens: list[str] = []
    for piece in re.split(pattern, text):
        if piece in SPECIAL_TOKENS:
            tokens.append(piece)
        else:
            tokens.extend(piece.split())
    return tokens


def _user(*parts: dict) -> dict:
    return {"role": "user", "content": list(parts)}


def _text(text: str) -> dict:
    return {"type": "text", "text": text}


def test_text_request_matches_reference_context():
    rendered = _render([_user(_text(QUESTION))])

    assert rendered == f"{BOS}{QUESTION}{EOS}{BOS}"
    assert _tokens(rendered) == [BOS, *QUESTION.split(), EOS, BOS]


def test_string_content_is_framed_the_same_way():
    assert _render([{"role": "user", "content": QUESTION}]) == f"{BOS}{QUESTION}{EOS}{BOS}"


@pytest.mark.parametrize(
    "image_part",
    [{"type": "image"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}],
    ids=["image", "image_url"],
)
def test_image_comes_first_with_vision_markers(image_part: dict):
    rendered = _render([_user(_text(DESCRIBE), image_part)])

    assert rendered == f"{IMAGE}{BOS}{DESCRIBE}{EOS}{BOS}"
    assert _tokens(rendered) == [
        "<|vision_start|>",
        "<|image_pad|>",
        "<|vision_end|>",
        BOS,
        *DESCRIBE.split(),
        EOS,
        BOS,
    ]


def test_every_message_is_a_framed_segment_without_roles():
    rendered = _render(
        [
            {"role": "system", "content": "Think first."},
            _user(_text(QUESTION)),
            {"role": "assistant", "content": "Paris."},
            _user(_text("And of Italy?")),
        ]
    )

    assert rendered == f"{BOS}Think first.{EOS}{BOS}{QUESTION}{EOS}{BOS}Paris.{EOS}{BOS}And of Italy?{EOS}{BOS}"
    assert "system" not in rendered and "user" not in rendered and "assistant" not in rendered


def test_already_framed_text_is_not_framed_twice():
    assert _render([_user(_text(f"{BOS}{QUESTION}{EOS}"))]) == _render([_user(_text(QUESTION))])


def test_empty_text_adds_no_tokens():
    assert _render([_user(_text(""), {"type": "image"})]) == f"{IMAGE}{BOS}"


def test_without_generation_prompt():
    assert _render([_user(_text(QUESTION))], add_generation_prompt=False) == f"{BOS}{QUESTION}{EOS}"


def test_two_stage_pipelines_declare_the_template():
    assert BAGEL_PIPELINE.chat_template is BAGEL_CHAT_TEMPLATE
    assert BAGEL_THINK_PIPELINE.chat_template is BAGEL_CHAT_TEMPLATE
    assert BAGEL_SINGLE_STAGE_PIPELINE.chat_template is None


@pytest.mark.parametrize(
    ("pipeline", "expected"),
    [(BAGEL_PIPELINE, BAGEL_CHAT_TEMPLATE), (BAGEL_SINGLE_STAGE_PIPELINE, None), (None, None)],
    ids=["two-stage", "single-stage", "unknown-model"],
)
def test_load_pipeline_chat_template(monkeypatch: pytest.MonkeyPatch, pipeline, expected):
    seen: dict = {}

    def get_pipeline_config(**kwargs):
        seen.update(kwargs)
        return pipeline

    monkeypatch.setattr(
        "vllm_omni.config.config_factory.StageConfigFactory.get_pipeline_config",
        staticmethod(get_pipeline_config),
    )

    template = load_pipeline_chat_template("bagel", trust_remote_code=True, deploy_config_path="deploy.yaml")

    assert template is expected
    assert seen == {"model": "bagel", "trust_remote_code": True, "deploy_config_path": "deploy.yaml"}


def test_load_pipeline_chat_template_survives_resolution_errors(monkeypatch: pytest.MonkeyPatch):
    def get_pipeline_config(**kwargs):
        raise RuntimeError("no config")

    monkeypatch.setattr(
        "vllm_omni.config.config_factory.StageConfigFactory.get_pipeline_config",
        staticmethod(get_pipeline_config),
    )

    assert load_pipeline_chat_template("bagel") is None
