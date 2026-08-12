# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import builtins

import pytest

from vllm_omni.model_executor.models.indextts2 import text_processing_v2_5
from vllm_omni.model_executor.models.indextts2.text_processing_v2_5 import (
    apply_pronunciation_annotations,
    clean_indextts25_text,
    prepare_indextts25_text,
)
from vllm_omni.model_executor.models.indextts2.tokenizer_v2_5 import (
    INDEXTTS25_SPECIAL_TOKENS,
    INDEXTTS25_VOCAB_SIZE,
    LANGUAGE_DICT,
    lang_to_token,
    normalize_language_code,
    resolve_indextts25_tokenizer_file,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_special_token_layout_matches_official_checkpoint_vocab():
    assert len(INDEXTTS25_SPECIAL_TOKENS) == 1673
    assert len(set(INDEXTTS25_SPECIAL_TOKENS)) == 1673
    assert 58836 + len(INDEXTTS25_SPECIAL_TOKENS) == INDEXTTS25_VOCAB_SIZE
    assert INDEXTTS25_VOCAB_SIZE == 60509


def test_pronunciation_annotations_select_english_and_chinese_markers():
    assert apply_pronunciation_annotations("<going|G OW1 . IH0 NG>") == (
        "<|SPECIAL_TOKEN_1|>G OW1 . IH0 NG<|SPECIAL_TOKEN_1|>"
    )
    assert apply_pronunciation_annotations("<行|xing2>") == ("<|SPECIAL_TOKEN_2|>XING2<|SPECIAL_TOKEN_2|>")


def test_prepare_text_applies_language_prefix_case_and_annotation(monkeypatch):
    captured = {}

    def fake_encode(text, *, model_dir, tokenizer_file):
        captured["text"] = text
        captured["model_dir"] = model_dir
        captured["tokenizer_file"] = tokenizer_file
        return [7, 8, 9]

    monkeypatch.setattr(text_processing_v2_5, "encode_indextts25_text", fake_encode)

    token_ids, lang_id = prepare_indextts25_text(
        "HELLO <行|xing2>",
        lang="zh",
        model_dir="/model",
        text_normalization=False,
    )

    assert token_ids == [7, 8, 9]
    assert lang_id == LANGUAGE_DICT["zh"]
    assert captured == {
        "text": "<|zh|> hello <|SPECIAL_TOKEN_2|>XING2<|SPECIAL_TOKEN_2|>",
        "model_dir": "/model",
        "tokenizer_file": "multilingual_zh_ja_yue_char_del.tiktoken",
    }


def test_mandarin_alias_is_an_intentional_vllm_omni_convenience():
    assert normalize_language_code("Mandarin") == "zh"
    assert lang_to_token("mandarin") == LANGUAGE_DICT["zh"]


def test_invalid_language_does_not_fall_back_to_common():
    with pytest.raises(ValueError, match="Unsupported IndexTTS 2.5 language"):
        lang_to_token("xx-invalid")


def test_zhen_uses_mixed_normalization():
    calls = []

    def fake_mixed_normalizer(text):
        calls.append(text)
        return "MIXED"

    original = text_processing_v2_5._normalize_zh_or_en
    text_processing_v2_5._normalize_zh_or_en = fake_mixed_normalizer
    try:
        result = text_processing_v2_5._normalize_with_official_backend(
            "中文 and English",
            "zhen",
        )
    finally:
        text_processing_v2_5._normalize_zh_or_en = original

    assert calls == ["中文 and English"]
    assert result == "MIXED"


def test_spanish_normalization_does_not_hide_broken_dependencies(monkeypatch):
    original_import = builtins.__import__

    def controlled_import(name, *args, **kwargs):
        if name == "nemo_text_processing.text_normalization.normalize":
            raise RuntimeError("broken NeMo grammar")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", controlled_import)

    with pytest.raises(RuntimeError, match="broken NeMo grammar"):
        text_processing_v2_5._normalize_with_official_backend("hola", "es")


def test_zhen_keeps_literal_prefix_and_uses_common_embedding(monkeypatch):
    captured = {}

    def fake_encode(text, *, model_dir, tokenizer_file):
        captured["text"] = text
        return [7, 8, 9]

    monkeypatch.setattr(text_processing_v2_5, "encode_indextts25_text", fake_encode)

    token_ids, lang_id = prepare_indextts25_text(
        "HELLO 中文",
        lang="zhen",
        model_dir="/model",
        text_normalization=False,
    )

    assert token_ids == [7, 8, 9]
    assert captured["text"] == "<|zhen|> hello 中文"
    assert lang_id == LANGUAGE_DICT["common"]
    assert "<|zhen|>" not in INDEXTTS25_SPECIAL_TOKENS
    assert "<|common|>" not in INDEXTTS25_SPECIAL_TOKENS


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("，，，", "…"),
        ("测试，，，结束", "测试…结束"),
    ],
)
def test_repeated_full_width_commas_use_longest_first_cleanup(text, expected):
    assert clean_indextts25_text(text) == expected


def test_default_text_normalizer_is_shared_by_prompt_and_talker(monkeypatch):
    calls = []

    def fake_normalizer(text, lang):
        calls.append((text, lang))
        return "twenty five percent"

    monkeypatch.setattr(
        text_processing_v2_5,
        "_normalize_with_official_backend",
        fake_normalizer,
    )

    result = text_processing_v2_5.normalize_indextts25_text(
        "25%",
        lang="en",
        text_normalization=True,
    )

    assert calls == [("25%", "en")]
    assert result == "twenty five percent"


def test_mixed_text_uses_official_content_routing_and_protection(monkeypatch):
    calls = []

    class FakeNormalizer:
        def __init__(self, lang):
            self.lang = lang

        def normalize(self, text):
            calls.append((self.lang, text))
            return text.replace("<H>", " <H> ").replace("5", "five")

    monkeypatch.setattr(
        text_processing_v2_5,
        "_load_wetext_normalizer",
        lambda lang: FakeNormalizer(lang),
    )

    result = text_processing_v2_5._normalize_with_official_backend(
        "GPT-5-nano 中文 xuan4",
        "en",
    )

    assert calls[0][0] == "zh"
    assert result == "GPT-five-nano 中文 XVAN4"


def test_tokenizer_resolves_native_checkpoints_subdirectory(tmp_path):
    tokenizer = tmp_path / "checkpoints" / "multilingual_zh_ja_yue_char_del.tiktoken"
    tokenizer.parent.mkdir()
    tokenizer.touch()

    assert resolve_indextts25_tokenizer_file(str(tmp_path)) == str(tokenizer)


@pytest.mark.parametrize(
    ("lang", "text", "expected"),
    [
        (
            "es",
            "En la actualidad se continúa utilizando una corona abierta.",
            "<|es|> EN LA ACTUALIDAD SE CONTINÚA UTILIZANDO UNA CORONA ABIERTA.",
        ),
        (
            "ar",
            "فكلُّها خياراتٌ اتَّخذتَها بنفسِك.",
            "<|ar|> فكلُّها خياراتٌ اتَّخذتَها بنفسِك.",
        ),
    ],
)
def test_official_spanish_and_arabic_cases_preserve_expected_text_contract(
    monkeypatch,
    lang,
    text,
    expected,
):
    captured = {}

    def fake_encode(value, *, model_dir, tokenizer_file):
        captured["text"] = value
        return [1]

    monkeypatch.setattr(
        text_processing_v2_5,
        "encode_indextts25_text",
        fake_encode,
    )

    _, lang_id = prepare_indextts25_text(
        text,
        lang=lang,
        model_dir="/model",
        text_normalization=False,
    )

    assert captured["text"] == expected
    assert lang_id == LANGUAGE_DICT[lang]
